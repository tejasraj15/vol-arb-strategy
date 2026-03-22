"""
DS3M Vol Arb - 4-Hour Absolute Log Returns

Trains on 4-hour |log returns| from regular trading hours.
Computes RV from forecasted |returns| and compares to actual RV.

Usage:
    python run.py --train AAPL
    python run.py --test AAPL
    python run.py --train AAPL --train-start 2021-06-01 --train-end 2024-12-31
    python run.py --test AAPL --test-start 2025-01-01 --test-end 2026-03-11
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os

from ds3m_model import DS3M
from training import train_ds3m
from data_pipeline import (
    prepare_ds3m_data, get_live_window, unnormalize_abs_return, compute_daily_rv
)
from volarb_integration import DS3MSignalGenerator

# ==============================================================
# CONFIG
# ==============================================================

DATA_DIR = '.'
MODEL_DIR = './models'
SYMBOLS = ['AAPL', 'MSFT', 'DIS', 'PFE', 'UNH', 'WMT', 'XOM']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for candidate in ['.', os.environ.get('VOL_ARB_DATA_DIR', '')]:
    if candidate and os.path.exists(os.path.join(candidate, 'AAPL_1h.csv')):
        DATA_DIR = candidate
        break

print(f"Using data directory: {os.path.abspath(DATA_DIR)}")
print(f"Runtime device: {DEVICE}")

HPARAMS = {
    'x_dim': 1,        # lagged |log return|
    'y_dim': 1,        # |log return|
    'h_dim': 32,
    'z_dim': 4,
    'd_dim': 2,
    'n_layers': 1,
    'seq_len': 20,     # 20 four-hour bars = 10 trading days
}

TRAIN_CFG = {
    'n_epochs': 200,
    'batch_size': 256,
    'lr': 0.0005,
    'patience': 40,
}


# ==============================================================
# TRAIN
# ==============================================================

def train_stock(sym, stock_data):
    d = stock_data[sym]
    print(f"\n{'='*60}")
    print(f"TRAINING DS3M FOR {sym} (4-hour |log returns|)")
    print(f"{'='*60}")
    print(f"  Train period:    {d['train_start']} to {d['train_end']}")
    print(f"  Train sequences: {d['n_train']:,}")
    print(f"  |Return| std:    {d['y_std']:.6f}")

    to_t = lambda a: torch.FloatTensor(a).to(DEVICE)
    train_x, train_y = to_t(d['train_x']), to_t(d['train_y'])

    # last 15% of training sequences for validation
    n = train_x.shape[1]
    split = int(n * 0.85)
    val_x = train_x[:, split:, :]
    val_y = train_y[:, split:, :]
    train_x = train_x[:, :split, :]
    train_y = train_y[:, :split, :]

    model = DS3M(
        x_dim=HPARAMS['x_dim'], y_dim=HPARAMS['y_dim'],
        h_dim=HPARAMS['h_dim'], z_dim=HPARAMS['z_dim'],
        d_dim=HPARAMS['d_dim'], n_layers=HPARAMS['n_layers'],
        device=DEVICE,
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {params:,}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f'ds3m_{sym}.pt')

    history = train_ds3m(
        model, train_x, train_y, val_x, val_y,
        save_path=save_path, **TRAIN_CFG,
    )

    return model, history


# ==============================================================
# TEST: Forecast |returns| -> compute RV -> compare
# ==============================================================

def test_stock(sym, model, stock_data):
    d = stock_data[sym]
    seq_len = HPARAMS['seq_len']
    test_indices = d['test_indices']

    if len(test_indices) == 0:
        print(f"No test data for {sym}")
        return None

    # Only use test indices where we have enough history for a window
    usable = test_indices[test_indices >= seq_len]
    n_test = len(usable)

    # Calibrate regimes using training data
    to_t = lambda a: torch.FloatTensor(a).to(DEVICE)
    model.eval()

    temp_gen = DS3MSignalGenerator(model, DEVICE, config={
        'n_mc_samples': 50, 'forecast_steps': 1,
        'min_iv_spread': 0, 'min_iv_percentile': 0,
        'min_regime_confidence': 0, 'min_regime_stability': 0,
        'base_position_size': 0.12, 'max_position_size': 0.15,
        'min_position_size': 0.02, 'base_tp_threshold': 0.03,
        'base_sl_threshold': 0.02, 'max_loss_multiple': 2.0,
        'max_holding_days': 14,
    })
    temp_gen.calibrate_regimes(to_t(d['train_x']), to_t(d['train_y']))
    calm_idx = temp_gen.calm_idx
    stress_idx = temp_gen.stress_idx

    print(f"\n{'='*70}")
    print(f"  FORECAST EVALUATION: {sym} (4-hour |log returns|)")
    print(f"  Test period:  {d['test_start']} to {d['test_end']}")
    print(f"  Test bars:    {n_test:,} (2 bars/day)")
    print(f"  Regime:       regime {calm_idx} = CALM, regime {stress_idx} = STRESS")
    print(f"{'='*70}")

    results = []

    for i, bar_idx in enumerate(usable):
        x_tensor, y_tensor = get_live_window(d, bar_idx, seq_len)
        x_tensor = x_tensor.to(DEVICE)
        y_tensor = y_tensor.to(DEVICE)

        actual_abs_ret = d['y_raw'][bar_idx]

        forecast = model.forecast(x_tensor, y_tensor, steps=1, n_samples=50)

        fc_norm = float(forecast['vol_forecast_mean'][0, 0, 0])
        fc_std_norm = float(forecast['vol_forecast_std'][0, 0, 0])
        regime_probs = forecast['regime_probs'][0, 0, :]
        trans = forecast['transition_matrix']

        fc_abs_ret = unnormalize_abs_return(fc_norm, d)
        fc_abs_ret = max(fc_abs_ret, 0)  # absolute returns can't be negative
        fc_std = fc_std_norm * d['y_std']

        p_calm = float(regime_probs[calm_idx])
        p_stress = float(regime_probs[stress_idx])
        regime = 'CALM' if p_calm > 0.5 else 'STRESS'

        ts = pd.Timestamp(d['timestamps'][bar_idx])

        results.append({
            'timestamp': ts,
            'date': ts.date(),
            'session': 'AM' if ts.hour <= 16 else 'PM',
            'actual_abs_return': actual_abs_ret,
            'forecast_abs_return': fc_abs_ret,
            'forecast_std': fc_std,
            'error': fc_abs_ret - actual_abs_ret,
            'abs_error': abs(fc_abs_ret - actual_abs_ret),
            'p_calm': p_calm,
            'p_stress': p_stress,
            'regime': regime,
        })

        if i % 100 == 0:
            print(f"  {ts} | |Ret|: {actual_abs_ret:.4f} | Forecast: {fc_abs_ret:.4f} | "
                  f"Error: {fc_abs_ret - actual_abs_ret:+.4f} | {regime:6s} P(calm): {p_calm:.2f}")

    df = pd.DataFrame(results)

    # ---- 4-Hour Bar Accuracy ----
    print(f"\n{'='*70}")
    print(f"  4-HOUR |LOG RETURN| FORECAST ACCURACY")
    print(f"{'='*70}")

    mae = df['abs_error'].mean()
    rmse = np.sqrt((df['error'] ** 2).mean())
    bias = df['error'].mean()
    corr = df[['actual_abs_return', 'forecast_abs_return']].corr().iloc[0, 1]

    ss_res = ((df['actual_abs_return'] - df['forecast_abs_return']) ** 2).sum()
    ss_tot = ((df['actual_abs_return'] - df['actual_abs_return'].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n = len(df)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - 2)) if n > 2 else 0

    print(f"  Bars:              {len(df):,}")
    print(f"  MAE:               {mae:.6f}")
    print(f"  RMSE:              {rmse:.6f}")
    print(f"  R²:                {r2:.4f}")
    print(f"  Adjusted R²:       {adj_r2:.4f}")
    print(f"  Bias:              {bias:+.6f}")
    print(f"  Correlation:       {corr:.4f}")

    # ---- Daily RV Comparison ----
    print(f"\n{'='*70}")
    print(f"  DAILY REALIZED VOL COMPARISON")
    print(f"{'='*70}")

    actual_daily = compute_daily_rv(df['date'], df['actual_abs_return'], annualize=True)
    forecast_daily = compute_daily_rv(df['date'], df['forecast_abs_return'], annualize=True)

    # Merge on date
    merged = actual_daily[['rv_abs', 'rv_sq']].rename(
        columns={'rv_abs': 'actual_rv_abs', 'rv_sq': 'actual_rv_sq'}
    ).join(
        forecast_daily[['rv_abs', 'rv_sq']].rename(
            columns={'rv_abs': 'forecast_rv_abs', 'rv_sq': 'forecast_rv_sq'}
        ),
        how='inner'
    )

    # Add regime per day
    daily_regime = df.groupby('date')['p_calm'].mean()
    merged['p_calm'] = daily_regime
    merged['regime'] = merged['p_calm'].apply(lambda x: 'CALM' if x > 0.5 else 'STRESS')

    if len(merged) > 0:
        # RV from absolute returns method
        rv_err = merged['forecast_rv_abs'] - merged['actual_rv_abs']
        rv_mae = rv_err.abs().mean()
        rv_rmse = np.sqrt((rv_err ** 2).mean())
        rv_bias = rv_err.mean()
        rv_corr = merged[['actual_rv_abs', 'forecast_rv_abs']].corr().iloc[0, 1]

        ss_res_rv = ((merged['actual_rv_abs'] - merged['forecast_rv_abs']) ** 2).sum()
        ss_tot_rv = ((merged['actual_rv_abs'] - merged['actual_rv_abs'].mean()) ** 2).sum()
        r2_rv = 1 - ss_res_rv / ss_tot_rv if ss_tot_rv > 0 else 0
        n_rv = len(merged)
        adj_r2_rv = 1 - ((1 - r2_rv) * (n_rv - 1) / (n_rv - 2)) if n_rv > 2 else 0

        print(f"\n  Method: RV from |returns| (annualized)")
        print(f"  Trading days:      {len(merged)}")
        print(f"  Avg actual RV:     {merged['actual_rv_abs'].mean():.1%}")
        print(f"  Avg forecast RV:   {merged['forecast_rv_abs'].mean():.1%}")
        print(f"  MAE:               {rv_mae:.4f} ({rv_mae*100:.1f}%)")
        print(f"  RMSE:              {rv_rmse:.4f} ({rv_rmse*100:.1f}%)")
        print(f"  R²:                {r2_rv:.4f}")
        print(f"  Adjusted R²:       {adj_r2_rv:.4f}")
        print(f"  Bias:              {rv_bias:+.4f} ({rv_bias*100:+.1f}%)")
        print(f"  Correlation:       {rv_corr:.4f}")

        # RV from squared returns method
        rv_sq_err = merged['forecast_rv_sq'] - merged['actual_rv_sq']
        rv_sq_corr = merged[['actual_rv_sq', 'forecast_rv_sq']].corr().iloc[0, 1]
        ss_res_sq = ((merged['actual_rv_sq'] - merged['forecast_rv_sq']) ** 2).sum()
        ss_tot_sq = ((merged['actual_rv_sq'] - merged['actual_rv_sq'].mean()) ** 2).sum()
        r2_sq = 1 - ss_res_sq / ss_tot_sq if ss_tot_sq > 0 else 0

        print(f"\n  Cross-check: RV from squared returns")
        print(f"  R²:                {r2_sq:.4f}")
        print(f"  Correlation:       {rv_sq_corr:.4f}")

        # Per-regime
        calm_d = merged[merged['regime'] == 'CALM']
        stress_d = merged[merged['regime'] == 'STRESS']

        print(f"\n{'='*70}")
        print(f"  REGIME DETECTION")
        print(f"{'='*70}")
        print(f"  Days in CALM:      {len(calm_d)} ({len(calm_d)/len(merged):.1%})")
        print(f"  Days in STRESS:    {len(stress_d)} ({len(stress_d)/len(merged):.1%})")

        def regime_r2(sub):
            if len(sub) < 3:
                return np.nan
            ss_r = ((sub['actual_rv_abs'] - sub['forecast_rv_abs']) ** 2).sum()
            ss_t = ((sub['actual_rv_abs'] - sub['actual_rv_abs'].mean()) ** 2).sum()
            return 1 - ss_r / ss_t if ss_t > 0 else 0

        if len(calm_d) > 0:
            print(f"\n  CALM:")
            print(f"    Avg actual RV:   {calm_d['actual_rv_abs'].mean():.1%}")
            print(f"    Avg forecast RV: {calm_d['forecast_rv_abs'].mean():.1%}")
            print(f"    R²:              {regime_r2(calm_d):.4f}")

        if len(stress_d) > 0:
            print(f"\n  STRESS:")
            print(f"    Avg actual RV:   {stress_d['actual_rv_abs'].mean():.1%}")
            print(f"    Avg forecast RV: {stress_d['forecast_rv_abs'].mean():.1%}")
            print(f"    R²:              {regime_r2(stress_d):.4f}")

        # Transition matrix
        trans = model.get_transition_matrix().detach().cpu().numpy()
        calm_dur = 1.0 / max(1.0 - trans[calm_idx, calm_idx], 0.001)
        stress_dur = 1.0 / max(1.0 - trans[stress_idx, stress_idx], 0.001)
        print(f"\n  Transition matrix:")
        print(f"    P(calm->calm):   {trans[calm_idx, calm_idx]:.3f} -> ~{calm_dur:.0f} bars ({calm_dur/2:.0f} days)")
        print(f"    P(stress->str):  {trans[stress_idx, stress_idx]:.3f} -> ~{stress_dur:.0f} bars ({stress_dur/2:.0f} days)")

        # Sample daily output
        print(f"\n  Sample daily RV comparison:")
        sample = merged.iloc[::max(1, len(merged) // 15)]
        for date, row in sample.iterrows():
            print(f"    {date} | Actual: {row['actual_rv_abs']:.1%} | "
                  f"Forecast: {row['forecast_rv_abs']:.1%} | "
                  f"Error: {row['forecast_rv_abs'] - row['actual_rv_abs']:+.1%} | {row['regime']}")

        # Worst days
        merged['rv_abs_error'] = (merged['forecast_rv_abs'] - merged['actual_rv_abs']).abs()
        worst = merged.nlargest(5, 'rv_abs_error')
        print(f"\n  Worst forecast days:")
        for date, row in worst.iterrows():
            print(f"    {date} | Actual: {row['actual_rv_abs']:.1%} | "
                  f"Forecast: {row['forecast_rv_abs']:.1%} | {row['regime']}")

    # Save
    csv_path = f'results_ds3m_{sym.lower()}_4h_absret.csv'
    df.to_csv(csv_path, index=False)
    rv_path = f'results_ds3m_{sym.lower()}_4h_daily_rv.csv'
    merged.to_csv(rv_path)
    print(f"\n  Bar results saved to {csv_path}")
    print(f"  Daily RV saved to {rv_path}")

    return df, merged


# ==============================================================
# CLI
# ==============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DS3M 4-Hour |Log Returns|')
    parser.add_argument('--train', type=str, default=None, help='Train model for symbol')
    parser.add_argument('--test', type=str, default=None, help='Test model for symbol')
    parser.add_argument('--all', action='store_true', help='Train all stocks')

    # Configurable date ranges
    parser.add_argument('--train-start', type=str, default=None, help='Training start date YYYY-MM-DD')
    parser.add_argument('--train-end', type=str, default=None, help='Training end date YYYY-MM-DD')
    parser.add_argument('--test-start', type=str, default=None, help='Test start date YYYY-MM-DD')
    parser.add_argument('--test-end', type=str, default=None, help='Test end date YYYY-MM-DD')

    parser.add_argument('--data-dir', type=str, default=None, help='Directory with *_1h.csv files')
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = args.data_dir

    date_args = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'test_start': args.test_start,
        'test_end': args.test_end,
    }

    if args.train:
        sym = args.train.upper()
        stock_data = prepare_ds3m_data(DATA_DIR, [sym], seq_len=HPARAMS['seq_len'], **date_args)
        if sym not in stock_data:
            print(f"Error: no data for {sym}")
            exit(1)
        train_stock(sym, stock_data)

    elif args.test:
        sym = args.test.upper()
        stock_data = prepare_ds3m_data(DATA_DIR, [sym], seq_len=HPARAMS['seq_len'], **date_args)
        if sym not in stock_data:
            print(f"Error: no data for {sym}")
            exit(1)
        model = DS3M(
            x_dim=HPARAMS['x_dim'], y_dim=HPARAMS['y_dim'],
            h_dim=HPARAMS['h_dim'], z_dim=HPARAMS['z_dim'],
            d_dim=HPARAMS['d_dim'], n_layers=HPARAMS['n_layers'],
            device=DEVICE,
        ).to(DEVICE)
        model_path = os.path.join(MODEL_DIR, f'ds3m_{sym}.pt')
        if not os.path.exists(model_path):
            print(f"Error: no trained model at {model_path}. Run --train {sym} first.")
            exit(1)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        test_stock(sym, model, stock_data)

    elif args.all:
        for sym in SYMBOLS:
            stock_data = prepare_ds3m_data(DATA_DIR, [sym], seq_len=HPARAMS['seq_len'], **date_args)
            if sym in stock_data:
                model, _ = train_stock(sym, stock_data)
                test_stock(sym, model, stock_data)

    else:
        print("Usage:")
        print("  python run.py --train AAPL")
        print("  python run.py --test AAPL")
        print("  python run.py --train AAPL --train-start 2021-06-01 --train-end 2024-12-31")
        print("  python run.py --test AAPL --test-start 2025-01-01 --test-end 2026-03-11")
        print("  python run.py --all")