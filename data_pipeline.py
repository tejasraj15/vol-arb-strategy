"""
Data Pipeline: 4-Hour Absolute Log Returns (Regular Hours Only)

Aggregates hourly bars into 4-hour bars using regular trading hours:
  Bar 1: 13:00-16:00 UTC  (9 AM - 12 PM ET)  morning session
  Bar 2: 17:00-20:00 UTC  (1 PM - 4 PM ET)   afternoon session

y_t = |log return| of the 4-hour bar (absolute, always positive)
x_t = lagged |log return| (x_dim = 1)

RV is computed from absolute returns for comparison.
"""

import pandas as pd
import numpy as np
import os


def load_and_build_4h_bars(filepath):
    """
    Load hourly CSV, filter to regular hours, aggregate into 4-hour bars.

    Regular hours (UTC): 13:00-20:00
    Bar 1: 13:00-16:00 (use close at 16:00, open at 13:00)
    Bar 2: 17:00-20:00 (use close at 20:00, open at 17:00)
    """
    df = pd.read_csv(filepath)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df['hour'] = df['ts_event'].dt.hour
    df['date'] = df['ts_event'].dt.date

    # Keep only regular hours: 13:00-20:00 UTC
    df = df[(df['hour'] >= 13) & (df['hour'] <= 20)].copy()

    # Assign each bar to a 4-hour session
    # Hours 13,14,15,16 -> session 1 (morning)
    # Hours 17,18,19,20 -> session 2 (afternoon)
    df['session'] = np.where(df['hour'] <= 16, 1, 2)

    # Aggregate to 4-hour bars
    bars = df.groupby(['date', 'session']).agg(
        ts_start=('ts_event', 'first'),
        ts_end=('ts_event', 'last'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        n_hours=('close', 'count'),
    ).reset_index()

    bars = bars.sort_values('ts_start').reset_index(drop=True)

    # Only keep bars with at least 3 hourly bars (avoid partial sessions)
    bars = bars[bars['n_hours'] >= 3].copy()

    # Compute log return of the 4-hour bar
    bars['log_return'] = np.log(bars['close'] / bars['close'].shift(1))

    # Absolute log return
    bars['abs_log_return'] = bars['log_return'].abs()

    bars = bars.dropna(subset=['abs_log_return']).reset_index(drop=True)

    return bars


def prepare_ds3m_data(data_dir, symbols=None, seq_len=20,
                      train_start=None, train_end=None,
                      test_start=None, test_end=None):
    """
    4-hour absolute log returns -> DS3M tensors.

    x_t = lagged |log return| (x_dim = 1)
    y_t = |log return| (y_dim = 1)

    Args:
        data_dir:    directory containing *_1h.csv files
        symbols:     list of tickers
        seq_len:     subsequence length in 4-hour bars
        train_start: start date for training (str 'YYYY-MM-DD' or None)
        train_end:   end date for training (str 'YYYY-MM-DD' or None)
        test_start:  start date for testing (str 'YYYY-MM-DD' or None)
        test_end:    end date for testing (str 'YYYY-MM-DD' or None)
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'DIS', 'PFE', 'UNH', 'WMT', 'XOM']

    all_stock_data = {}

    for sym in symbols:
        filepath = os.path.join(data_dir, f'{sym}_1h.csv')
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue

        print(f"Processing {sym} (4-hour |log returns|, regular hours only)...")

        bars = load_and_build_4h_bars(filepath)

        # y_t = absolute log return
        y = bars['abs_log_return'].values

        # x_t = lagged absolute log return
        x = bars['abs_log_return'].shift(1).values.reshape(-1, 1)

        # Timestamps and dates for filtering
        timestamps = bars['ts_start'].values
        bar_dates = pd.to_datetime(bars['ts_start']).dt.date.values

        # Drop NaN from lag
        valid = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        timestamps = timestamps[valid]
        bar_dates = bar_dates[valid]

        # ---- Date-based train/test split ----
        all_dates = pd.to_datetime(pd.Series(bar_dates))

        if train_start is None:
            train_start_dt = all_dates.min()
        else:
            train_start_dt = pd.Timestamp(train_start)

        if train_end is None:
            # Default: first 60% of dates
            unique_dates = sorted(set(bar_dates))
            train_end_dt = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.6)])
        else:
            train_end_dt = pd.Timestamp(train_end)

        if test_start is None:
            test_start_dt = train_end_dt + pd.Timedelta(days=1)
        else:
            test_start_dt = pd.Timestamp(test_start)

        if test_end is None:
            test_end_dt = all_dates.max()
        else:
            test_end_dt = pd.Timestamp(test_end)

        train_mask = (all_dates >= train_start_dt) & (all_dates <= train_end_dt)
        test_mask = (all_dates >= test_start_dt) & (all_dates <= test_end_dt)

        train_indices = np.where(train_mask.values)[0]
        test_indices = np.where(test_mask.values)[0]

        # Normalize using training data only
        y_train_raw = y[train_indices]
        x_train_raw = x[train_indices]

        y_mean, y_std = y_train_raw.mean(), y_train_raw.std()
        x_mean, x_std = x_train_raw.mean(axis=0), x_train_raw.std(axis=0)
        x_std[x_std == 0] = 1.0

        y_norm = (y - y_mean) / y_std
        x_norm = (x - x_mean) / x_std

        # Slice train sequences
        train_X, train_Y = [], []
        for i in range(len(train_indices) - seq_len):
            idx_start = train_indices[i]
            idx_end = train_indices[i] + seq_len
            if idx_end <= train_indices[-1] + 1:
                train_X.append(x_norm[idx_start:idx_end])
                train_Y.append(y_norm[idx_start:idx_end].reshape(-1, 1))

        if len(train_X) > 0:
            train_X = np.stack(train_X, axis=1)
            train_Y = np.stack(train_Y, axis=1)
        else:
            print(f"  Warning: no training sequences for {sym}")
            continue

        # Test: store full normalized arrays for rolling window inference
        all_stock_data[sym] = {
            'timestamps': timestamps,
            'dates': timestamps,
            'bar_dates': bar_dates,

            'y_mean': y_mean,
            'y_std': y_std,
            'x_mean': x_mean,
            'x_std': x_std,

            'y_full': y_norm,
            'x_full': x_norm,
            'y_raw': y,
            'x_raw': x.flatten(),
            'log_returns_raw': bars['log_return'].values[valid],

            'train_x': train_X,
            'train_y': train_Y,

            'train_start': train_start_dt.date(),
            'train_end': train_end_dt.date(),
            'test_start': test_start_dt.date(),
            'test_end': test_end_dt.date(),

            'test_indices': test_indices,

            'x_dim': 1,
            'y_dim': 1,
            'n_train': train_X.shape[1],
            'n_test': len(test_indices),
            'seq_len': seq_len,
        }

        print(f"  {sym}: {len(y):,} 4h bars (2 per day, regular hours)")
        print(f"  |Log return| range: {y.min():.6f} to {y.max():.4f} (mean {y.mean():.6f})")
        print(f"  Time span: {pd.Timestamp(timestamps[0]).date()} to {pd.Timestamp(timestamps[-1]).date()}")
        print(f"  Train: {train_start_dt.date()} to {train_end_dt.date()} ({train_X.shape[1]:,} sequences)")
        print(f"  Test:  {test_start_dt.date()} to {test_end_dt.date()} ({len(test_indices):,} bars)")

    return all_stock_data


def get_live_window(stock_data, idx, seq_len=20):
    """Get the most recent seq_len bars of normalized data."""
    import torch
    x = stock_data['x_full'][idx - seq_len:idx]
    y = stock_data['y_full'][idx - seq_len:idx]
    x_tensor = torch.FloatTensor(x).unsqueeze(1)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1)).unsqueeze(1)
    return x_tensor, y_tensor


def unnormalize_abs_return(value, stock_data):
    """Convert normalized |log return| back to actual."""
    return value * stock_data['y_std'] + stock_data['y_mean']


def compute_daily_rv(dates, abs_returns, annualize=True):
    """
    Compute daily realized vol from absolute returns.
    Groups by date, computes RV = sqrt(pi/2) * mean(|r|) * sqrt(n) per day.
    Annualizes by * sqrt(252) if requested.
    """
    df = pd.DataFrame({'date': dates, 'abs_ret': abs_returns})
    daily = df.groupby('date').agg(
        rv_abs=('abs_ret', lambda x: np.sqrt(np.pi / 2) * x.mean() * np.sqrt(len(x))),
        n_bars=('abs_ret', 'count'),
        sum_sq=('abs_ret', lambda x: np.sqrt(np.sum(x ** 2))),
    )
    if annualize:
        daily['rv_abs'] = daily['rv_abs'] * np.sqrt(252)
        daily['rv_sq'] = daily['sum_sq'] * np.sqrt(252)
    else:
        daily['rv_sq'] = daily['sum_sq']
    return daily


if __name__ == '__main__':
    data = prepare_ds3m_data('/mnt/user-data/uploads', ['AAPL'], seq_len=20)
    for sym, d in data.items():
        print(f"\n{sym}:")
        print(f"  x_dim: {d['x_dim']}, y_dim: {d['y_dim']}")
        print(f"  train_x shape: {d['train_x'].shape}")
        print(f"  n_train: {d['n_train']:,}, n_test: {d['n_test']:,}")