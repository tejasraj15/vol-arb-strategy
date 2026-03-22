"""
DS3M → Vol Arb Integration

This module replaces:
  - GARCH (vol forecasting)       → DS3M vol_forecast_mean
  - XGBoost (regime blocking)     → DS3M regime_probs
  - Fixed exit thresholds         → DS3M regime-adaptive thresholds
  - Fixed position sizing         → DS3M probability-weighted sizing

It plugs into your existing main.py by replacing the forecast + regime calls.
"""

import numpy as np


class DS3MSignalGenerator:
    """
    Wraps a trained DS3M model and produces trading signals
    for your short vol strategy.

    Usage:
        signal_gen = DS3MSignalGenerator(model, device, config)
        signal = signal_gen.generate_signal(x_history, y_history, current_iv)
    """

    def __init__(self, model, device, config=None):
        """
        Args:
            model:   trained DS3M model instance
            device:  torch device
            config:  dict of strategy parameters (uses defaults if None)
        """
        self.model = model
        self.device = device

        # Default config - tune these to your strategy
        self.config = config or {
            # Entry
            'min_iv_spread': 0.02,          # minimum IV - forecast spread to enter
            'min_iv_percentile': 0.75,      # minimum IV percentile to enter
            'min_regime_confidence': 0.60,  # minimum P(calm) to enter
            'min_regime_stability': 0.85,   # minimum Gamma[calm,calm] to enter

            # Position sizing
            'base_position_size': 0.12,     # 12% of capital
            'max_position_size': 0.15,      # cap at 15%
            'min_position_size': 0.02,      # floor at 2%

            # Exit thresholds (base values, scaled by regime)
            'base_tp_threshold': 0.03,      # take profit if IV drops 3 vol points
            'base_sl_threshold': 0.02,      # stop loss if IV rises 2 vol points
            'max_loss_multiple': 2.0,       # exit if loss > 2x premium
            'max_holding_days': 14,         # mandatory exit

            # Forecast
            'n_mc_samples': 100,            # Monte Carlo samples for forecast
            'forecast_steps': 1,            # steps ahead (1 for daily)
        }
        self.calm_idx = 0
        self.stress_idx = 1

    def calibrate_regimes(self, x_history, y_history, y_std=None):
        """
        Infer which latent regime corresponds to lower realized vol.

        Args:
            x_history: training inputs, shape (seq_len, batch, x_dim)
            y_history: training targets, shape (seq_len, batch, y_dim)
            y_std: unused compatibility argument kept for older call sites

        Returns:
            tuple (calm_idx, stress_idx)
        """
        del y_std

        self.model.eval()
        with np.errstate(invalid='ignore'):
            pass

        import torch

        with torch.no_grad():
            out = self.model(x_history, y_history)

        # Skip the t=0 initialization entry and use posterior probabilities.
        regime_scores = out['d_posterior'][1:]
        target_steps = y_history[:, :, 0]

        weighted_means = []
        for regime_idx in range(self.model.d_dim):
            weights = torch.cat([step[:, regime_idx] for step in regime_scores], dim=0)
            targets = target_steps.reshape(-1)
            weight_sum = weights.sum()

            if float(weight_sum) <= 0:
                weighted_means.append(float('inf'))
                continue

            mean_vol = (weights * targets).sum() / weight_sum
            weighted_means.append(float(mean_vol))

        order = np.argsort(weighted_means)
        self.calm_idx = int(order[0])
        self.stress_idx = int(order[-1])
        return self.calm_idx, self.stress_idx

    def generate_signal(self, x_history, y_history, current_iv, iv_percentile,
                        days_to_earnings=None, current_positions=None):
        """
        Produce a complete trading signal from DS3M.

        This is the function you call daily in your main.py loop,
        replacing both the GARCH forecast call and the XGBoost regime check.

        Args:
            x_history:      recent input features as tensor (seq_len, 1, x_dim)
            y_history:      recent realized vol as tensor (seq_len, 1, y_dim)
            current_iv:     current implied volatility (float)
            iv_percentile:  where IV sits in its 252-day history (float, 0-1)
            days_to_earnings: days until next earnings (int or None)
            current_positions: number of open positions (int or None)

        Returns:
            dict with all signal components
        """
        cfg = self.config

        # ---- Run DS3M forecast ----
        forecast = self.model.forecast(
            x_history, y_history,
            steps=cfg['forecast_steps'],
            n_samples=cfg['n_mc_samples']
        )

        # Extract key outputs (squeeze batch dim since we're doing one stock)
        vol_forecast = float(forecast['vol_forecast_mean'][0, 0, 0])
        vol_uncertainty = float(forecast['vol_forecast_std'][0, 0, 0])
        vol_q05 = float(forecast['vol_forecast_q05'][0, 0, 0])
        vol_q95 = float(forecast['vol_forecast_q95'][0, 0, 0])
        regime_probs = forecast['regime_probs'][0, 0, :]  # (d_dim,)
        trans_matrix = forecast['transition_matrix']

        # ---- Interpret regimes ----
        # Regime labels can be calibrated after training.
        p_calm = float(regime_probs[self.calm_idx])
        p_stress = float(regime_probs[self.stress_idx])
        regime_stability = float(trans_matrix[self.calm_idx, self.calm_idx])

        # ---- IV spread ----
        iv_spread = current_iv - vol_forecast

        # ---- Entry decision ----
        entry_conditions = {
            'iv_spread_ok': iv_spread > cfg['min_iv_spread'],
            'iv_percentile_ok': iv_percentile > cfg['min_iv_percentile'],
            'regime_calm': p_calm > cfg['min_regime_confidence'],
            'regime_stable': regime_stability > cfg['min_regime_stability'],
            'not_near_earnings': (days_to_earnings is None or days_to_earnings > 14),
        }
        should_enter = all(entry_conditions.values())

        # ---- Position sizing (smooth, not binary) ----
        raw_size = cfg['base_position_size'] * p_calm * regime_stability
        position_size = np.clip(raw_size, cfg['min_position_size'], cfg['max_position_size'])

        # If regime is clearly stressed, size goes near zero naturally
        # No hard XGBoost block needed

        # ---- Adaptive exit thresholds ----
        # Scale thresholds by forecast uncertainty relative to average
        # In calm regime: tighter thresholds (take profits faster)
        # In stress regime: wider thresholds (avoid false exits)
        uncertainty_ratio = max(vol_uncertainty / max(vol_forecast, 0.01), 0.5)
        uncertainty_ratio = min(uncertainty_ratio, 2.0)  # cap scaling

        tp_threshold = cfg['base_tp_threshold'] * (1.0 / uncertainty_ratio)
        sl_threshold = cfg['base_sl_threshold'] * uncertainty_ratio

        # ---- Regime shift warning ----
        # If P(stress) is rising, warn even if still in calm
        regime_shift_risk = p_stress  # simple: just use stress probability
        # Could also use: 1 - regime_stability

        return {
            # Core forecast (replaces GARCH)
            'vol_forecast': vol_forecast,
            'vol_uncertainty': vol_uncertainty,
            'vol_confidence_interval': (vol_q05, vol_q95),
            'iv_spread': iv_spread,

            # Regime info (replaces XGBoost)
            'p_calm': p_calm,
            'p_stress': p_stress,
            'regime_stability': regime_stability,
            'regime_shift_risk': regime_shift_risk,
            'transition_matrix': trans_matrix,

            # Trading decision
            'should_enter': should_enter,
            'entry_conditions': entry_conditions,
            'position_size': position_size if should_enter else 0.0,

            # Adaptive exits
            'tp_threshold': tp_threshold,
            'sl_threshold': sl_threshold,
            'max_loss_multiple': cfg['max_loss_multiple'],
            'max_holding_days': cfg['max_holding_days'],
        }

    def check_exit(self, signal, days_held, current_pnl, premium_collected,
                   iv_change_since_entry):
        """
        Check whether an existing position should be exited.

        Call this daily for each open position.

        Args:
            signal:                output from generate_signal()
            days_held:             days since entry
            current_pnl:           current P&L of the position
            premium_collected:     premium received at entry
            iv_change_since_entry: IV now minus IV at entry (positive = IV rose)

        Returns:
            dict with exit decision and reason
        """
        cfg = self.config

        # ---- Mandatory time exit ----
        if days_held >= cfg['max_holding_days']:
            return {'should_exit': True, 'reason': 'max_holding_days'}

        # ---- Loss limit ----
        if current_pnl < -cfg['max_loss_multiple'] * premium_collected:
            return {'should_exit': True, 'reason': 'max_loss_exceeded'}

        # ---- Regime shift exit ----
        # If regime flips to stress, exit regardless of P&L
        if signal['p_stress'] > 0.75:
            return {'should_exit': True, 'reason': 'regime_shift_to_stress'}

        # ---- Adaptive IV-based exits ----
        # Thresholds scale with regime uncertainty
        tp = signal['tp_threshold']
        sl = signal['sl_threshold']

        # Days 7-9
        if 7 <= days_held <= 9:
            if iv_change_since_entry <= -tp:
                return {'should_exit': True, 'reason': 'take_profit_early'}
            if iv_change_since_entry >= sl:
                return {'should_exit': True, 'reason': 'stop_loss_early'}

        # Days 10-13
        if 10 <= days_held <= 13:
            # Tighter take profit, wider stop loss as expiry approaches
            if iv_change_since_entry <= -(tp * 0.67):
                return {'should_exit': True, 'reason': 'take_profit_late'}
            if iv_change_since_entry >= (sl * 1.5):
                return {'should_exit': True, 'reason': 'stop_loss_late'}

        return {'should_exit': False, 'reason': None}


class DS3MRegimeMonitor:
    """
    Tracks regime state over time for logging and analysis.
    Run this alongside your strategy to build a picture of regime history.
    """

    def __init__(self):
        self.history = []

    def update(self, date, signal):
        self.history.append({
            'date': date,
            'p_calm': signal['p_calm'],
            'p_stress': signal['p_stress'],
            'regime_stability': signal['regime_stability'],
            'vol_forecast': signal['vol_forecast'],
            'vol_uncertainty': signal['vol_uncertainty'],
            'iv_spread': signal['iv_spread'],
            'regime_shift_risk': signal['regime_shift_risk'],
        })

    def get_regime_durations(self, threshold=0.5):
        """How long each regime has lasted (for analysis)."""
        regimes = ['calm' if h['p_calm'] > threshold else 'stress'
                   for h in self.history]
        durations = []
        current = regimes[0]
        count = 1
        for r in regimes[1:]:
            if r == current:
                count += 1
            else:
                durations.append((current, count))
                current = r
                count = 1
        durations.append((current, count))
        return durations

    def to_dataframe(self):
        """Convert history to pandas DataFrame for analysis."""
        import pandas as pd
        return pd.DataFrame(self.history).set_index('date')
