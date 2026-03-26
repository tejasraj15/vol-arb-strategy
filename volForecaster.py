import os
import pickle
import torch
import numpy as np
import pandas as pd
from enum import Enum
from garch import garch_modelling
from preprocess_data import parse_data
from harcnn_train import train_harcnn, CNN_HAR_KS
from ds3m_train import train_ds3m
from harcnn_ridge import forecast_next_rv, fit_ridge_for_ticker

class Model(Enum):
    EGARCH = (1, 126, 21)
    HARCNN = (2, 252, 63)
    GNN    = (3, 126, 21)
    DS3M   = (4, 126, 21)

    def __init__(self, _id, train_window, refit_frequency):
        self.train_window = train_window
        self.refit_frequency = refit_frequency

    def __str__(self):
        labels = {
            Model.EGARCH: "EGARCH",
            Model.HARCNN: "HARCNN",
            Model.GNN: "GNN",
            Model.DS3M: "DS3M",
        }
        return labels[self]


class VolForecaster:

    def __init__(self, stock_data: pd.DataFrame, ticker: str = None, model: Model = Model.EGARCH, verbose=False):
        self.stock_data = stock_data
        self.ticker = ticker
        self.model = model
        self.train_window = model.train_window
        self.refit_frequency = model.refit_frequency

        if self.model == Model.HARCNN:
            if not (os.path.exists("cnn_har_ks_weights.pth") and os.path.exists("cnn_image_scaler.pkl")):
                train_harcnn()
            self._cnn_model = CNN_HAR_KS(dropout=0.5).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self._cnn_model.load_state_dict(torch.load("cnn_har_ks_weights.pth", map_location="cpu"))
            with open("cnn_image_scaler.pkl", "rb") as f:
                self._img_scaler = pickle.load(f)
            self._ridge_bundle = fit_ridge_for_ticker(ticker, self._cnn_model, self._img_scaler)

        if self.model == Model.DS3M:
            _device = torch.device("cpu")
            _ticker = (ticker or 'AAPL').upper()
            model_path = f"models/ds3m_{_ticker}.pt"
            scaler_path = f"models/ds3m_scaler_{_ticker}.pkl"
            self._ds3m_device = _device
            self._ds3m_seq_len = 20
            from ds3m_model import DS3M as _DS3M
            self._ds3m = _DS3M(1, 1, 32, 4, 2, 1, _device).to(_device)
            if not os.path.exists(model_path):
                print(f"[DS3M] No trained model at {model_path}, training now...")
                train_ds3m(_ticker)
            self._ds3m.load_state_dict(torch.load(model_path, map_location=_device))
            self._ds3m.eval()
            self._ds3m_scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self._ds3m_scaler = pickle.load(f)

        self._last_fit_date: pd.Timestamp | None = None
        self._cached_forecast: float | None = None

        # Rolling history for risk premium calibration
        self._forecast_history: list[float] = []
        self._market_iv_history: list[float] = []
        
        self.verbose = verbose

    def get_forecast(self, current_date: pd.Timestamp) -> float | None:
        if self._needs_refit(current_date):
            success = self._refit(current_date)
            if not success:
                return None

        return self._cached_forecast
    
    def _refit(self, current_date):
        if self.model == Model.EGARCH:
            return self._refit_egarch(current_date)
        elif self.model == Model.HARCNN:
            return self._refit_harcnn(current_date)
        elif self.model == Model.GNN:
            return self._refit_gnn(current_date)
        elif self.model == Model.DS3M:
            return self._refit_ds3m(current_date)

    def record_market_iv(self, market_iv: float) -> None:
        if self._cached_forecast is not None:
            self._forecast_history.append(self._cached_forecast)
            self._market_iv_history.append(market_iv)
            # Keep a rolling window
            if len(self._forecast_history) > 252:
                self._forecast_history = self._forecast_history[-252:]
                self._market_iv_history = self._market_iv_history[-252:]

    def get_forecast_diagnostics(self) -> dict:
        if not self._forecast_history:
            return {}
        f = np.array(self._forecast_history)
        m = np.array(self._market_iv_history)
        err = f - m
        return {
            "mean_forecast": f.mean(),
            "mean_market_iv": m.mean(),
            "mean_error": err.mean(),
            "rmse": np.sqrt((err ** 2).mean()),
            "correlation": np.corrcoef(f, m)[0, 1],
            "vol_risk_premium": self._vol_risk_premium(),
        }

    def _needs_refit(self, current_date: pd.Timestamp) -> bool:
        if self._cached_forecast is None:
            return True
        days_since = (current_date - self._last_fit_date).days
        return days_since >= self.refit_frequency

    def _refit_egarch(self, current_date: pd.Timestamp) -> bool:
        print(f"Refitting EGARCH: {current_date}")
        historical_data = (
            self.stock_data.loc[:current_date]
            .iloc[-(self.train_window + 1):-1]
        )
        log_returns = parse_data(historical_data, price_col='prc')

        try:
            _, base_forecast = garch_modelling(
                log_returns, "AR", "EGARCH", "t", validity_checks=False
            )
            ensemble = self._ensemble_forecast(log_returns, base_forecast)
            risk_premium = self._vol_risk_premium()
            self._cached_forecast = ensemble + risk_premium
            self._last_fit_date = current_date
            return True
        except Exception:
            return False

    def _ensemble_forecast(self, returns: pd.Series, base_forecast: float) -> float:
        # Median of GARCH, 21-day realised vol, and 30-day EWMA vol
        # will be replaced by new modules
        
        forecasts = [base_forecast]

        if len(returns) >= 21:
            forecasts.append(returns.tail(21).std() * np.sqrt(252))

        if len(returns) >= 30:
            forecasts.append(returns.ewm(span=30).std().iloc[-1] * np.sqrt(252))

        return float(np.median(forecasts))
    
    def _refit_harcnn(self, current_date: pd.Timestamp) -> bool:
        print(f"Refitting HARCNN: {current_date}")
        
        historical_data = (
            self.stock_data.loc[:current_date]
            .iloc[-(self.train_window + 1):-1]
        )
        try:
            rv_forecast = forecast_next_rv(
                historical_data, self._cnn_model, self._img_scaler, self._ridge_bundle
            )
            risk_premium = self._vol_risk_premium()
            # RV is variance; convert to annualised vol to match EGARCH output
            self._cached_forecast = np.sqrt(rv_forecast * 252) + risk_premium
            self._last_fit_date = current_date
            return True
        except Exception:
            return False
    
    def _refit_gnn(self, current_date: pd.Timestamp) -> bool:
        return True
    
    def _refit_ds3m(self, current_date: pd.Timestamp) -> bool:
        seq_len = self._ds3m_seq_len
        device = self._ds3m_device

        # Use stock data available up to current_date (no lookahead), same as HARCNN
        historical_data = self.stock_data.loc[:current_date].iloc[:-1]
        prices = historical_data["prc"].astype(float)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        abs_log_returns = np.abs(log_returns)

        if len(abs_log_returns) < seq_len + 1:
            if self.verbose:
                print(f"[DS3M] Not enough data at {current_date} (need {seq_len + 1}, have {len(abs_log_returns)})")
            return False

        # Build only the most recent sequence window
        x_vals = abs_log_returns.iloc[-(seq_len + 1):-1].values   # (seq_len,)
        y_vals = abs_log_returns.iloc[-seq_len:].values            # (seq_len,)

        scaler = self._ds3m_scaler
        if scaler is not None:
            x_vals = (x_vals - scaler['mean']) / scaler['std']
            y_vals = (y_vals - scaler['mean']) / scaler['std']

        X_input = torch.tensor(x_vals, dtype=torch.float32, device=device).view(seq_len, 1, 1)  # (seq_len, 1, 1)
        Y_input = torch.tensor(y_vals, dtype=torch.float32, device=device).view(seq_len, 1, 1)

        try:
            with torch.no_grad():
                out = self._ds3m.forecast(X_input, Y_input, steps=1, n_samples=100)
            raw_z = float(out['vol_forecast_mean'][-1][0][0])
            if scaler is not None:
                raw_forecast = abs(raw_z * scaler['std'] + scaler['mean'])
            else:
                raw_forecast = abs(raw_z)
            risk_premium = self._vol_risk_premium()
            self._cached_forecast = raw_forecast * np.sqrt(252) + risk_premium
            if self.verbose:
                print(f"  [DS3M] {current_date.date()} raw={raw_forecast:.4f}, annualised={self._cached_forecast:.1%}")
            self._last_fit_date = current_date
            return True
        except Exception as e:
            import traceback
            print(f"DS3M forecast failed: {e}")
            traceback.print_exc()
            return False

    def _vol_risk_premium(self, min_samples: int = 60) -> float:
        if len(self._forecast_history) < min_samples:
            return 0.03

        lookback = min(126, len(self._forecast_history))
        recent_forecasts = np.array(self._forecast_history[-lookback:])
        recent_markets = np.array(self._market_iv_history[-lookback:])

        premium = np.median(recent_markets - recent_forecasts)
        return float(np.clip(premium, 0.0, 0.10))