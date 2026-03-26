import os
import os
import pickle
import torch
import numpy as np
import pandas as pd
from enum import Enum
from garch import garch_modelling
from preprocess_data import parse_data
from ds3m_model import DS3M
from harcnn_train import train_harcnn, CNN_HAR_KS
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
        print(f"[DS3M] Using DS3M model for forecasting on {self.ticker} at {current_date}")
        # DS3M hyperparameters
        x_dim = 1
        y_dim = 1
        h_dim = 32
        z_dim = 4
        d_dim = 2
        n_layers = 1
        seq_len = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dynamically select the correct data file for the ticker
        ticker = self.ticker or 'AAPL'
        data_path = f"data/{ticker.upper()}_stock_prices_2020_2024.csv"
        if not os.path.exists(data_path):
            if self.verbose:
                print(f"Data file not found for ticker {ticker}: {data_path}")
            return False

        df = pd.read_csv(data_path, parse_dates=["date"])
        df = df.sort_values("date")
        # Only use data available up to current_date (no lookahead)
        df = df[df["date"] <= current_date]
        prices = df["prc"].astype(float)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        abs_log_returns = np.abs(log_returns)

        # Prepare rolling windows for DS3M (seq_len=20)
        # X: lagged |log return|, Y: |log return|
        X_list, Y_list = [], []
        for i in range(len(abs_log_returns) - seq_len):
            X_list.append(abs_log_returns.iloc[i:i+seq_len].values)
            Y_list.append(abs_log_returns.iloc[i+1:i+seq_len+1].values)
        if not X_list or not Y_list:
            if self.verbose:
                print("Not enough data for DS3M input sequences.")
            return False
        X = torch.tensor(np.array(X_list), dtype=torch.float32, device=device).unsqueeze(-1)  # (N, seq_len, 1)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float32, device=device).unsqueeze(-1)  # (N, seq_len, 1)

        # Use only the most recent window for forecasting
        X_input = X[-1:].transpose(0, 1)  # (seq_len, 1, 1)
        Y_input = Y[-1:].transpose(0, 1)  # (seq_len, 1, 1)

        model_path = f"models/ds3m_{ticker}.pt"
        scaler_path = f"models/ds3m_scaler_{ticker}.pkl"

        # Load or initialize DS3M model
        ds3m = DS3M(x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device).to(device)
        if os.path.exists(model_path):
            ds3m.load_state_dict(torch.load(model_path, map_location=device))
        else:
            if self.verbose:
                print(f"  [DS3M] No trained model found at {model_path}, using random weights")
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        # Forecast using DS3M
        try:
            out = ds3m.forecast(X_input, Y_input, steps=1, n_samples=100)
            # DS3M outputs |log return| per bar; annualise to match EGARCH/HARCNN scale
            raw_z = float(out['vol_forecast_mean'][-1][0][0])
            # Inverse-transform from z-score back to |log return| scale
            if scaler is not None:
                raw_forecast = abs(raw_z * scaler['std'] + scaler['mean'])
            else:
                raw_forecast = abs(raw_z)
            risk_premium = self._vol_risk_premium()
            self._cached_forecast = raw_forecast * np.sqrt(252) + risk_premium
            print(f"  [DS3M] raw={raw_forecast:.4f}, annualised={self._cached_forecast:.1%}")
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