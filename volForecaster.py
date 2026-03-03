import numpy as np
import pandas as pd
from garch import garch_modelling
from preprocess_data import parse_data

DE_MEAN = "AR"
MODEL = "EGARCH"
DISTRIBUTION = {"GARCH": "normal", "EGARCH": "t"}[MODEL]


class VolForecaster:

    def __init__(self, stock_data: pd.DataFrame, train_window: int = 126, refit_frequency: int = 21, verbose=False):
        self.stock_data = stock_data
        self.train_window = train_window
        self.refit_frequency = refit_frequency

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

    def _refit(self, current_date: pd.Timestamp) -> bool:
        historical_data = (
            self.stock_data.loc[:current_date]
            .iloc[-(self.train_window + 1):-1]
        )
        log_returns = parse_data(historical_data, price_col='prc')

        try:
            _, base_forecast = garch_modelling(
                log_returns, DE_MEAN, MODEL, DISTRIBUTION, validity_checks=False
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

    def _vol_risk_premium(self, min_samples: int = 60) -> float:
        if len(self._forecast_history) < min_samples:
            return 0.03

        lookback = min(126, len(self._forecast_history))
        recent_forecasts = np.array(self._forecast_history[-lookback:])
        recent_markets = np.array(self._market_iv_history[-lookback:])

        premium = np.median(recent_markets - recent_forecasts)
        return float(np.clip(premium, 0.0, 0.10))