# Simple per-ticker dividend-yield assumptions

from __future__ import annotations
from typing import Final

DIVIDEND_YIELD_BY_TICKER: Final[dict[str, float]] = {
    "AAPL": 0.005,
    "AMD": 0.000,
    "AMZN": 0.000,
    "BA": 0.000,
    "BAC": 0.019,
    "CAT": 0.018,
    "DIS": 0.000,
    "GE": 0.003,
    "GOOG": 0.001,
    "GS": 0.024,
    "INTC": 0.013,
    "MSFT": 0.008,
    "MU": 0.004,
    "NFLX": 0.000,
    "NKE": 0.012,
    "NVDA": 0.0003,
    "PFE": 0.045,
    "SBUX": 0.022,
    "TSLA": 0.000,
    "UNH": 0.013,
    "WMT": 0.015,
    "XOM": 0.035,
}


def get_dividend_yield(ticker: str | None, default: float = 0.02) -> float:
    """Return an assumed annualised dividend yield q for the given ticker."""
    if not ticker:
        return float(default)
    return float(DIVIDEND_YIELD_BY_TICKER.get(ticker.upper(), default))
