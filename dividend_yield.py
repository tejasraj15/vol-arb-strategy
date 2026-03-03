# Simple per-ticker dividend-yield assumptions

from __future__ import annotations

from typing import Final


DIVIDEND_YIELD_BY_TICKER: Final[dict[str, float]] = {
    "AAPL": 0.006,
    "DIS": 0.000,
    "MSFT": 0.008,
    "PFE": 0.045,
    "UNH": 0.012,
    "WMT": 0.015,
    "XOM": 0.035,
}


def get_dividend_yield(ticker: str | None, default: float = 0.02) -> float:
    """Return an assumed annualised dividend yield q for the given ticker."""
    if not ticker:
        return float(default)
    return float(DIVIDEND_YIELD_BY_TICKER.get(ticker.upper(), default))
