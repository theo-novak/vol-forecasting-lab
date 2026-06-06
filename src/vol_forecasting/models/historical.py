from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

_TRADING_DAYS = 252


@dataclass
class HistoricalVolResult:
    ticker: str
    vol_series: pd.Series   # rolling std, annualised
    window: int


def rolling_historical_vol(returns: pd.Series, window: int = 21,
                            annualise: bool = True) -> pd.Series:
    vol = returns.rolling(window).std(ddof=1)
    if annualise:
        vol = vol * np.sqrt(_TRADING_DAYS)
    return vol.rename("hist_vol")


def fit_historical(returns: pd.Series, window: int = 21) -> HistoricalVolResult:
    return HistoricalVolResult(
        ticker=str(returns.name),
        vol_series=rolling_historical_vol(returns, window),
        window=window,
    )


def forecast_historical(vol_series: pd.Series, horizon: int = 1) -> pd.Series:
    """Carry the last rolling-std estimate forward h periods."""
    return vol_series.shift(horizon).rename(f"hist_vol_fwd{horizon}")
