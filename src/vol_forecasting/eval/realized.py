from __future__ import annotations
import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def squared_return_proxy(returns: pd.Series) -> pd.Series:
    """Daily squared return, annualised — the simplest realized variance proxy."""
    return (returns**2 * _TRADING_DAYS).rename("rv_sq")


def rolling_realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualised realised vol: sqrt of mean squared returns over rolling window."""
    rv = returns.rolling(window).apply(
        lambda x: np.sqrt(np.mean(x**2) * _TRADING_DAYS), raw=True
    )
    return rv.rename("rv_roll")


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Parkinson (1980) high-low range estimator, annualised.

    More efficient than close-to-close when intraday range is available.
    """
    hl2 = (np.log(high / low) ** 2) / (4.0 * np.log(2.0))
    rv = np.sqrt(hl2.rolling(window).mean() * _TRADING_DAYS)
    return rv.rename("rv_parkinson")


def horizon_realized_vol(returns: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward-looking h-day realised vol proxy.

    rv_{t,h} = sqrt( (252/h) * sum_{i=1}^{h} r_{t+i}^2 )

    Aligns with h-step-ahead forecasts: rv[t] is realized over [t+1, t+h].
    """
    sq = returns**2
    # rolling sum of next h squared returns — reverse, roll, reverse
    fwd_sum = sq[::-1].rolling(horizon).sum()[::-1]
    rv = np.sqrt(fwd_sum * _TRADING_DAYS / horizon)
    return rv.shift(-horizon).rename(f"rv_fwd{horizon}")
