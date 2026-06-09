from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

_TRADING_DAYS = 252


@dataclass
class EWMAResult:
    ticker: str
    vol_series: pd.Series    # annualised EWMA vol
    var_series: pd.Series    # daily variance (not annualised)
    lam: float


def fit_ewma(returns: pd.Series, lam: float = 0.94,
             annualise: bool = True) -> EWMAResult:
    """RiskMetrics EWMA: var_t = λ·var_{t-1} + (1-λ)·r_t²."""
    # Use the uncentered second moment (mean of squares), not the centered
    # variance, which is the exact RiskMetrics recursion.
    var = (returns ** 2).ewm(alpha=1.0 - lam, adjust=False).mean()
    vol = np.sqrt(var)
    if annualise:
        vol = vol * np.sqrt(_TRADING_DAYS)
    return EWMAResult(
        ticker=str(returns.name),
        vol_series=vol.rename("ewma_vol"),
        var_series=var,
        lam=lam,
    )


def forecast_ewma(result: EWMAResult, horizon: int = 1) -> pd.Series:
    """EWMA h-step forecast.

    EWMA is IGARCH(1,1): persistence = 1, no unconditional variance. The
    h-step forecast is flat — the current estimate is carried forward unchanged.
    """
    return result.vol_series.shift(horizon).rename(f"ewma_vol_fwd{horizon}")
