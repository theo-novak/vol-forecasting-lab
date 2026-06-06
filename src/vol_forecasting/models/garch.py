from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

_TRADING_DAYS = 252

ModelSpec = Literal["garch", "egarch", "gjr-garch"]


@dataclass
class GARCHResult:
    ticker: str
    spec: ModelSpec
    cond_vol: pd.Series          # conditional daily vol, annualised
    long_run_var: float          # ω/(1−α−β), daily variance in decimal units
    params: dict[str, float]
    _fitted: object = field(repr=False, default=None)


def fit_garch(returns: pd.Series, spec: ModelSpec = "garch",
              p: int = 1, q: int = 1) -> GARCHResult:
    """Fit GARCH(p,q), EGARCH(p,q), or GJR-GARCH(p,q) via the arch package.

    Returns multiply by 100 before fitting for numerical stability, then unscale.
    """
    from arch import arch_model
    _SCALE = 100.0
    r = returns.dropna() * _SCALE

    if spec == "garch":
        am = arch_model(r, vol="GARCH", p=p, q=q, dist="normal")
    elif spec == "egarch":
        am = arch_model(r, vol="EGARCH", p=p, q=q, dist="normal")
    elif spec == "gjr-garch":
        am = arch_model(r, vol="GARCH", p=p, o=1, q=q, dist="normal")
    else:
        raise ValueError(f"Unknown spec: {spec!r}")

    res = am.fit(disp="off", show_warning=False)

    cond_vol = pd.Series(
        res.conditional_volatility.values / _SCALE * np.sqrt(_TRADING_DAYS),
        index=returns.dropna().index,
        name="garch_vol",
    )

    # long-run variance in decimal² per day (only valid for GARCH, not EGARCH)
    omega = float(res.params.get("omega", np.nan)) / _SCALE**2
    alpha = float(res.params.get("alpha[1]", 0.0))
    beta  = float(res.params.get("beta[1]", 0.0))
    denom = 1.0 - alpha - beta
    long_run_var = omega / denom if denom > 1e-8 else np.nan

    return GARCHResult(
        ticker=str(returns.name),
        spec=spec,
        cond_vol=cond_vol,
        long_run_var=long_run_var,
        params=dict(res.params),
        _fitted=res,
    )


def forecast_garch(result: GARCHResult, horizon: int = 1) -> pd.Series:
    """Analytical h-step GARCH(1,1) mean-reversion forecast.

    σ²_{t+h} = σ̄² + (α+β)^(h−1) · (σ²_t − σ̄²)

    The conditional variance reverts geometrically to the long-run mean.
    EGARCH and GJR-GARCH fall back to h=1 (current estimate) since the
    analytical recursion differs.
    """
    if result.spec != "garch":
        return result.cond_vol.shift(horizon).rename(f"garch_vol_fwd{horizon}")

    alpha = float(result.params.get("alpha[1]", 0.0))
    beta  = float(result.params.get("beta[1]", 0.0))
    persistence = alpha + beta
    lrv = result.long_run_var   # daily variance, decimal²

    daily_var = (result.cond_vol / np.sqrt(_TRADING_DAYS)) ** 2

    if horizon <= 1:
        fwd_daily_var = daily_var
    else:
        fwd_daily_var = lrv + (persistence ** (horizon - 1)) * (daily_var - lrv)

    fwd_vol = np.sqrt(fwd_daily_var.clip(lower=0.0)) * np.sqrt(_TRADING_DAYS)
    return fwd_vol.shift(horizon).rename(f"garch_vol_fwd{horizon}")
