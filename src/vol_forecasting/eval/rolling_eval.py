from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from vol_forecasting.models.historical import rolling_historical_vol, forecast_historical
from vol_forecasting.models.ewma import fit_ewma, forecast_ewma
from vol_forecasting.models.garch import fit_garch, forecast_garch
from vol_forecasting.eval.realized import horizon_realized_vol
from vol_forecasting.eval.metrics import evaluate, EvalMetrics

_DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 21)


@dataclass
class RollingEvalResult:
    ticker: str
    horizon: int
    metrics: list[EvalMetrics]
    forecasts: pd.DataFrame    # dates × model names
    realized: pd.Series


def rolling_eval(
    returns: pd.Series,
    horizons: tuple[int, ...] = _DEFAULT_HORIZONS,
    window: int = 21,
    fit_garch_flag: bool = True,
) -> list[RollingEvalResult]:
    """Evaluate model forecast accuracy across multiple horizons.

    Fits each model on the full return history (filtered estimates, not smoothed),
    then aligns h-step-ahead forecasts against forward-looking realized vol proxies.
    """
    ewma_res = fit_ewma(returns)

    garch_res = None
    if fit_garch_flag:
        try:
            garch_res = fit_garch(returns)
        except Exception:
            pass

    results: list[RollingEvalResult] = []

    for h in horizons:
        rv = horizon_realized_vol(returns, horizon=h)

        hist_vol = rolling_historical_vol(returns, window=window)
        models: dict[str, pd.Series] = {
            "historical": forecast_historical(hist_vol, horizon=h),
            "ewma":       forecast_ewma(ewma_res, horizon=h),
        }
        if garch_res is not None:
            models["garch"] = forecast_garch(garch_res, horizon=h)

        forecasts_df = pd.DataFrame(models).dropna(how="all")
        rv_aligned = rv.reindex(forecasts_df.index)

        metrics = [
            evaluate(name, forecasts_df[name], rv_aligned, h)
            for name in forecasts_df.columns
        ]

        results.append(RollingEvalResult(
            ticker=str(returns.name),
            horizon=h,
            metrics=metrics,
            forecasts=forecasts_df,
            realized=rv_aligned,
        ))

    return results
