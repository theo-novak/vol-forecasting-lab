from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Literal

RegimeLabel = Literal["calm", "normal", "stressed"]

_CALM_Q    = 0.25
_STRESSED_Q = 0.75


@dataclass
class RegimeResult:
    ticker: str
    labels: pd.Series              # dated string series: calm / normal / stressed
    vol_series: pd.Series          # underlying volatility used for classification
    thresholds: dict[str, float]   # calm_cap, stressed_floor (annualised vol)


def detect_regimes(
    vol_series: pd.Series,
    calm_q: float = _CALM_Q,
    stressed_q: float = _STRESSED_Q,
) -> RegimeResult:
    """Classify annualised conditional vol into calm / normal / stressed.

    Thresholds are the calm_q and stressed_q full-sample quantiles of the vol
    series. Intended for report generation on fitted in-sample data.
    """
    vol = vol_series.dropna()
    calm_cap      = float(vol.quantile(calm_q))
    stressed_floor = float(vol.quantile(stressed_q))

    labels = pd.Series("normal", index=vol.index, name="regime", dtype=object)
    labels[vol <= calm_cap]       = "calm"
    labels[vol >= stressed_floor] = "stressed"

    return RegimeResult(
        ticker=str(vol_series.name),
        labels=labels,
        vol_series=vol,
        thresholds={"calm_cap": calm_cap, "stressed_floor": stressed_floor},
    )


def regime_summary(result: RegimeResult) -> pd.DataFrame:
    """Days per regime and fraction of total sample."""
    counts = result.labels.value_counts().rename("days")
    pct    = (counts / counts.sum() * 100).round(2).rename("pct")
    df = pd.DataFrame({"days": counts, "pct": pct})
    return df.reindex(["calm", "normal", "stressed"]).fillna(0)
