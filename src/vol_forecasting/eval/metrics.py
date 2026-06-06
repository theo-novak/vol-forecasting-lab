from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class EvalMetrics:
    model: str
    horizon: int
    mse: float
    mae: float
    qlike: float
    directional_accuracy: float
    n_obs: int


def _mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def mse(forecast: np.ndarray, realized: np.ndarray) -> float:
    m = _mask(forecast, realized)
    return float(np.mean((forecast[m] - realized[m]) ** 2))


def mae(forecast: np.ndarray, realized: np.ndarray) -> float:
    m = _mask(forecast, realized)
    return float(np.mean(np.abs(forecast[m] - realized[m])))


def qlike(forecast: np.ndarray, realized: np.ndarray) -> float:
    """Quasi-likelihood loss: QLIKE = E[log σ² + rv/σ²].

    Scale-invariant and more robust to outliers than MSE for volatility comparison.
    """
    m = _mask(forecast, realized) & (forecast > 0) & (realized > 0)
    f, r = forecast[m], realized[m]
    return float(np.mean(np.log(f) + r / f))


def directional_accuracy(forecast: np.ndarray, realized: np.ndarray,
                          prev: np.ndarray) -> float:
    """Fraction of periods where forecast and realized agree on vol direction."""
    m = _mask(forecast, realized) & np.isfinite(prev)
    pred_dir = np.sign(forecast[m] - prev[m])
    real_dir = np.sign(realized[m] - prev[m])
    return float(np.mean(pred_dir == real_dir))


def evaluate(model_name: str, forecast: pd.Series, realized: pd.Series,
             horizon: int) -> EvalMetrics:
    df = pd.DataFrame({"f": forecast, "r": realized}).dropna()
    f, r = df["f"].values, df["r"].values
    prev = np.roll(r, 1)
    prev[0] = np.nan
    return EvalMetrics(
        model=model_name,
        horizon=horizon,
        mse=mse(f, r),
        mae=mae(f, r),
        qlike=qlike(f, r),
        directional_accuracy=directional_accuracy(f, r, prev),
        n_obs=len(df),
    )
