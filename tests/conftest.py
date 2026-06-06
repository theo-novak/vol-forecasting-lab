from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from vol_forecasting.data.schemas import PriceHistory


@pytest.fixture
def sample_returns() -> pd.Series:
    """Deterministic 500-day log-normal SPY-like return series (seed 42)."""
    rng = np.random.default_rng(42)
    n   = 500
    dates = pd.bdate_range("2022-01-01", periods=n)
    rets  = rng.normal(0.0004, 0.011, size=n)
    return pd.Series(rets, index=dates, name="SPY")


@pytest.fixture
def sample_price_history(sample_returns: pd.Series) -> PriceHistory:
    prices = 100.0 * np.exp(np.cumsum(sample_returns.values))
    df = pd.DataFrame({"SPY": prices}, index=sample_returns.index)
    return PriceHistory(
        prices=df,
        start=df.index[0].date(),
        end=df.index[-1].date(),
    )
