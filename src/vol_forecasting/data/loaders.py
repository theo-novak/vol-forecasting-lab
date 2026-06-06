from __future__ import annotations
from pathlib import Path
import pandas as pd
from .schemas import PriceHistory


def load_prices(path: str | Path) -> PriceHistory:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).date
    df = df.sort_index().astype(float)
    return PriceHistory(
        prices=df,
        start=df.index[0],
        end=df.index[-1],
    )
