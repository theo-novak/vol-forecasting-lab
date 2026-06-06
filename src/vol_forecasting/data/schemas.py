from __future__ import annotations
from datetime import date
from typing import Literal
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

AssetClass = Literal["equity", "etf", "fx", "crypto", "other"]


class AssetConfig(BaseModel):
    ticker: str
    asset_class: AssetClass = "equity"
    currency: str = "USD"

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        return v.strip().upper()


class PriceHistory(BaseModel):
    prices: pd.DataFrame   # dates × tickers, adjusted close
    start: date
    end: date
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_prices(self) -> "PriceHistory":
        if self.prices.empty:
            raise ValueError("price history is empty")
        if self.prices.isnull().all().any():
            bad = self.prices.columns[self.prices.isnull().all()].tolist()
            raise ValueError(f"all-NaN columns: {bad}")
        return self

    def returns(self, fill: bool = True) -> pd.DataFrame:
        df = self.prices.copy()
        if fill:
            df = df.ffill()
        return df.pct_change().dropna(how="all")
