from __future__ import annotations
import os
from datetime import date
import pandas as pd


def fetch_prices_yfinance(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.dropna(how="all")


def fetch_fred_series(series_ids: list[str], start: date, end: date) -> pd.DataFrame:
    from fredapi import Fred
    api_key = os.environ.get("FRED_API_KEY", "")
    fred = Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}
    for sid in series_ids:
        try:
            frames[sid] = fred.get_series(sid, observation_start=start, observation_end=end)
        except Exception as exc:
            raise RuntimeError(f"FRED fetch failed for {sid!r}: {exc}") from exc
    return pd.DataFrame(frames)


def fetch_prices(tickers: list[str], start: date, end: date,
                 source: str = "yfinance") -> pd.DataFrame:
    if source == "yfinance":
        return fetch_prices_yfinance(tickers, start, end)
    raise ValueError(f"Unknown data source: {source!r}. Supported: 'yfinance'")
