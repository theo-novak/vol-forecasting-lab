from __future__ import annotations
"""Populate data/ from yfinance and FRED. Run once before using offline CSVs."""
import os
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "GLD", "TLT", "USO",
    "BTC-USD", "EURUSD=X", "^VIX",
]

FRED_SERIES: dict[str, str] = {
    "DFF":         "fed_funds",
    "T10Y2Y":      "term_spread",
    "BAMLH0A0HYM2": "hy_oas",
    "VIXCLS":      "vix_fred",
    "CPIAUCSL":    "cpi",
}


def download_prices(tickers: list[str] = DEFAULT_TICKERS, years: int = 5) -> None:
    import yfinance as yf
    end   = date.today()
    start = end - timedelta(days=years * 365 + 60)
    raw   = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    out = DATA_DIR / "prices_yfinance.csv"
    prices.to_csv(out)
    print(f"Saved {len(prices)} rows × {len(prices.columns)} tickers → {out}")


def download_fred() -> None:
    from fredapi import Fred
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("FRED_API_KEY not set — skipping FRED download")
        return
    fred  = Fred(api_key=api_key)
    end   = date.today()
    start = end - timedelta(days=365 * 5)
    frames: dict[str, pd.Series] = {}
    for sid, name in FRED_SERIES.items():
        try:
            frames[name] = fred.get_series(sid, observation_start=start, observation_end=end)
            print(f"  Fetched {sid} → {name}")
        except Exception as exc:
            print(f"  Failed {sid}: {exc}")
    if frames:
        out = DATA_DIR / "fred_macro.csv"
        pd.DataFrame(frames).to_csv(out)
        print(f"Saved FRED macro series → {out}")


if __name__ == "__main__":
    download_prices()
    download_fred()
