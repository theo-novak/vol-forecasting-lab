from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

st.set_page_config(page_title="Volatility Forecasting", layout="wide")
st.title("Conditional Volatility Forecasting")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    ticker   = st.text_input("Ticker", "SPY")
    lookback = st.slider("Lookback (years)", 1, 10, 5)

    st.subheader("Models")
    use_hist  = st.checkbox("Rolling Historical (21d)", value=True)
    use_ewma  = st.checkbox("EWMA", value=True)
    lam       = st.slider("λ (EWMA decay)", 0.85, 0.99, 0.94, step=0.01)
    use_garch = st.checkbox("GARCH(1,1)", value=True)

    st.subheader("Forecast Horizon")
    horizon = st.selectbox("Horizon (trading days)", [1, 5, 10, 21], index=1)


@st.cache_data(ttl=3600)
def load_data(ticker: str, years: int) -> pd.DataFrame:
    import yfinance as yf
    end   = date.today()
    start = end - timedelta(days=years * 365 + 60)
    raw   = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        st.error(f"No data returned for ticker '{ticker}'.")
        st.stop()
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(ticker).to_frame()


prices  = load_data(ticker, lookback)
returns = prices[ticker].pct_change().dropna().rename(ticker)

# ── Fit models ────────────────────────────────────────────────────────────────
vol_series: dict[str, pd.Series] = {}

if use_hist:
    from vol_forecasting.models.historical import rolling_historical_vol
    vol_series["historical"] = rolling_historical_vol(returns, window=21)

if use_ewma:
    from vol_forecasting.models.ewma import fit_ewma
    ewma_res = fit_ewma(returns, lam=lam)
    vol_series["ewma"] = ewma_res.vol_series

garch_res = None
if use_garch:
    from vol_forecasting.models.garch import fit_garch
    try:
        with st.spinner("Fitting GARCH(1,1)..."):
            garch_res = fit_garch(returns)
        vol_series["garch"] = garch_res.cond_vol
    except Exception as exc:
        st.warning(f"GARCH fit failed: {exc}")

# Realized vol proxies
from vol_forecasting.eval.realized import rolling_realized_vol, horizon_realized_vol
realized_vol = rolling_realized_vol(returns, window=21)
realized_fwd = horizon_realized_vol(returns, horizon=horizon)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📈 Conditional Volatility", "📉 Forecast Error", "🗺️ Regime Map"]
)

with tab1:
    from vol_forecasting.report.plots import plot_conditional_vol
    fig = plot_conditional_vol(
        vol_series, realized=realized_vol,
        title=f"{ticker} — Conditional Volatility (annualised)",
    )
    st.plotly_chart(fig, width="stretch")

    if vol_series:
        cols = st.columns(len(vol_series))
        for col, (name, s) in zip(cols, vol_series.items()):
            latest = s.dropna().iloc[-1] if not s.dropna().empty else float("nan")
            col.metric(f"{name.capitalize()} Vol", f"{latest:.2%}")

with tab2:
    from vol_forecasting.eval.rolling_eval import rolling_eval
    from vol_forecasting.report.plots import plot_forecast_error

    with st.spinner("Running evaluation..."):
        eval_results = rolling_eval(
            returns, horizons=(horizon,), fit_garch_flag=(garch_res is not None)
        )
    r = eval_results[0]

    fig2 = plot_forecast_error(
        r.forecasts, r.realized,
        title=f"Forecast Error — {horizon}d horizon ({ticker})",
    )
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Evaluation Metrics")
    rows = [
        {
            "Model":    m.model,
            "MSE":      f"{m.mse:.6f}",
            "MAE":      f"{m.mae:.4f}",
            "QLIKE":    f"{m.qlike:.4f}",
            "Dir. Acc.": f"{m.directional_accuracy:.2%}",
            "N":        m.n_obs,
        }
        for m in r.metrics
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab3:
    from vol_forecasting.regime.detector import detect_regimes, regime_summary
    from vol_forecasting.report.plots import plot_regime_bands

    base = next(
        (vol_series[k] for k in ("ewma", "garch") if k in vol_series),
        realized_vol,
    )
    regime_res = detect_regimes(base.dropna())

    fig3 = plot_regime_bands(
        regime_res.vol_series, regime_res.labels,
        title=f"{ticker} — Volatility Regimes",
    )
    st.plotly_chart(fig3, width="stretch")

    summary = regime_summary(regime_res)
    st.subheader("Regime Breakdown")
    cols = st.columns(3)
    for col, (reg, row) in zip(cols, summary.iterrows()):
        col.metric(str(reg).capitalize(), f"{int(row['days'])} days", f"{row['pct']}%")

    st.caption(
        f"Calm ≤ {regime_res.thresholds['calm_cap']:.2%}   |   "
        f"Stressed ≥ {regime_res.thresholds['stressed_floor']:.2%}   (full-sample quantiles)"
    )
