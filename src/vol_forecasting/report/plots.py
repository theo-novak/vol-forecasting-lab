from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go

_REGIME_FILL = {
    "calm":     "rgba(52, 168, 83,  0.14)",
    "normal":   "rgba(251, 188, 5,  0.10)",
    "stressed": "rgba(234, 67,  53, 0.18)",
}
_MODEL_LINE = {
    "historical": "#636EFA",
    "ewma":       "#EF553B",
    "garch":      "#00CC96",
    "realized":   "#AB63FA",
}


def plot_conditional_vol(
    vol_series: dict[str, pd.Series],
    realized: pd.Series | None = None,
    title: str = "Conditional Volatility",
) -> go.Figure:
    fig = go.Figure()
    if realized is not None:
        fig.add_trace(go.Scatter(
            x=realized.index, y=realized.values,
            name="Realized Vol",
            line=dict(color=_MODEL_LINE["realized"], width=1, dash="dot"),
        ))
    for name, series in vol_series.items():
        s = series.dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name,
            line=dict(color=_MODEL_LINE.get(name, "#888"), width=1.5),
        ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Annualised Vol",
        template="plotly_white", legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_regime_bands(
    vol_series: pd.Series,
    labels: pd.Series,
    title: str = "Volatility Regimes",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vol_series.index, y=vol_series.values,
        name="EWMA Vol", line=dict(color=_MODEL_LINE["ewma"], width=1.5),
    ))
    current = labels.iloc[0]
    band_start = vol_series.index[0]
    for dt, regime in labels.items():
        if regime != current:
            fig.add_vrect(
                x0=band_start, x1=dt,
                fillcolor=_REGIME_FILL[current],
                layer="below", line_width=0,
            )
            band_start = dt
            current = regime
    fig.add_vrect(
        x0=band_start, x1=vol_series.index[-1],
        fillcolor=_REGIME_FILL[current], layer="below", line_width=0,
    )
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Annualised Vol",
        template="plotly_white",
    )
    return fig


def plot_forecast_error(
    forecast_df: pd.DataFrame,
    realized: pd.Series,
    title: str = "Forecast Error",
    smooth: int = 21,
) -> go.Figure:
    fig = go.Figure()
    for col in forecast_df.columns:
        err = (forecast_df[col] - realized).abs().rolling(smooth).mean().dropna()
        fig.add_trace(go.Scatter(
            x=err.index, y=err.values,
            name=f"{col} ({smooth}d MA)", line=dict(width=1.5),
        ))
    fig.update_layout(
        title=title, xaxis_title="Date",
        yaxis_title=f"|Forecast − Realised| ({smooth}d MA)",
        template="plotly_white", legend=dict(orientation="h", y=-0.15),
    )
    return fig
