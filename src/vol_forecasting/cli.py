from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
app = typer.Typer(
    name="vol-forecast",
    help="Conditional volatility modelling: EWMA, GARCH(1,1), and regime detection.",
    add_completion=False,
)
console = Console()


def _load_returns(ticker: str, prices_file: Optional[Path], window: int) -> "pd.Series":
    import pandas as pd
    from datetime import date, timedelta
    from vol_forecasting.data.fetchers import fetch_prices
    from vol_forecasting.data.loaders import load_prices

    if prices_file is not None:
        ph = load_prices(prices_file)
        if ticker not in ph.prices.columns:
            raise typer.BadParameter(f"Ticker {ticker!r} not in {prices_file}")
        returns = ph.returns()[ticker].rename(ticker)
        console.print(f"[green]Loaded prices from file:[/green] {len(ph.prices)} days")
    else:
        end = date.today()
        start = end - timedelta(days=max(window * 3, 365 * 5))
        with console.status(f"Fetching {ticker} from yfinance..."):
            prices = fetch_prices([ticker], start=start, end=end)
        returns = prices[ticker].pct_change().dropna().rename(ticker)
        console.print(f"[green]Fetched:[/green] {len(returns)} return observations")

    return returns


@app.command("fit")
def cmd_fit(
    ticker: str = "SPY",
    model: str = "garch",
    prices_file: Optional[Path] = None,
    window: int = 252,
) -> None:
    """Fit a volatility model and print a summary of the conditional vol series."""
    returns = _load_returns(ticker, prices_file, window)

    if model == "ewma":
        from vol_forecasting.models.ewma import fit_ewma
        res = fit_ewma(returns)
        vol = res.vol_series
    elif model == "historical":
        from vol_forecasting.models.historical import rolling_historical_vol
        vol = rolling_historical_vol(returns, window=21)
    elif model in ("garch", "egarch", "gjr-garch"):
        from vol_forecasting.models.garch import fit_garch
        with console.status(f"Fitting {model.upper()}(1,1)..."):
            res = fit_garch(returns, spec=model)
        vol = res.cond_vol
        if model == "garch" and not __import__("math").isnan(res.long_run_var):
            lr_vol = (res.long_run_var * 252) ** 0.5
            console.print(f"[bold]Long-run vol (annualised):[/bold] {lr_vol:.2%}")
            alpha = res.params.get("alpha[1]", 0.0)
            beta  = res.params.get("beta[1]",  0.0)
            console.print(f"[bold]Persistence (α+β):[/bold] {alpha + beta:.4f}")
    else:
        raise typer.BadParameter(f"Unknown model: {model!r}")

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Stat"); t.add_column("Value", justify="right")
    t.add_row("Model",             model)
    t.add_row("Ticker",            ticker)
    t.add_row("Observations",      str(len(vol.dropna())))
    t.add_row("Latest vol (ann.)", f"{vol.dropna().iloc[-1]:.4%}")
    t.add_row("Mean vol (ann.)",   f"{vol.dropna().mean():.4%}")
    t.add_row("Max vol (ann.)",    f"{vol.dropna().max():.4%}")
    t.add_row("Min vol (ann.)",    f"{vol.dropna().min():.4%}")
    console.print(t)


@app.command("eval")
def cmd_eval(
    ticker: str = "SPY",
    horizons: str = "1,5,10,21",
    prices_file: Optional[Path] = None,
    window: int = 252,
    fit_garch: bool = True,
) -> None:
    """Walk-forward evaluation across models and forecast horizons."""
    returns = _load_returns(ticker, prices_file, window)
    from vol_forecasting.eval.rolling_eval import rolling_eval

    h_list = tuple(int(h) for h in horizons.split(","))
    console.print(f"[bold]Evaluating horizons:[/bold] {h_list}")
    results = rolling_eval(returns, horizons=h_list, fit_garch_flag=fit_garch)

    for r in results:
        console.rule(f"[bold blue]Horizon = {r.horizon}d  |  {ticker}")
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Model")
        t.add_column("MSE",      justify="right")
        t.add_column("MAE",      justify="right")
        t.add_column("QLIKE",    justify="right")
        t.add_column("Dir. Acc.", justify="right")
        t.add_column("N",        justify="right")
        for m in r.metrics:
            t.add_row(
                m.model,
                f"{m.mse:.6f}",
                f"{m.mae:.4f}",
                f"{m.qlike:.4f}",
                f"{m.directional_accuracy:.2%}",
                str(m.n_obs),
            )
        console.print(t)


@app.command("regime")
def cmd_regime(
    ticker: str = "SPY",
    prices_file: Optional[Path] = None,
    window: int = 252,
) -> None:
    """Classify volatility into calm / normal / stressed regimes."""
    returns = _load_returns(ticker, prices_file, window)
    from vol_forecasting.models.ewma import fit_ewma
    from vol_forecasting.regime.detector import detect_regimes, regime_summary

    ewma_res   = fit_ewma(returns)
    regime_res = detect_regimes(ewma_res.vol_series.dropna())
    summary    = regime_summary(regime_res)

    console.rule(f"[bold blue]Regime Summary — {ticker}")
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Regime"); t.add_column("Days", justify="right")
    t.add_column("% of Sample", justify="right")
    _color = {"calm": "green", "normal": "yellow", "stressed": "red"}
    for reg, row in summary.iterrows():
        c = _color.get(str(reg), "white")
        t.add_row(f"[{c}]{reg}[/{c}]", str(int(row["days"])), f"{row['pct']}%")
    console.print(t)
    console.print(
        f"\nCalm threshold   (≤25th pct): "
        f"{regime_res.thresholds['calm_cap']:.4%}"
    )
    console.print(
        f"Stressed floor   (≥75th pct): "
        f"{regime_res.thresholds['stressed_floor']:.4%}"
    )


if __name__ == "__main__":
    app()
