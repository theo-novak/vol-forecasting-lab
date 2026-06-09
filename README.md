# Conditional Volatility Forecasting

A reusable Python module that fits rolling historical volatility, EWMA (RiskMetrics), and GARCH(1,1) to daily asset returns, forecasts conditional volatility across multiple horizons (1-, 5-, 10-, and 21-day), and benchmarks each model against realized volatility proxies using MSE, MAE, and the quasi-likelihood (QLIKE) loss. A three-state regime detector classifies each trading day as *calm*, *normal*, or *stressed* based on quantile thresholds applied to the fitted EWMA series. Output is exposed through a `typer` CLI and a Streamlit dashboard with interactive tabs for conditional vol plots, forecast-error traces, and shaded regime bands.

Built on Python 3.11+ using `pandas`, `numpy`, `arch`, `scipy`, `pydantic` v2, `plotly`, and `streamlit`; packaged with `hatchling` and tested with `pytest` against a deterministic 500-day synthetic fixture.

---

## Project Layout

```
vol-forecasting/
├── pyproject.toml                      # Build config, deps, ruff + pytest settings
├── .env.example                        # API key template
├── data/                               # Populated by scripts/download_data.py
├── scripts/
│   └── download_data.py                # Fetches prices from yfinance + FRED macro series
├── src/vol_forecasting/
│   ├── data/
│   │   ├── schemas.py                  # Pydantic v2: AssetConfig, PriceHistory
│   │   ├── fetchers.py                 # yfinance and FRED HTTP fetchers
│   │   └── loaders.py                  # CSV → validated PriceHistory schema
│   ├── models/
│   │   ├── historical.py               # Rolling std baseline + carry-forward forecast
│   │   ├── ewma.py                     # RiskMetrics EWMA (λ=0.94); flat h-step forecast
│   │   └── garch.py                    # GARCH / EGARCH / GJR-GARCH via arch package
│   ├── eval/
│   │   ├── realized.py                 # Realized vol proxies: squared return, rolling, Parkinson
│   │   ├── metrics.py                  # MSE, MAE, QLIKE, directional accuracy
│   │   └── rolling_eval.py             # Multi-horizon evaluation across all models
│   ├── regime/
│   │   └── detector.py                 # calm / normal / stressed quantile classifier
│   ├── report/
│   │   └── plots.py                    # Plotly: conditional vol, regime bands, forecast error
│   ├── cli.py                          # Typer CLI: fit | eval | regime
│   └── app.py                          # Streamlit dashboard (3 tabs)
└── tests/
    ├── conftest.py                     # Deterministic 500-day synthetic return fixture
    ├── test_models.py                  # Invariant tests: vol, EWMA, GARCH
    ├── test_eval.py                    # Realized vol proxies and metric correctness
    └── test_regime.py                  # Regime classification invariants
```

---

## Setup

**Requirements:** Python 3.11+

```bash
# 1. Navigate to the project root
cd assets/projects/vol_forecasting

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install the package and all dependencies
pip install -e ".[dev]"

# 4. (Optional) add a FRED API key for macro overlays
cp .env.example .env
```

---

## CLI Usage

```bash
# Fit GARCH(1,1) to SPY (live fetch from yfinance)
vol-forecast fit --ticker SPY --model garch

# Fit EWMA with custom decay
vol-forecast fit --ticker QQQ --model ewma

# Evaluate all three models across 1-, 5-, 10-, and 21-day horizons (offline)
vol-forecast eval --ticker SPY --prices-file data/prices_yfinance.csv --horizons 1,5,10,21

# Regime map for BTC-USD
vol-forecast regime --ticker BTC-USD --prices-file data/prices_yfinance.csv

# Launch Streamlit dashboard
streamlit run src/vol_forecasting/app.py
```

---

## Streamlit Dashboard

```bash
streamlit run src/vol_forecasting/app.py
```

Opens at `http://localhost:8501`. Data is fetched live from Yahoo Finance on first load and cached for one hour; changing the ticker or lookback invalidates the cache. GARCH(1,1) fitting takes 2–5 seconds and re-runs whenever the ticker or lookback changes.

To work fully offline, first populate `data/` from the download script:

```bash
python scripts/download_data.py
```

### Sidebar controls

| Control | Default | Effect |
|---|---|---|
| **Ticker** | `SPY` | Any yfinance-valid symbol: `QQQ`, `BTC-USD`, `EURUSD=X`, `^VIX`, etc. |
| **Lookback (years)** | 5 | 1–10 years of daily price history |
| **Rolling Historical** | on | Toggles the 21-day rolling std baseline |
| **EWMA** | on | Toggles the RiskMetrics exponentially weighted estimator |
| **λ (EWMA decay)** | 0.94 | Lower values react faster to new shocks; 0.94 is the JP Morgan daily calibration |
| **GARCH(1,1)** | on | Fits the full GARCH model via MLE (2–5 s); uncheck to skip |
| **Horizon (trading days)** | 5 | Forecast horizon for the Forecast Error tab: 1, 5, 10, or 21 days |

### Tabs

| Tab | Content |
|---|---|
| 📈 **Conditional Volatility** | All fitted vol series overlaid with a dotted rolling-realized-vol reference. Metric cards show the latest annualised vol per model. |
| 📉 **Forecast Error** | Absolute error per model smoothed with a 21-day MA. Below: MSE, MAE, QLIKE, directional accuracy, and observation count. |
| 🗺️ **Regime Map** | EWMA vol with shaded bands (green = calm, yellow = normal, red = stressed). Regime breakdown tiles and quantile thresholds below. |

---

## Models

### I. Rolling Historical Volatility (baseline)

Rolling window standard deviation, annualised by $\sqrt{252}$. The $h$-step forecast is the current estimate carried forward unchanged — no mean-reversion, no parameter estimation.

$$\hat\sigma_{t,w} = \sqrt{\frac{252}{w-1}\sum_{i=0}^{w-1}(r_{t-i} - \bar r_w)^2}$$

Default window: $w = 21$ trading days.

### II. EWMA / RiskMetrics

Exponentially weighted variance recursion with decay factor $\lambda = 0.94$ (JP Morgan daily calibration):

$$\sigma^2_t = \lambda\,\sigma^2_{t-1} + (1-\lambda)\,r_t^2$$

EWMA is structurally equivalent to IGARCH(1,1) — persistence $= 1$, no finite unconditional variance. The $h$-step forecast is flat: $\hat\sigma^2_{t+h|t} = \sigma^2_t$ for all $h$.

### III. GARCH(1,1)

Adds a mean-reverting intercept to the EWMA recursion (Bollerslev, 1986):

$$\sigma^2_t = \omega + \alpha\,\epsilon^2_{t-1} + \beta\,\sigma^2_{t-1}$$

Long-run (unconditional) variance — finite when $\alpha + \beta < 1$:

$$\bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}$$

Analytical $h$-step forecast reverts geometrically to the long-run level:

$$\hat\sigma^2_{t+h} = \bar\sigma^2 + (\alpha+\beta)^{h-1}\!\left(\sigma^2_{t+1} - \bar\sigma^2\right)$$

The module also supports **EGARCH(1,1)** (Nelson, 1991), which models $\log\sigma^2_t$ and captures the leverage effect ($\gamma < 0$ for equities), and **GJR-GARCH**. Fitting uses the `arch` package with MLE under Gaussian innovations.

---

## Realized Volatility Proxies

| Proxy | Formula | Notes |
|---|---|---|
| Squared return | $rv_t = 252 \cdot r_t^2$ | Unbiased but high variance |
| Rolling RV | $rv_{t,w} = \sqrt{\frac{252}{w}\sum_{i=0}^{w-1}r_{t-i}^2}$ | Smoothed; introduces look-back bias |
| Parkinson (1980) | $rv_t^P = \sqrt{\frac{252}{4\ln 2}(\ln H_t/L_t)^2}$ | Up to 5× more efficient when OHLC data available |
| Forward $h$-day RV | $rv_{t,h} = \sqrt{\frac{252}{h}\sum_{i=1}^{h}r_{t+i}^2}$ | Primary evaluation target; aligns with $h$-step forecast made at $t$ |

---

## Evaluation

Three loss functions computed per model-horizon pair:

| Loss | Formula | Properties |
|---|---|---|
| MSE | $\mathbb{E}[(\hat\sigma_h - rv_h)^2]$ | Penalises large errors quadratically; sensitive to outliers |
| MAE | $\mathbb{E}[|\hat\sigma_h - rv_h|]$ | Robust to extreme realized-vol spikes |
| QLIKE | $\mathbb{E}[\log\hat\sigma^2 + rv/\hat\sigma^2]$ | Scale-invariant; standard loss in Patton (2011) for robust vol ranking |

Directional accuracy measures the fraction of periods where the model correctly predicts whether volatility will rise or fall relative to the current level.

---

## Regime Detector

Applies full-sample quantile thresholds to the EWMA conditional vol series:

$$\text{regime}_t = \begin{cases}\textit{calm} & \sigma_t \le Q_{0.25} \\ \textit{stressed} & \sigma_t \ge Q_{0.75} \\ \textit{normal} & \text{otherwise}\end{cases}$$

The 25th and 75th percentiles are computed on the full fitted series. Thresholds can be overridden via `calm_q` and `stressed_q` arguments at the call site.

---

## Tests

All tests are fully offline. A deterministic 500-day SPY-like return series (seed 42) is generated in `conftest.py` and shared across all test classes via pytest fixtures.

```bash
pytest
```

Tests verify mathematical invariants: vol non-negativity, annualisation factor, GARCH persistence < 1, long-run variance positivity, regime percentages summing to 100%, and that $h=21$ GARCH forecasts lie closer to the long-run vol than $h=1$ forecasts.

---

## Configuration & Data Sources

| Variable | Default | Description |
|---|---|---|
| `DATA_SOURCE` | `yfinance` | Live-fetch source when `--prices-file` is omitted |
| `FRED_API_KEY` | *(none)* | Required for FRED macro overlay download |

| File (generated by `download_data.py`) | Source | Coverage |
|---|---|---|
| `prices_yfinance.csv` | Yahoo Finance | 5 years, 9 tickers: SPY, QQQ, IWM, GLD, TLT, USO, BTC-USD, EURUSD=X, ^VIX |
| `fred_macro.csv` | FRED | 5 years, 5 series: Fed Funds, 10Y−2Y, HY OAS, VIX, CPI |

---

## Tools & Methods

Python 3.11, pandas, NumPy, SciPy, arch (GARCH/EGARCH/GJR-GARCH via MLE), Pydantic v2, Typer, rich, Plotly, Streamlit, yfinance, FRED, pytest, ruff, hatchling.

**Author:** Theodosios Dimitrasopoulos
