from __future__ import annotations
import math
import numpy as np
import pytest
from vol_forecasting.models.historical import rolling_historical_vol, forecast_historical
from vol_forecasting.models.ewma import fit_ewma, forecast_ewma
from vol_forecasting.models.garch import fit_garch, forecast_garch


class TestHistoricalVol:
    def test_non_negative(self, sample_returns):
        vol = rolling_historical_vol(sample_returns, window=21)
        assert (vol.dropna() >= 0).all()

    def test_annualisation_factor(self, sample_returns):
        ann = rolling_historical_vol(sample_returns, window=21, annualise=True)
        raw = rolling_historical_vol(sample_returns, window=21, annualise=False)
        ratio = (ann.dropna() / raw.dropna()).dropna()
        assert np.allclose(ratio, np.sqrt(252), atol=1e-9)

    def test_forecast_shift(self, sample_returns):
        vol  = rolling_historical_vol(sample_returns, window=21)
        fwd5 = forecast_historical(vol, horizon=5)
        overlap = vol.dropna().index.intersection(fwd5.dropna().index)
        assert len(overlap) > 0


class TestEWMA:
    def test_non_negative(self, sample_returns):
        res = fit_ewma(sample_returns)
        assert (res.vol_series.dropna() >= 0).all()

    def test_higher_lambda_smoother(self, sample_returns):
        """Higher λ → slower adaptation → lower std of daily vol changes."""
        res_low  = fit_ewma(sample_returns, lam=0.90)
        res_high = fit_ewma(sample_returns, lam=0.99)
        assert res_high.vol_series.diff().dropna().std() < \
               res_low.vol_series.diff().dropna().std()

    def test_flat_forecast_is_shift(self, sample_returns):
        res  = fit_ewma(sample_returns)
        fwd1 = forecast_ewma(res, horizon=1)
        fwd5 = forecast_ewma(res, horizon=5)
        assert len(fwd1.dropna()) > 0
        assert len(fwd5.dropna()) > 0


class TestGARCH:
    def test_non_negative(self, sample_returns):
        res = fit_garch(sample_returns)
        assert (res.cond_vol >= 0).all()

    def test_long_run_var_positive(self, sample_returns):
        res = fit_garch(sample_returns)
        assert res.long_run_var > 0

    def test_persistence_less_than_one(self, sample_returns):
        res   = fit_garch(sample_returns)
        alpha = res.params.get("alpha[1]", 0.0)
        beta  = res.params.get("beta[1]",  0.0)
        assert alpha + beta < 1.0

    def test_forecast_reverts_to_long_run(self, sample_returns):
        """h=21 forecast should be closer to long-run vol than h=1 forecast."""
        res  = fit_garch(sample_returns)
        fwd1 = forecast_garch(res, horizon=1).dropna()
        fwd21 = forecast_garch(res, horizon=21).dropna()
        lr_vol = math.sqrt(res.long_run_var * 252)
        overlap = fwd1.index.intersection(fwd21.index)
        assert len(overlap) > 0
        dev1  = (fwd1[overlap]  - lr_vol).abs().mean()
        dev21 = (fwd21[overlap] - lr_vol).abs().mean()
        assert dev21 < dev1
