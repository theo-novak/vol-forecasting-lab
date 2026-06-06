from __future__ import annotations
import numpy as np
import pytest
from vol_forecasting.eval.realized import (
    squared_return_proxy,
    rolling_realized_vol,
    horizon_realized_vol,
)
from vol_forecasting.eval.metrics import mse, mae, qlike, evaluate


class TestRealizedVol:
    def test_squared_return_non_negative(self, sample_returns):
        rv = squared_return_proxy(sample_returns)
        assert (rv >= 0).all()

    def test_rolling_rv_non_negative(self, sample_returns):
        rv = rolling_realized_vol(sample_returns, window=21)
        assert (rv.dropna() >= 0).all()

    def test_horizon_rv_has_observations(self, sample_returns):
        for h in (1, 5, 21):
            rv = horizon_realized_vol(sample_returns, horizon=h)
            assert len(rv.dropna()) > 0


class TestMetrics:
    def test_mse_perfect_zero(self):
        x = np.array([0.1, 0.2, 0.3])
        assert mse(x, x) == pytest.approx(0.0)

    def test_mse_non_negative(self):
        rng = np.random.default_rng(1)
        a, b = rng.uniform(0.05, 0.3, 100), rng.uniform(0.05, 0.3, 100)
        assert mse(a, b) >= 0.0

    def test_mae_non_negative(self):
        rng = np.random.default_rng(2)
        a, b = rng.uniform(0.05, 0.3, 100), rng.uniform(0.05, 0.3, 100)
        assert mae(a, b) >= 0.0

    def test_qlike_finite_for_positive_inputs(self):
        rng = np.random.default_rng(3)
        f = rng.uniform(0.05, 0.3, 100)
        r = rng.uniform(0.05, 0.3, 100)
        assert np.isfinite(qlike(f, r))

    def test_evaluate_returns_correct_n_obs(self, sample_returns):
        from vol_forecasting.models.ewma import fit_ewma
        from vol_forecasting.eval.realized import horizon_realized_vol
        res  = fit_ewma(sample_returns)
        fwd  = res.vol_series.shift(1)
        rv   = horizon_realized_vol(sample_returns, horizon=1)
        m    = evaluate("ewma", fwd, rv, horizon=1)
        assert m.n_obs > 0
        assert m.model == "ewma"
