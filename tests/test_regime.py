from __future__ import annotations
import pytest
from vol_forecasting.models.ewma import fit_ewma
from vol_forecasting.regime.detector import detect_regimes, regime_summary


class TestRegimeDetector:
    def test_labels_only_valid_values(self, sample_returns):
        res    = fit_ewma(sample_returns)
        regime = detect_regimes(res.vol_series.dropna())
        assert set(regime.labels.unique()).issubset({"calm", "normal", "stressed"})

    def test_calm_threshold_below_stressed(self, sample_returns):
        res    = fit_ewma(sample_returns)
        regime = detect_regimes(res.vol_series.dropna())
        assert regime.thresholds["calm_cap"] <= regime.thresholds["stressed_floor"]

    def test_summary_percentages_sum_to_100(self, sample_returns):
        res     = fit_ewma(sample_returns)
        regime  = detect_regimes(res.vol_series.dropna())
        summary = regime_summary(regime)
        assert abs(summary["pct"].sum() - 100.0) < 0.1

    def test_all_three_regimes_present(self, sample_returns):
        res    = fit_ewma(sample_returns)
        regime = detect_regimes(res.vol_series.dropna())
        assert set(regime.labels.unique()) == {"calm", "normal", "stressed"}

    def test_custom_quantiles(self, sample_returns):
        res    = fit_ewma(sample_returns)
        regime = detect_regimes(res.vol_series.dropna(), calm_q=0.10, stressed_q=0.90)
        assert regime.thresholds["calm_cap"] < regime.thresholds["stressed_floor"]
