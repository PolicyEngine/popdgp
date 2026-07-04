"""Metric identity, mode-collapse, tail, and stress tests.

Ported from imputation-paper's ``tests/test_smoke.py`` metric block (the
guardian-owned semantics must survive the extraction). The task-only marginal
metrics (``weighted_pinball_loss``, ``zero_share_error``) stayed in the paper
repo, so they are not tested here; ``weighted_wasserstein1`` came with the view
tail block and is tested here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from popdgp import metrics


def test_wasserstein_identity_is_zero_on_equal_samples() -> None:
    """Identical weighted samples have zero Wasserstein-1 distance."""
    rng = np.random.default_rng(0)
    values = rng.lognormal(1.0, 0.5, 300)
    weights = rng.uniform(1.0, 5.0, 300)
    assert metrics.weighted_wasserstein1(
        values, values, imputed_weights=weights, donor_weights=weights
    ) == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_orders_shifts() -> None:
    """A larger location shift is a larger Wasserstein-1 distance."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, 500)
    donor_weights = rng.uniform(1.0, 4.0, 500)
    small = metrics.weighted_wasserstein1(base + 0.3, base, donor_weights=donor_weights)
    large = metrics.weighted_wasserstein1(base + 1.5, base, donor_weights=donor_weights)
    assert 0.0 < small < large


def test_energy_distance_is_zero_iff_same_and_orders_shifts() -> None:
    """Identical weighted samples score 0; larger shifts score strictly worse."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, (500, 3))
    weights = rng.uniform(1.0, 4.0, 500)
    assert metrics.energy_distance(
        base, base, imputed_weights=weights, holdout_weights=weights
    ) == pytest.approx(0.0, abs=1e-9)
    small = metrics.energy_distance(base + 0.3, base, holdout_weights=weights)
    large = metrics.energy_distance(base + 1.5, base, holdout_weights=weights)
    assert 0.0 < small < large


def test_prdc_coverage_detects_mode_collapse() -> None:
    """A modal-point candidate scores near-zero coverage; a true sample doesn't.

    This is the harness's reason for carrying coverage alongside marginal
    distances: a candidate collapsed onto the modal household can look tolerable
    on a marginal metric while covering none of the real manifold.
    """
    rng = np.random.default_rng(2)
    real = rng.normal(0.0, 1.0, (600, 2))
    faithful = rng.normal(0.0, 1.0, (600, 2))
    modal = np.tile(np.median(real, axis=0), (600, 1))

    good = metrics.prdc(real, faithful, seed=0)
    collapsed = metrics.prdc(real, modal, seed=0)
    assert good["coverage"] > 0.7
    assert collapsed["coverage"] < 0.1
    assert good["recall"] > collapsed["recall"]
    for value in (*good.values(), *collapsed.values()):
        assert math.isfinite(value) and value >= 0.0


def test_c2st_auc_separates_shifted_from_identical() -> None:
    """Same distribution scores near 0.5; a strongly shifted one near 1."""
    rng = np.random.default_rng(3)
    real = rng.normal(0.0, 1.0, (400, 2))
    same = rng.normal(0.0, 1.0, (400, 2))
    shifted = rng.normal(3.0, 1.0, (400, 2))
    assert abs(metrics.classifier_two_sample_auc(real, same, seed=0) - 0.5) < 0.12
    assert metrics.classifier_two_sample_auc(real, shifted, seed=0) > 0.9


def test_reweight_fragility_closed_form_and_landmine() -> None:
    """Uniform contributions match the closed form; a landmine approaches 1."""
    n = 100
    uniform = metrics.reweight_fragility(np.ones(n), np.ones(n), kappa=1.0)
    assert uniform == pytest.approx(1.0 / n)
    # Equal contributions at kappa=5: k^2*c / (k^2*c + (n-1)*c) = 25/124.
    boosted = metrics.reweight_fragility(np.ones(n), np.ones(n), kappa=5.0)
    assert boosted == pytest.approx(25.0 / 124.0)
    # One record carrying 100x the contribution of each other record.
    landmine = metrics.reweight_fragility(
        np.r_[np.ones(n - 1), 100.0], np.ones(n), kappa=5.0
    )
    assert landmine > 0.9
    assert metrics.reweight_fragility(np.zeros(4), np.ones(4)) == 0.0
    with pytest.raises(ValueError, match="kappa"):
        metrics.reweight_fragility(np.ones(4), np.ones(4), kappa=0.5)


def test_weight_validation_rejects_bad_vectors() -> None:
    """The shared weight validator refuses negative, mis-shaped, all-zero weights."""
    values = np.arange(5.0)
    with pytest.raises(ValueError, match="non-negative"):
        metrics.weighted_wasserstein1(values, values, imputed_weights=-np.ones(5))
    with pytest.raises(ValueError, match="shape"):
        metrics.weighted_wasserstein1(values, values, imputed_weights=np.ones(4))
    with pytest.raises(ValueError, match="sum to zero"):
        metrics.weighted_wasserstein1(values, values, imputed_weights=np.zeros(5))


def test_metric_dimension_mismatch_is_named() -> None:
    """The joint metrics refuse mismatched column counts with a clear message."""
    real = np.zeros((10, 2))
    synthetic = np.zeros((10, 3))
    with pytest.raises(ValueError, match="Dimension mismatch"):
        metrics.energy_distance(synthetic, real)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        metrics.prdc(real, synthetic)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        metrics.classifier_two_sample_auc(real, synthetic)


def test_prdc_requires_more_than_k_points() -> None:
    """PRDC needs more than k points per side after capping."""
    real = np.random.default_rng(0).normal(size=(4, 2))
    synthetic = np.random.default_rng(1).normal(size=(4, 2))
    with pytest.raises(ValueError, match="more than k"):
        metrics.prdc(real, synthetic, k=5)


def test_energy_distance_resample_cap_is_deterministic() -> None:
    """Above the cap, the seeded weight-proportional resample is reproducible."""
    rng = np.random.default_rng(4)
    imputed = rng.normal(0.0, 1.0, (5000, 2))
    holdout = rng.normal(0.2, 1.0, (5000, 2))
    a = metrics.energy_distance(imputed, holdout, max_points=1024, seed=7)
    b = metrics.energy_distance(imputed, holdout, max_points=1024, seed=7)
    assert a == b
