"""Parity against the imputation-paper originals.

Behavior-preserving means *parity*: on the same synthetic fixtures and seeds,
the extracted popdgp functions must reproduce the imputation-paper originals'
outputs to numerical tolerance -- bit-identical where the computation is
deterministic. This is the test that guards the extraction against a silent
metric-semantics drift (the failure mode the CONTRIBUTING guardian note calls
out).

The paper is *not* a dependency of popdgp, so these tests import it opportunis-
tically: if the imputation-paper source tree can be found (a sibling checkout,
or ``IMPUTATION_PAPER_SRC``), the module is added to ``sys.path`` and parity is
asserted; otherwise the whole module is skipped. CI installs only popdgp, so
parity runs where the paper is checked out (locally, and any environment that
sets the path) and skips cleanly elsewhere -- it never silently passes by not
comparing anything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _find_paper_src() -> Path | None:
    """Locate the imputation-paper ``src`` directory, or ``None``."""
    override = os.environ.get("IMPUTATION_PAPER_SRC")
    candidates = []
    if override:
        candidates.append(Path(override))
    # Sibling checkout: .../PolicyEngine/popdgp[/_worktrees/...] alongside
    # .../PolicyEngine/imputation-paper. Walk up looking for the sibling.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "imputation-paper" / "src")
        candidates.append(parent.parent / "imputation-paper" / "src")
    for candidate in candidates:
        if (candidate / "imputation_paper" / "experiments" / "metrics.py").exists():
            return candidate
    return None


_PAPER_SRC = _find_paper_src()
if _PAPER_SRC is not None and str(_PAPER_SRC) not in sys.path:
    sys.path.insert(0, str(_PAPER_SRC))

# Import the originals; skip the whole module if the paper is not available.
paper_metrics = pytest.importorskip(
    "imputation_paper.experiments.metrics",
    reason="imputation-paper source not found; set IMPUTATION_PAPER_SRC to run parity.",
)
paper_views = pytest.importorskip(
    "imputation_paper.experiments.views",
    reason="imputation-paper source not found; set IMPUTATION_PAPER_SRC to run parity.",
)

from popdgp import metrics as popdgp_metrics  # noqa: E402
from popdgp import views as popdgp_views  # noqa: E402


def _weighted_frame(seed: int, n: int = 900) -> pd.DataFrame:
    """A weighted multivariate fixture with a heavy-tailed target column."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, n)
    b = 0.7 * a + rng.normal(0.0, 0.5, n)
    wealth = np.expm1(rng.lognormal(0.5, 0.8, n)) * (1.0 + 0.3 * a)
    weight = rng.uniform(1.0, 6.0, n)
    return pd.DataFrame({"a": a, "b": b, "wealth": wealth, "weight": weight})


# --- Metric-level parity -------------------------------------------------


def test_energy_distance_parity() -> None:
    """Extracted energy_distance matches the original bit-for-bit."""
    x = _weighted_frame(1)
    y = _weighted_frame(2)
    xp = x[["a", "b", "wealth"]].to_numpy()
    yp = y[["a", "b", "wealth"]].to_numpy()
    for seed in (0, 3, 11):
        got = popdgp_metrics.energy_distance(
            xp,
            yp,
            imputed_weights=x["weight"].to_numpy(),
            holdout_weights=y["weight"].to_numpy(),
            seed=seed,
        )
        expected = paper_metrics.energy_distance(
            xp,
            yp,
            imputed_weights=x["weight"].to_numpy(),
            holdout_weights=y["weight"].to_numpy(),
            seed=seed,
        )
        assert got == expected


def test_energy_distance_parity_above_cap() -> None:
    """Parity holds through the seeded weight-proportional resample path."""
    x = _weighted_frame(4, n=5000)
    y = _weighted_frame(5, n=5000)
    xp = x[["a", "b", "wealth"]].to_numpy()
    yp = y[["a", "b", "wealth"]].to_numpy()
    got = popdgp_metrics.energy_distance(
        xp,
        yp,
        imputed_weights=x["weight"].to_numpy(),
        holdout_weights=y["weight"].to_numpy(),
        max_points=1500,
        seed=9,
    )
    expected = paper_metrics.energy_distance(
        xp,
        yp,
        imputed_weights=x["weight"].to_numpy(),
        holdout_weights=y["weight"].to_numpy(),
        max_points=1500,
        seed=9,
    )
    assert got == expected


def test_prdc_parity() -> None:
    """Extracted prdc matches the original on every component."""
    x = _weighted_frame(6)
    y = _weighted_frame(7)
    xp = x[["a", "b", "wealth"]].to_numpy()
    yp = y[["a", "b", "wealth"]].to_numpy()
    got = popdgp_metrics.prdc(
        xp,
        yp,
        real_weights=x["weight"].to_numpy(),
        synthetic_weights=y["weight"].to_numpy(),
        seed=0,
    )
    expected = paper_metrics.prdc(
        xp,
        yp,
        real_weights=x["weight"].to_numpy(),
        synthetic_weights=y["weight"].to_numpy(),
        seed=0,
    )
    assert got.keys() == expected.keys()
    for key in expected:
        assert got[key] == expected[key], key


def test_classifier_two_sample_auc_parity() -> None:
    """Extracted C2ST matches the original (same folds, same classifier seed)."""
    x = _weighted_frame(8)
    y = _weighted_frame(9)
    xp = x[["a", "b", "wealth"]].to_numpy()
    yp = y[["a", "b", "wealth"]].to_numpy()
    got = popdgp_metrics.classifier_two_sample_auc(
        xp,
        yp,
        real_weights=x["weight"].to_numpy(),
        synthetic_weights=y["weight"].to_numpy(),
        seed=0,
    )
    expected = paper_metrics.classifier_two_sample_auc(
        xp,
        yp,
        real_weights=x["weight"].to_numpy(),
        synthetic_weights=y["weight"].to_numpy(),
        seed=0,
    )
    assert got == expected


def test_weighted_wasserstein1_parity() -> None:
    """Extracted weighted_wasserstein1 matches the original."""
    x = _weighted_frame(10)
    y = _weighted_frame(11)
    got = popdgp_metrics.weighted_wasserstein1(
        x["wealth"].to_numpy(),
        y["wealth"].to_numpy(),
        imputed_weights=x["weight"].to_numpy(),
        donor_weights=y["weight"].to_numpy(),
    )
    expected = paper_metrics.weighted_wasserstein1(
        x["wealth"].to_numpy(),
        y["wealth"].to_numpy(),
        imputed_weights=x["weight"].to_numpy(),
        donor_weights=y["weight"].to_numpy(),
    )
    assert got == expected


def test_reweight_fragility_parity() -> None:
    """Extracted reweight_fragility matches the original."""
    x = _weighted_frame(12)
    for kappa in (1.0, 2.5, 5.0):
        got = popdgp_metrics.reweight_fragility(
            x["wealth"].to_numpy(), x["weight"].to_numpy(), kappa=kappa
        )
        expected = paper_metrics.reweight_fragility(
            x["wealth"].to_numpy(), x["weight"].to_numpy(), kappa=kappa
        )
        assert got == expected


def test_private_reductions_parity() -> None:
    """The private weighted reductions the tail block reaches into match."""
    x = _weighted_frame(13)
    points = x[["a", "b", "wealth"]].to_numpy()
    weights = x["weight"].to_numpy()
    got_mean, got_std = popdgp_metrics._weighted_moments(points, weights)
    exp_mean, exp_std = paper_metrics._weighted_moments(points, weights)
    assert np.array_equal(got_mean, exp_mean)
    assert np.array_equal(got_std, exp_std)

    q = np.array([0.1, 0.5, 0.9, 0.99])
    got_q = popdgp_metrics._weighted_quantile(x["wealth"].to_numpy(), weights, q)
    exp_q = paper_metrics._weighted_quantile(x["wealth"].to_numpy(), weights, q)
    assert np.array_equal(got_q, exp_q)


# --- Harness-level parity ------------------------------------------------


def test_score_view_parity_with_tail_block() -> None:
    """score_view (incl. the tail block) matches the original row for row."""
    candidate = _weighted_frame(14)
    holdout = _weighted_frame(15)
    cols = ["a", "b", "wealth"]
    cand_pts = candidate[cols].to_numpy()
    cand_w = candidate["weight"].to_numpy()
    hold_pts = holdout[cols].to_numpy()
    hold_w = holdout["weight"].to_numpy()
    target_dims = {"wealth": 2}
    got = popdgp_views.score_view(
        cand_pts, cand_w, hold_pts, hold_w, seed=0, target_dims=target_dims
    )
    expected = paper_views.score_view(
        cand_pts, cand_w, hold_pts, hold_w, seed=0, target_dims=target_dims
    )
    assert got.keys() == expected.keys()
    for key in expected:
        assert got[key] == expected[key], key


def test_harness_scorecard_parity() -> None:
    """harness_scorecard produces identical long rows across the two packages."""
    candidate = _weighted_frame(16)
    holdout = _weighted_frame(17)

    view_popdgp = popdgp_views.SurveyView(
        name="scf",
        columns=("a", "b", "wealth"),
        weight_column="weight",
        target_columns=("wealth",),
    )
    view_paper = paper_views.SurveyView(
        name="scf",
        columns=("a", "b", "wealth"),
        weight_column="weight",
        target_columns=("wealth",),
    )
    got = popdgp_views.harness_scorecard(
        candidate, "weight", [view_popdgp], {"scf": holdout}, seed=0
    )
    expected = paper_views.harness_scorecard(
        candidate, "weight", [view_paper], {"scf": holdout}, seed=0
    )
    assert len(got) == len(expected)
    got_by_metric = {r["metric"]: r["value"] for r in got}
    exp_by_metric = {r["metric"]: r["value"] for r in expected}
    assert got_by_metric.keys() == exp_by_metric.keys()
    for metric in exp_by_metric:
        assert got_by_metric[metric] == exp_by_metric[metric], metric
