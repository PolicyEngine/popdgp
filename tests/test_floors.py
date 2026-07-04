"""Sampling-noise floors: split determinism and the reference-scorecard pattern."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from popdgp.floors import sample_reference_scorecard, split_frame
from popdgp.views import SurveyView, harness_scorecard


def _population(seed: int, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, n)
    c = -0.5 * a + rng.normal(0.0, 0.8, n)
    weight = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({"a": a, "c": c, "weight": weight})


def test_split_frame_is_a_deterministic_partition() -> None:
    """train and test partition the frame, and the split is a function of seed."""
    frame = _population(seed=1)
    split_a = split_frame(frame, seed=3)
    split_b = split_frame(frame, seed=3)
    # Reproducible.
    assert split_a.train.equals(split_b.train)
    assert split_a.test.equals(split_b.test)
    # Partition: disjoint and covering, no duplicates.
    assert len(split_a.train) + len(split_a.test) == len(frame)
    combined = pd.concat([split_a.train, split_a.test]).sort_values("a")
    expected = frame.sort_values("a")
    assert np.allclose(combined["a"].to_numpy(), expected["a"].to_numpy())
    # Default holdout fraction is 0.2.
    assert len(split_a.test) == round(0.2 * len(frame))


def test_split_frame_different_seeds_differ() -> None:
    """Different seeds give different partitions."""
    frame = _population(seed=1)
    a = split_frame(frame, seed=0)
    b = split_frame(frame, seed=1)
    assert not a.test.equals(b.test)


def test_split_frame_validates_fraction() -> None:
    """holdout_frac must be strictly inside (0, 1)."""
    frame = _population(seed=1)
    with pytest.raises(ValueError, match="holdout_frac"):
        split_frame(frame, holdout_frac=0.0)
    with pytest.raises(ValueError, match="holdout_frac"):
        split_frame(frame, holdout_frac=1.0)


def test_sample_reference_matches_manual_split_and_score() -> None:
    """The floor helper equals the split-then-harness_scorecard pattern.

    This is exactly the paper harness's ``scf_sample_reference`` block:
    ``split = split_frame(frame, seed=seed); harness_scorecard(split.train, ...,
    {name: split.test}, seed=seed)``. The helper must reproduce it row for row.
    """
    survey = _population(seed=2, n=1200)
    view = SurveyView(
        name="scf", columns=("a", "c"), weight_column="weight", target_columns=("c",)
    )

    got = sample_reference_scorecard({"scf": survey}, [view], seed=5)

    split = split_frame(survey, seed=5)
    expected = harness_scorecard(
        split.train, "weight", [view], {"scf": split.test}, seed=5
    )
    assert len(got) == len(expected)
    got_by_metric = {r["metric"]: r["value"] for r in got}
    exp_by_metric = {r["metric"]: r["value"] for r in expected}
    assert got_by_metric.keys() == exp_by_metric.keys()
    for metric in exp_by_metric:
        assert got_by_metric[metric] == exp_by_metric[metric], metric


def test_sample_reference_requires_a_survey_per_view() -> None:
    """A view without a survey table is refused, named."""
    view = SurveyView(name="scf", columns=("a", "c"), weight_column="weight")
    with pytest.raises(KeyError, match="No survey table"):
        sample_reference_scorecard({"cps": _population(seed=1)}, [view])


def test_floor_is_a_low_score_reference() -> None:
    """The floor's own split scores as near-indistinguishable (low, not zero).

    Two complementary halves of one survey differ only by sampling noise, so
    the C2ST AUC sits near 0.5 and coverage is high -- the anchor no candidate
    can beat.
    """
    survey = _population(seed=3, n=3000)
    view = SurveyView(name="scf", columns=("a", "c"), weight_column="weight")
    rows = sample_reference_scorecard({"scf": survey}, [view], seed=0)
    by_metric = {r["metric"]: r["value"] for r in rows}
    assert abs(by_metric["c2st_auc"] - 0.5) < 0.15
    assert by_metric["prdc_coverage"] > 0.5
