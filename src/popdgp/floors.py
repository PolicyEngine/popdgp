"""Sampling-noise floors: a survey's own complementary split, scored as a candidate.

Every axis of the scorecard needs an anchor. A candidate's energy distance to a
holdout is not zero even when the candidate *is* an independent sample from the
same population -- two finite weighted samples differ by sampling noise alone.
The floor measures that irreducible noise directly: split a survey into two
complementary halves, treat one half as the candidate population and the other
as the holdout, and score it through the survey's own view. No generator can
beat this floor; the gap between a candidate and the floor is the part of its
score that is *not* sampling noise.

This is the ``scf_sample_reference`` pattern from the paper's harness, made
generator-agnostic: it is a property of the surveys and their views, computed
without reference to any candidate under test. :func:`split_frame` is the
single-split primitive (a pure function of its seed, so a floor is reproducible
and paired across runs); :func:`sample_reference_scorecard` applies it to a set
of views and scores each survey's train half against its own test half.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from popdgp.views import SurveyView, harness_scorecard

__all__ = ["Split", "split_frame", "sample_reference_scorecard"]


@dataclass(frozen=True)
class Split:
    """One complementary split of a survey table.

    Attributes:
        seed: The seed that produced this split.
        train: The rows scored *as the candidate* (the reference population).
        test: The complementary rows scored *as the holdout*.
    """

    seed: int
    train: pd.DataFrame
    test: pd.DataFrame


def split_frame(
    frame: pd.DataFrame, *, holdout_frac: float = 0.2, seed: int = 0
) -> Split:
    """Split ``frame`` into two complementary halves, deterministically.

    Args:
        frame: The full survey table.
        holdout_frac: Fraction of rows placed in the holdout (test) half.
        seed: Seed for the permutation.

    Returns:
        A :class:`Split`; ``train`` and ``test`` partition ``frame`` and each is
        row-index-reset. A pure function of ``(frame, holdout_frac, seed)``.

    Raises:
        ValueError: If ``holdout_frac`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}.")
    rng = np.random.default_rng(seed)
    n = len(frame)
    order = rng.permutation(n)
    n_holdout = int(round(n * holdout_frac))
    test_idx = np.sort(order[:n_holdout])
    train_idx = np.sort(order[n_holdout:])
    return Split(
        seed=seed,
        train=frame.iloc[train_idx].reset_index(drop=True),
        test=frame.iloc[test_idx].reset_index(drop=True),
    )


def sample_reference_scorecard(
    surveys: Mapping[str, pd.DataFrame],
    views: Iterable[SurveyView],
    *,
    holdout_frac: float = 0.2,
    k: int = 5,
    max_points: int = 2048,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Score each survey's own train half against its test half (the floor).

    For each view, its survey table is split into complementary halves by
    :func:`split_frame`; the train half becomes the candidate and the test half
    the holdout, and the pair is scored through that view alone. The result is
    the sampling-noise floor for every axis: the score a perfect generator (an
    independent draw from the same population) still incurs.

    Args:
        surveys: View name -> the full survey table for that view (carrying the
            view's columns and its ``weight_column``). Each is split into a
            candidate half and a holdout half.
        views: The survey views to compute floors for.
        holdout_frac: Fraction of each survey placed in the holdout half.
        k: PRDC neighbour rank.
        max_points: Pairwise-metric size cap.
        seed: Seed for the split and for resampling/C2ST folds; the split is
            threaded with the same seed so the floor is reproducible.

    Returns:
        Long-format rows ``{"view", "metric", "value"}``, one per view and
        metric -- the same schema :func:`popdgp.views.harness_scorecard`
        returns, so floor rows and candidate rows concatenate directly.

    Raises:
        KeyError: If a view has no survey table in ``surveys``.
        ValueError: If a projection fails (missing/non-numeric columns).
    """
    rows: list[dict[str, Any]] = []
    for view in views:
        if view.name not in surveys:
            raise KeyError(
                f"No survey table for view {view.name!r}; have {sorted(surveys)}."
            )
        split = split_frame(surveys[view.name], holdout_frac=holdout_frac, seed=seed)
        rows.extend(
            harness_scorecard(
                split.train,
                view.weight_column,
                [view],
                {view.name: split.test},
                k=k,
                max_points=max_points,
                seed=seed,
            )
        )
    return rows
