"""popdgp: the population data-generating process, as a referee.

An eval-only harness that scores any candidate synthetic population against the
surveys that observe it. One latent population produces individuals; each survey
is a *view* of it (a variable subset, a sampling design, a measurement idiom),
and a survey's holdout is a weighted sample from that view. popdgp projects a
candidate weighted file through each view and scores it against that view's
holdout in the view's own variable space.

Public surface:

* :mod:`popdgp.metrics` -- the weighted joint/tail/stress metrics
  (:func:`~popdgp.metrics.energy_distance`, :func:`~popdgp.metrics.prdc`,
  :func:`~popdgp.metrics.classifier_two_sample_auc`,
  :func:`~popdgp.metrics.weighted_wasserstein1`,
  :func:`~popdgp.metrics.reweight_fragility`);
* :mod:`popdgp.views` -- the population-view harness
  (:class:`~popdgp.views.SurveyView`, :func:`~popdgp.views.project_view`,
  :func:`~popdgp.views.score_view`, :func:`~popdgp.views.harness_scorecard`);
* :mod:`popdgp.floors` -- sampling-noise floors
  (:func:`~popdgp.floors.split_frame`,
  :func:`~popdgp.floors.sample_reference_scorecard`).
"""

from __future__ import annotations

from popdgp import floors, metrics, views
from popdgp.floors import Split, sample_reference_scorecard, split_frame
from popdgp.metrics import (
    classifier_two_sample_auc,
    energy_distance,
    prdc,
    reweight_fragility,
    weighted_wasserstein1,
)
from popdgp.views import SurveyView, harness_scorecard, project_view, score_view

__version__ = "0.1.0"

__all__ = [
    # submodules
    "metrics",
    "views",
    "floors",
    # metrics
    "energy_distance",
    "prdc",
    "classifier_two_sample_auc",
    "weighted_wasserstein1",
    "reweight_fragility",
    # views
    "SurveyView",
    "project_view",
    "score_view",
    "harness_scorecard",
    # floors
    "Split",
    "split_frame",
    "sample_reference_scorecard",
]
