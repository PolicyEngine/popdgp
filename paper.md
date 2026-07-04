---
title: "popdgp: A referee for candidate synthetic populations under survey weights"
tags:
  - Python
  - synthetic data
  - survey weights
  - generative model evaluation
  - microsimulation
  - energy distance
authors:
  # TODO(authorship): confirm the final author list and order before submission.
  # Seeded as Max Ghenis (sole author) per the extraction brief; the harness
  # originated in the imputation-paper repository, so co-authorship of that
  # work (e.g. contributors to experiments/metrics.py and experiments/views.py)
  # may need to be reflected here. Do not presume co-authors without sign-off.
  - name: Max Ghenis
    orcid: 0000-0002-1335-8277
    affiliation: '1'
    corresponding: true
affiliations:
  - name: PolicyEngine, Washington, DC, United States
    index: '1'
date: 4 July 2026
bibliography: paper.bib
---

# Summary

`popdgp` scores a candidate synthetic population against the surveys that
observe it. The framing is a population data-generating process seen through
*views*: one latent population produces individuals, and each survey is a view
of that population -- a variable subset, a sampling design, a measurement idiom.
A survey's holdout is a weighted sample from its view. A candidate is any
weighted population file (the output of an imputation model, a reweighting
routine, or a generative model). `popdgp` projects the candidate through each
view and scores its projection against that view's holdout, in the view's own
variable space, under both sides' survey weights.

The package is evaluation-only: it does not build populations, and it takes no
dependency on any particular generator. It exposes three modules with a plain
pandas/NumPy API. `popdgp.metrics` provides the weighted distributional
scores -- a weighted energy distance [@szekely2013energy], the
precision/recall/density/coverage family [@naeem2020prdc], a weighted
classifier two-sample AUC [@lopezpaz2017c2st], a weighted Wasserstein-1
distance [@ramdas2017wasserstein], and a reweighting-fragility diagnostic.
`popdgp.views` defines a `SurveyView` and the projection-and-score harness that
produces a per-view scorecard. `popdgp.floors` computes sampling-noise floors
by splitting a survey into complementary halves and scoring one half against the
other, so each axis of the scorecard has an anchor: the score an independent
draw from the same population still incurs.

# Statement of need

Microsimulation and survey-analysis workflows increasingly run on *synthetic*
or *enhanced* microdata: a base survey augmented with imputed variables,
reweighted to hit external totals, or replaced outright by a generated file.
Whether such a file is a faithful stand-in for the population is usually checked
one margin at a time -- a mean here, a total there -- which cannot detect a file
that matches every margin while getting the joint distribution, the tails, or
the dependence structure wrong. Two properties that matter specifically for
population files are routinely neglected. First, the file is weighted, so any
honest comparison must be between *weighted* measures on both sides; an
unweighted metric scores the wrong distribution. Second, the file is a draw from
an estimated distribution, so the question is distributional fidelity, not
point accuracy against a single reference.

General-purpose generative-model metrics address the second property but assume
unweighted samples and typically operate on a single reference set, and their
sample-geometry variants are weakest exactly where economic variables carry
their policy signal: deep in a heavy right tail. `popdgp` targets the gap. Every
metric accepts survey weights (and reduces to the unweighted computation when
weights are uniform, so the weighted/unweighted contrast is a single code path).
The harness scores a candidate against *multiple* survey views without ever
requiring cross-survey consistency -- each view asks only that the candidate
explain that survey in its own idiom -- which is what makes the tool usable when
surveys disagree with one another. And each view carries an explicit
tail-sensitive block on the imputed columns, computed on the full weighted
samples rather than a subsample, because the tail is exactly what subsampled
sample-geometry metrics blur.

The design is deliberately non-self-referential: holdouts must never have been
used upstream in fitting or calibration, so the harness is a genuine test
surface rather than a re-scoring of training data. Because the scorer is
separated from any generator, competing methods can be compared on the same
axes, and the sampling-noise floors make the scores interpretable: the gap
between a candidate and the floor is the part of its score that is not
irreducible sampling noise.

`popdgp` was extracted, behavior-preserving, from the evaluation harness of the
PolicyEngine imputation study [@imputation_paper], where it scored candidate
enhanced-survey populations. Packaging it independently lets other projects
reuse the harness without depending on that study, and gives the framework a
citable reference implementation.

# Metrics

Each metric is a weighted estimator; the descriptions below state what the
weighted quantity is and why it belongs on the scorecard.

- **Weighted energy distance** [@szekely2013energy]. The squared energy
  distance $D^2(P, Q) = 2\,\mathbb{E}\lVert X - Y\rVert - \mathbb{E}\lVert X -
  X'\rVert - \mathbb{E}\lVert Y - Y'\rVert$ is evaluated under the weighted
  empirical measures, with columns standardized by the holdout's weighted
  moments so scores are comparable across views on different scales. It is
  non-negative and zero if and only if the distributions coincide, and it
  induces a strictly proper scoring rule: a candidate that hedges toward modal
  households scores strictly worse than one matching the full joint. This is the
  harness's primary weight-sensitive, joint-distribution block.
- **Precision, recall, density, and coverage** [@naeem2020prdc]. Neighbourhood
  radii are $k$-th-nearest-neighbour distances (support geometry); the averages
  over points are weighted. Coverage is the anti-mode-collapse axis -- the
  weighted fraction of real points with a candidate neighbour inside their
  radius -- and, because it depends on the candidate only through which records
  exist and not how they are weighted, it is invariant to any reweighting of the
  candidate. It is therefore the calibration-blind block: a diagnostic of
  support that a reweighting cannot game.
- **Weighted classifier two-sample AUC** [@lopezpaz2017c2st]. A gradient-boosted
  classifier is trained to tell holdout rows from candidate rows, with sample
  weights normalized so each side carries equal total mass, and the
  stratified cross-validated AUC is reported. An AUC of 0.5 means nothing the
  classifier finds separates the two files -- the omnibus complement to the
  per-axis metrics.
- **Weighted Wasserstein-1 distance** [@ramdas2017wasserstein]. The
  earth-mover distance between two one-dimensional weighted samples, obtained by
  integrating the absolute difference of their weighted quantile functions. Per
  imputed target it is scaled by the holdout's weighted standard deviation to
  give a unit-free marginal-fit summary; the view's tail block reports it
  alongside weighted q90 and q99 ratios (candidate over holdout).
- **Reweighting fragility.** A stress diagnostic, independent of any view: the
  worst-case single-record share of a population aggregate over the family of
  bounded multiplicative reweightings $w_i \mapsto m_i w_i$ with $m_i \in
  [1/\kappa, \kappa]$. The worst case has a closed form, so the diagnostic is
  exact. It flags the "landmine" failure mode -- a record carrying an extreme
  value at low weight that is invisible in the shipped aggregates but can
  dominate one after a downstream reweighting.

# Functionality and use

The public surface is small. A `SurveyView` names the columns a survey
observes, its weight column, and which of those columns were imputed (the
targets that receive the tail block). `harness_scorecard` takes a candidate
weighted file, a set of views, and the matching holdouts, and returns
long-format rows `{view, metric, value}`. `sample_reference_scorecard` returns
the same schema for the floors, so floor rows and candidate rows concatenate
directly for plotting or tabulation. The individual metric functions in
`popdgp.metrics` can also be called on their own for ad hoc comparisons. The
package depends only on NumPy, pandas, scikit-learn, and SciPy, supports Python
3.11 through 3.13, and is installed from PyPI with `pip install popdgp`.

Fidelity to the originating harness is enforced by a parity test suite that,
when the `imputation-paper` source is available, asserts that every extracted
function reproduces the original's output to numerical equality on shared
fixtures and seeds; it skips cleanly when the source is absent, so it never
passes vacuously.

# Limitations

The current release emulates a survey's design by *weighting*: a view compares
weighted measures on both sides but does not reproduce a survey's record-level
selection mechanism. Modeling that selection is future work and is noted as such
in the harness documentation. The metrics are sample-based estimators, so the
subsampled joint metrics carry Monte Carlo noise above their size cap (the
sampling-noise floors quantify this); the tail block is computed without a cap
for that reason.

# Acknowledgements

`popdgp` was extracted from the evaluation harness of the PolicyEngine
imputation study [@imputation_paper]; the author thanks the PolicyEngine team
and that study's contributors. <!-- TODO(authorship): if any of those
contributors should be co-authors rather than acknowledged, move them to the
author list above before submission. -->

# AI usage disclosure

The author used a generative AI assistant (Anthropic's Claude) to help with the
mechanical extraction, packaging, and drafting of this software. The author
reviewed, edited, and validated all outputs, and made all design and modeling
decisions. The author is responsible for the correctness of the submitted
materials.

# References
