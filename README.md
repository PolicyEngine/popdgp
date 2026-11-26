# popdgp

The population data-generating process, as a referee: an eval-only harness
that scores any candidate synthetic population against the surveys that
observe it.

One latent population produces individuals; each survey is a *view* of it — a
variable subset, a sampling design, a measurement idiom — and a survey's
holdout is a weighted sample from that view. popdgp projects a candidate
weighted file through each view and scores it against that view's holdout in
the view's own variable space, with four blocks per view:

- **weighted energy distance** — strictly proper joint score (a candidate
  hedged toward modal households scores strictly worse);
- **weighted PRDC coverage** — support geometry; invariant to any reweighting
  of the candidate, so it is the calibration-blind block;
- **weighted classifier two-sample AUC** — the omnibus distinguishability
  check (0.5 = indistinguishable);
- **an uncapped tail block** on imputed columns (per-variable weighted
  W1/sd and q90/q99 ratios) — added after demonstrating that capped
  sample-geometry metrics certify files whose top percentile is wrong by a
  factor of two.

Sampling-noise **floors** (a survey's own complementary split scored as a
candidate) anchor every axis. Generator-agnostic and non-self-referential:
holdouts appear nowhere upstream of the candidates being scored.

## Install

```bash
pip install popdgp
```

popdgp needs Python 3.11 or later and depends on NumPy, pandas, SciPy and
scikit-learn.

## Quick start

A view names the columns a survey observes, its weight column, and which
columns were imputed. `harness_scorecard` scores a candidate file against each
view's holdout, and `sample_reference_scorecard` gives the floor for the same
view. Both return long-format rows `{view, metric, value}`.

```python
"""Score a candidate population against one survey view, beside its floor."""

import numpy as np
import pandas as pd

from popdgp import (
    SurveyView,
    harness_scorecard,
    sample_reference_scorecard,
    split_frame,
)


def draw(seed: int, n: int = 1000) -> pd.DataFrame:
    """A weighted sample of age and wealth from a toy population."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 80, n)
    wealth = np.exp(rng.normal(9 + 0.03 * age, 1.0))
    weight = rng.uniform(1, 5, n)
    return pd.DataFrame({"age": age, "wealth": wealth, "weight": weight})


survey = draw(seed=1)  # the survey that observes the population
candidate = draw(seed=2)  # the file to score; here, another draw

# A view names the columns a survey observes, its weight column, and which
# columns were imputed. Imputed columns get the tail block.
view = SurveyView(
    name="survey",
    columns=("age", "wealth"),
    weight_column="weight",
    target_columns=("wealth",),
)

# Hold out 20% of the survey. The holdout must not be used to build the
# candidate.
split = split_frame(survey, seed=0)
scores = harness_scorecard(candidate, "weight", [view], {"survey": split.test}, seed=0)

# The floor scores the survey's other 80% against the same holdout: what an
# independent sample of the same population still incurs.
floor = sample_reference_scorecard({"survey": survey}, [view], seed=0)

table = pd.DataFrame(scores).merge(
    pd.DataFrame(floor), on=["view", "metric"], suffixes=("_candidate", "_floor")
)
print(table.to_string(index=False))
```

The same script is in
[`examples/quickstart.py`](https://github.com/PolicyEngine/popdgp/blob/main/examples/quickstart.py).

## Lineage and status

Successor to the original `CosilicoAI/popdgp` (which also tried to *build*
the generator; this package deliberately does not — Microcosm and anyone
else's generators are contestants, popdgp is the referee). First application
and reference implementation:
[imputation-paper](https://github.com/PolicyEngine/imputation-paper)
(`experiments/views.py`, `experiments/metrics.py`), from which the package is
being extracted — see the issues. Targets: PyPI, a JOSS software note, and a
methods paper (Survey Methodology / JOS) formalizing the framework with a
designed metric-blind-spot simulation study and a multi-view instantiation.
