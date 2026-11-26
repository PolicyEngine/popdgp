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
