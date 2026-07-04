# PROGRESS — popdgp extraction (issue #1)

Ephemeral work log for crash-recovery. **Deleted in final pre-PR cleanup.**

## State
Resuming a killed predecessor. Extraction + tests were committed at tip
`65d2a90` (lead-salvaged WIP). Absorbed full state; verified extraction is a
faithful, behavior-preserving copy of imputation-paper's
`experiments/{metrics,views}.py` + the `scf_sample_reference` floor pattern
(`experiments/holdout.py` `split_frame` + `cli/harness.py` orchestration).

## Done
- Read all extracted modules (`src/popdgp/{__init__,metrics,views,floors}.py`)
  and all four test files against the imputation-paper source of truth.
- Confirmed metric semantics extracted verbatim: `metrics.py` = paper minus the
  task-only marginals (`weighted_pinball_loss`, `zero_share_error`,
  `DEFAULT_QUANTILE_GRID`), keeping `weighted_wasserstein1` (tail block needs
  it), energy/prdc/c2st/reweight_fragility + private helpers. `views.py`
  byte-identical logic (only import path + docstring xrefs differ).
- Confirmed `test_parity.py` asserts exact `==` equality vs the paper originals
  with a non-vacuous opportunistic skip (finds sibling checkout).

## Next
1. Run full test suite in worktree venv (parity + floors + metrics + views).
2. Run ruff check + ruff format --check.
3. Run `uv build` + `twine check` locally (mirror CI build job).
4. Draft JOSS `paper.md` + `paper.bib` (author = Max Ghenis + TODO note on
   authorship alignment; do NOT presume María Juaristi co-authorship).
5. Push after each coherent step; update this file each push.
6. Open PR via `gh pr create --body-file` (draft; lead reviews). Do NOT merge.

## Decisions / constraints honored
- Metric semantics are guardian-owned: extracted faithfully, no "improvements".
- Do NOT modify imputation-paper.
- Do NOT publish to PyPI (prep only).
- JOSS author list = Max Ghenis + TODO on pending authorship alignment.
