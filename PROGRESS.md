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

## Verified (this session)
- [x] 1. Full test suite in worktree venv: **29 passed in 143.91s** (Python
  3.13.9). Breakdown: floors 6, metrics 10, parity 9, views 4. The 9 parity
  tests RAN (not skipped) against the sibling
  `~/PolicyEngine/imputation-paper` checkout and asserted exact `==` equality;
  extraction confirmed bit-identical to the paper originals.
- [x] 2. `ruff check .` → all checks passed. `ruff format --check .` → 8 files
  already formatted. Both exit 0.
- [x] 3. `uv build` → sdist + wheel built. `uvx twine check dist/*` → both
  PASSED. Exit 0. (`dist/` is gitignored; `uv.lock` gitignored+untracked.)

- [x] 4. Drafted JOSS `paper.md` + `paper.bib`. Author = Max Ghenis (sole),
  ORCID 0000-0002-1335-8277 (verified from PolicyEngine's own
  policyengine.py JOSS submission), affiliation PolicyEngine, Washington DC.
  TWO `TODO(authorship)` comments in paper.md (author block + Acknowledgements)
  flag pending authorship alignment; NO presumed co-authors (María Juaristi
  not listed). Canonical citations, all verified this session:
  energy distance = Szekely & Rizzo (2013) JSPI 143(8):1249-1272
  doi:10.1016/j.jspi.2013.03.018; PRDC = Naeem et al. (2020) ICML PMLR
  v119:7176-7185; C2ST = Lopez-Paz & Oquab (2017) ICLR arXiv:1610.06545;
  Wasserstein = Ramdas et al. (2017) Entropy 19(2):47 doi:10.3390/e19020047;
  plus @imputation_paper software entry. All 5 bib keys cross-checked to
  resolve. YAML front-matter parses cleanly (pyyaml); TODO comments stripped,
  do not leak into values. Every technical claim cross-checked against the
  source (weight->uniform fallback, coverage reweight-invariance, C2ST
  equal-mass, uncapped tail block, strictly-proper energy distance).
- [x] LICENSE added (MIT), matching the pre-existing `license = {text="MIT"}`
  in pyproject. JOSS requires an OSI license file and the repo had none.
  Now packaged into sdist+wheel (`License-File: LICENSE` in metadata);
  lint/build/twine re-run green with it present. FLAG FOR LEAD: org core
  packages (policyengine.py) are AGPL-3.0; MIT here follows the committed
  pyproject decision, not a new choice by me -- confirm MIT is intended.

## Next
5. Push after each coherent step; update this file each push. (Done through 4.)
6. Open PR via `gh pr create --body-file` (draft; lead reviews). Do NOT merge.
   Delete this file in the final cleanup commit.

## Decisions / constraints honored
- Metric semantics are guardian-owned: extracted faithfully, no "improvements".
- Do NOT modify imputation-paper.
- Do NOT publish to PyPI (prep only).
- JOSS author list = Max Ghenis + TODO on pending authorship alignment.
