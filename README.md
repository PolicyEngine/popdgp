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

## Lineage and status

Successor to the original `CosilicoAI/popdgp` (which also tried to *build*
the generator; this package deliberately does not — populace and anyone
else's generators are contestants, popdgp is the referee). First application
and reference implementation:
[imputation-paper](https://github.com/PolicyEngine/imputation-paper)
(`experiments/views.py`, `experiments/metrics.py`), from which the package is
being extracted — see the issues. Targets: PyPI, a JOSS software note, and a
methods paper (Survey Methodology / JOS) formalizing the framework with a
designed metric-blind-spot simulation study and a multi-view instantiation.
