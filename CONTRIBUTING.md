# Contributing

## Review tiers

Work in this portfolio routes by issue label: `tier:fable` (frontier-model:
design, claim-bearing prose, statistical semantics) versus `tier:standard`
(spec-complete build/assembly; acceptance tests judge the output). Full
doctrine: PolicyEngine/populace#305.

## Guardian files

Changes to the following require frontier-model (or maintainer) review
regardless of green CI — their failure mode is plausible-and-silently-wrong,
and this portfolio has already caught three such bugs that passed lint and
tests (an int16 overflow silently dropping 29% of SCF households, joint
metrics certifying a 2x tail error, and best-value bolding with inverted
semantics):

- the metrics (energy / PRDC / C2ST / tail block / fragility) and their
  weighted reductions
- view projection and floor construction
- any holdout or split logic
