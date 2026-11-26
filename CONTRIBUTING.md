# Contributing

## Review tiers

Work in this portfolio routes by issue label: `tier:fable` (frontier-model:
design, claim-bearing prose, statistical semantics) versus `tier:standard`
(spec-complete build/assembly; acceptance tests judge the output). Full
doctrine: PolicyEngine/microcosm#305.

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

## Releasing

Releases go to PyPI from `.github/workflows/release.yml`, by PyPI trusted
publishing (no stored token).

1. Set the new version in `pyproject.toml` and in `src/popdgp/__init__.py`
   (a test checks that the two agree), and merge.
2. Publish a GitHub Release whose tag is that version (`0.2.0` or `v0.2.0`).
   The workflow tests, builds, runs the quick start against the built wheel,
   and uploads to PyPI. It stops if the tag and the version differ.

Running the workflow by hand builds and checks the distribution without
publishing.

One-time setup, by an owner of the PyPI project: add a trusted publisher on
PyPI for owner `PolicyEngine`, repository `popdgp`, workflow `release.yml`,
environment `pypi`. Before the first release this is a "pending publisher",
which also reserves the name.

A release changes what downstream users pin. Say in the release notes whether
any metric's output changes for the same inputs and seed.
