# Pilot of the 120,000-series plan — 2026-11-05

The full plan was materialized as 120 native requested-generation configs totaling exactly
120,000 series. Full candidate execution was not started.

The local smoke pilot requested two series per job (240 total) using BeTiSe 0.6.1:

- 118/120 jobs passed structural, composition, provisional stationarity-label and categorical-count checks.
- 236 series passed; two jobs failed during generation.
- This is a limited smoke test: at most two categorical recipes per job are exercised.

## Failed jobs

- `short__single_seasonality__batch01`: No valid configured period found for single_seasonality.
- `short__multiple_seasonality__batch01`: multiple_seasonality needs 2 valid periods, but only [7] are available.

The short contextual-anomaly job passed for this seed, unlike the earlier pilot.
That does not resolve its seasonal-base feasibility issue: lengths below 72 remain
incompatible with the configured period 12 and six-cycle requirement. The plan
therefore still flags three short-seasonal job definitions for review.

Seven plan tests passed, covering the exact 120,000 total, 20,000/20,000 quotas per
length, 1,000 per job, recipe budget sharing, distinct seeds, explicit balance batches,
exports and rejection of invalid allocations/duplicate canonical scenarios.

No library release, full dataset generation or Zenodo upload was performed.

## Subsequent paper-coverage correction

The full-plan smoke results above precede the Section 4 compound correction:
`linear_mean_point` was replaced with `stochastic_mean_point` (random walk + mean
shift + point anomaly). The corrected compound was smoke-tested separately at all
three length categories with two series each: **3/3 jobs, 6/6 series passed**.
The other 117 job definitions are unchanged. This is not a claim that every
categorical variant or scientific quality criterion has been validated.

The plan test suite now has **9 passing tests**, including checks for every named
paper base/overlay family and the explicit Section 4 compound in each length group.
See `PAPER_COVERAGE.md` for the distinction between taxonomy coverage, historical
appendix templates and unresolved numerical/parameter compatibility issues.
