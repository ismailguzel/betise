# Paper-to-config coverage audit

Source: *BeTiSe: A Comprehensive Benchmark Dataset for Univariate Time Series
Stationarity and Structural Analysis*, DOI
[10.3390/cmsf2026016024](https://doi.org/10.3390/cmsf2026016024), Table 2 and
Sections 3–4. The supplied publication PDF was read directly; its hash is in scope.

## Named property families and model subtypes

| Paper class/model family | Required components | Active config coverage |
|---|---|---|
| Stationary bases (§3.1) | White noise, AR, MA, ARMA | `white_noise`, `ar`, `ma`, `arma` |
| Deterministic trends (§3.2) | Linear, quadratic, cubic, exponential, damped | All five overlays selected |
| Stochastic trends (§3.2) | RW, RW with drift, ARI, IMA, ARIMA | All five base types selected |
| Seasonality (§3.3) | Single frequency, multiple frequencies, SARMA, SARIMA | All four base types selected |
| Volatility (§3.3) | ARCH, GARCH, EGARCH, APARCH | All four base types selected |
| Anomalies (§3.4) | Point, collective, contextual | All three overlays selected |
| Structural breaks (§3.4) | Mean, variance, trend shift | All three overlays selected |
| Combined example (§4) | Stochastic trend + structural break + point anomaly | `random_walk` + `mean_shift` + `point_anomaly` |

All the above occur in each length category. The combined job is
`<length>__stochastic_mean_point__batch01`. This replaces the previous
deterministic linear-trend compound job without changing the 120,000 total.
The specific stochastic process and break subtype are representative choices;
Section 4 does not fix them. Other combinations include ARMA+GARCH and MA+EGARCH.

The plan builder checks that the required base and overlay sets and Section 4
example remain represented. The taxonomy is multi-property: e.g. volatility and
stationarity are not mutually exclusive classes. No 10-class or 39-class taxonomy
from the separate downstream classification project is imposed here.

## Categories are present; executability and scientific quality remain distinct

- Single/multiple anomaly and break modes are selected from the library's categorical
  recipes, within each 1,000-series budget. Up to 32 recipes/job are sampled, not all
  variants of every combination. Short-length rules restrict some multiple-break modes.
- Short single/multiple seasonality has known numerical feasibility failures under
  current period defaults. The dataset adapter resolves these for this package with
  explicit periods 7 (single) and 3/7 (multiple), preserving the six-cycle rule.
  Published library defaults are unchanged. Boundary tests cover lengths 50, 71 and 100.
- The library requires `linear_trend` before `trend_shift`, so the trend-shift job
  explicitly includes both. Contextual anomalies use a seasonal base and the code
  applies a local seasonal-pattern inversion, consistent with the paper's description.
- The current damped generator implements `(a*t+b)*exp(-damping_rate*t)` plus noise,
  matching the functional form in Equation (1). This does not establish equality of
  the historical numerical parameter distributions.
- Stationarity flags must be audited against the generation parameters. Conditional
  volatility alone does not make a process non-stationary. Planned binary quotas are
  provisional; labels must not be rewritten to achieve balance.

## What is not claimed to be identical

The old repository appendix is supplementary historical material, not an exhaustive
configuration list printed in the paper. Some of its representative templates are
not directly valid under the current library rules:

| Archived appendix example | Current status |
|---|---|
| Template 7: AR+GARCH with variance shift | Current rules reject variance shift with a volatility base |
| Template 17: cubic trend + trend shift + seasonal/volatility/stochastic bases + point anomaly | Current rules reject trend shift without `linear_trend`; this exact combination is not represented |
| Templates 13–16: additional seasonal/anomaly combinations | Not all included as separate jobs in the compact budget; all anomaly and seasonal families are present |
| Template 18: deterministic linear + mean shift + point anomaly | Replaced by the compound example explicitly stated in the paper's Section 4 |

Thus **all named primary classes/model subtypes are included**, but not every
historical appendix combination. Supporting those exact examples would require an
explicit review of library rules and budget allocation, not a metadata rename.

The original complete configuration/seed manifest is not provided in the paper.
The new 120,000 total, equal counts per length, chosen bases for overlays and sampled
categorical proportions do not prove original-dataset identity. Numerical defaults
(coefficients, strength, normalization, seasonal periods) are from BeTiSe 0.6.1 and
need their own parameter-level compatibility audit before full production.

## Metadata migration

The historical CSV flags and folder labels may be replaced by the current Parquet
schema. Preserve the meaning of each property, base model, component order, sampled
parameters and anomaly/break location when documenting the mapping. Renaming or
restructuring metadata must not conceal a missing class or a changed generator.

No claim is made that the paper's reported experiments have been rerun on this version.
