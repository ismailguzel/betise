# 2-Way Combination Rules — Stationary Base

### Stationary base types

- `ar`
- `ma`
- `arma`
- `white_noise`

A 2-way combination contains:

**1 stationary base + 1 independent additional component**

---

## A. Stationary + Base-Like Components

White noise is included as a stationary base, but it is treated as the minimal stationary case. Therefore, combinations with white noise are allowed only when the added component introduces a genuinely distinct mechanism that is not already represented by an existing base generator.

| Combination                    | Status | Rule                                                                                                                                                      |
| ------------------------------ | -----: | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| White noise + Stochastic trend |      ❌ | Stochastic-trend generators already represent the relevant integrated or evolving level behavior. Do not create a duplicate white-noise-plus-trend class. |
| White noise + Seasonal         |      ❌ | Seasonal generators already represent the relevant seasonal structure. Do not create a separate white-noise-plus-seasonality class.                       |
| White noise + Volatility       |      ❌ | A standalone ARCH/GARCH/EGARCH/APARCH process already represents white-noise innovations with conditional volatility.                                     |
| White noise + Fractional       |      ❌ | Fractional-integration generators already represent the relevant long-memory behavior without requiring a separate white-noise combination.               |

For the remaining stationary bases:

| Combination                   | Status | Rule                                                           |
| ----------------------------- | -----: | -------------------------------------------------------------- |
| AR/MA/ARMA + Stochastic trend |      ❌ | Already represented by ARIMA-family generators.                |
| AR/MA/ARMA + Seasonal         |      ❌ | Already represented by seasonal AR/MA/ARMA generators.         |
| AR/MA/ARMA + Volatility       |      ✅ | Adds conditional volatility to nontrivial stationary dynamics. |
| AR/MA/ARMA + Fractional       |      ❌ | Already represented by ARFIMA-family generators.               |

Thus, `white_noise` remains a valid stationary base, but it does not generate additional 2-way classes with stochastic trend, seasonality, volatility, or fractional integration. Its valid 2-way combinations are with deterministic trends, supported anomaly types, and supported structural shifts.

| Combination                   | Status | Rule                                                                                                                                            |
| ----------------------------- | -----: | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Stationary + Stochastic trend |      ❌ | `ARI`, `IMA`, `ARIMA` already represent stationary AR/MA dynamics combined with integration. Do not create a duplicate 2-way class.             |
| Stationary + Seasonal         |      ❌ | Seasonal generators, especially `SARMA`, already represent stationary dynamics with seasonality. Do not create a separate standard 2-way class. |
| Stationary + Volatility       |      ✅ | Represents a genuinely additional mechanism, e.g. AR/MA/ARMA dynamics with ARCH/GARCH-type conditional volatility.                              |
| Stationary + Fractional       |      ❌ | `ARFIMA` already combines AR/MA dynamics with fractional integration. Do not duplicate it as a separate 2-way class.                            |

### Stationary + Volatility subtype rule

| Stationary subtype | + ARCH/GARCH/EGARCH/APARCH |
| ------------------ | -------------------------: |
| `ar`               |                          ✅ |
| `ma`               |                          ✅ |
| `arma`             |                          ✅ |
| `white_noise`      |                          ❌ |

`white_noise + volatility` is excluded because a standalone ARCH/GARCH/EGARCH/APARCH process already represents zero-mean innovations with conditional volatility. Counting it separately would mainly duplicate the volatility base family.

---

## B. Stationary + Deterministic Trend

| Combination                      | Status |
| -------------------------------- | -----: |
| Stationary + `linear_trend`      |      ✅ |
| Stationary + `quadratic_trend`   |      ✅ |
| Stationary + `cubic_trend`       |      ✅ |
| Stationary + `exponential_trend` |      ✅ |
| Stationary + `damped_trend`      |      ✅ |

All five deterministic trend types are valid 2-way combinations with stationary bases.

---

## C. Stationary + Anomaly

| Combination                       | Status | Rule                                                              |
| --------------------------------- | -----: | ----------------------------------------------------------------- |
| Stationary + `point_anomaly`      |      ✅ | Valid for all stationary bases.                                   |
| Stationary + `collective_anomaly` |      ✅ | Valid for all stationary bases.                                   |
| Stationary + `contextual_anomaly` |      ❌ | Current contextual anomaly requires an existing seasonal context. |

Single and multiple point anomalies belong to the same conceptual `point_anomaly` feature.

Therefore:

`AR + multiple point anomalies`

is still a **2-way** combination.

---

## D. Stationary + Structural Break

| Combination                   |     Status | Rule                                   |
| ----------------------------- | ---------: | -------------------------------------- |
| Stationary + `mean_shift`     |          ✅ | Valid structural level change.         |
| Stationary + `variance_shift` |          ✅ | Valid structural variance change.      |
| Stationary + `trend_shift`    | ❌ as 2-way | `trend_shift` requires `linear_trend`. |

The minimum valid trend-shift combination is:

`stationary + linear_trend + trend_shift`

which is therefore **3-way**.

---

# Final Stationary 2-Way Rule

| Second component   |                              Status |
| ------------------ | ----------------------------------: |
| Stochastic trend   |     ❌ duplicate existing base logic |
| Seasonal           |     ❌ duplicate existing base logic |
| Volatility         | ✅ except `white_noise + volatility` |
| Fractional         |     ❌ duplicate existing base logic |
| Linear trend       |                                   ✅ |
| Quadratic trend    |                                   ✅ |
| Cubic trend        |                                   ✅ |
| Exponential trend  |                                   ✅ |
| Damped trend       |                                   ✅ |
| Point anomaly      |                                   ✅ |
| Collective anomaly |                                   ✅ |
| Contextual anomaly |                                   ❌ |
| Mean shift         |                                   ✅ |
| Variance shift     |                                   ✅ |
| Trend shift        |                          ❌ in 2-way |

### Stationary 2-Way Principle

A second component is included only when it introduces a **new distinguishable mechanism** that is not already represented by one of the existing generation-wise base types.
