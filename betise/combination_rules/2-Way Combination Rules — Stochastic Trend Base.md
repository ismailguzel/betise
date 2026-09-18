# 2-Way Combination Rules — Stochastic Trend Base

### Stochastic base types

- `random_walk`
- `random_walk_drift`
- `ari`
- `ima`
- `arima`

---

## A. Stochastic + Base-Like Components

| Combination | Status | Rule |
|---|---:|---|
| Stochastic + Stationary | ❌ | `ARI`, `IMA`, and `ARIMA` already contain AR/MA-type stationary dynamics. Would duplicate existing stochastic generators. |
| Stochastic + Seasonal | ✅ | Genuine additional mechanism. The current seasonal generators do not contain ordinary stochastic trend. |
| Stochastic + Volatility | ✅ | Genuine combination: integrated/evolving level + conditional heteroskedasticity. |
| Stochastic + Fractional | ❌ | Both introduce integration/persistence mechanisms. Combining integer and fractional integration would be redundant and unnecessarily difficult to interpret for the standard dataset. |

### Important seasonal note

`stochastic + seasonal` is **not already represented by the current SARIMA generator**.

Current SARIMA:

\[
(1-B^s)Y_t = F_t+\varepsilon_t
\]

contains a **seasonal unit root**, but no ordinary stochastic integration.

Therefore combinations such as:

- `ARIMA + single_seasonality`
- `ARIMA + multiple_seasonality`
- stochastic trend + SARMA-type seasonality
- ordinary stochastic integration + seasonal unit root

are legitimate **2-way concepts**.

---

## B. Stochastic + Deterministic Trend

| Combination | Status |
|---|---:|
| Stochastic + `linear_trend` | ✅ with drift restriction |
| Stochastic + `quadratic_trend` | ✅ |
| Stochastic + `cubic_trend` | ✅ |
| Stochastic + `exponential_trend` | ✅ |
| Stochastic + `damped_trend` | ✅ |

### Linear Trend Rule

An explicit linear trend must not be added when deterministic drift is already active.

| Case | Status |
|---|---:|
| `random_walk + linear_trend` | ✅ |
| `random_walk_drift + linear_trend` | ❌ |
| `ARI/IMA/ARIMA`, `const=False` + linear trend | ✅ |
| `ARI/IMA/ARIMA`, `const=True` + linear trend | ❌ |

Other nonlinear deterministic trends remain allowed even when drift exists.

---

## C. Stochastic + Anomaly

| Combination | Status | Rule |
|---|---:|---|
| Stochastic + `point_anomaly` | ✅ | Valid. |
| Stochastic + `collective_anomaly` | ✅ | Valid. |
| Stochastic + `contextual_anomaly` | ❌ | No seasonal context exists in the stochastic base alone. |

---

## D. Stochastic + Structural Break

| Combination | Status | Rule |
|---|---:|---|
| Stochastic + `mean_shift` | ✅ | Valid structural level change. |
| Stochastic + `variance_shift` | ✅ / 🔧 | Logically valid; technical implementation must be reviewed. |
| Stochastic + `trend_shift` | ❌ as 2-way | Requires `linear_trend`; therefore minimum 3-way. |

For variance shift, the intended structural change should ideally affect the **innovation variance** rather than simply rescaling an integrated observed path.

---

# Final Stochastic 2-Way Rule

| Second component | Status |
|---|---:|
| Stationary | ❌ duplicate existing base logic |
| Seasonal | ✅ |
| Volatility | ✅ |
| Fractional | ❌ overlapping integration/persistence |
| Linear trend | ✅ with drift restriction |
| Quadratic trend | ✅ |
| Cubic trend | ✅ |
| Exponential trend | ✅ |
| Damped trend | ✅ |
| Point anomaly | ✅ |
| Collective anomaly | ✅ |
| Contextual anomaly | ❌ |
| Mean shift | ✅ |
| Variance shift | ✅ / 🔧 |
| Trend shift | ❌ in 2-way |