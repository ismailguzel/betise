# 2-Way Combination Rules — Seasonal Base

### Seasonal base types

- `single_seasonality`
- `multiple_seasonality`
- `sarma`
- `sarima`

---

## A. Seasonal + Base-Like Components

| Combination | Status | Rule |
|---|---:|---|
| Seasonal + Stationary | ❌ | Already represented by seasonal generators such as `SARMA`; do not create a duplicate standard 2-way class. |
| Seasonal + Stochastic trend | ✅ | Distinct mechanism: ordinary stochastic integration + seasonal behavior. Current `SARIMA` does not represent ordinary stochastic trend. |
| Seasonal + Volatility | ✅ | Seasonal structure and conditional heteroskedasticity may coexist. |
| Seasonal + Fractional | ✅ | Seasonal structure and long-memory fractional dependence may coexist. |

### Important SARIMA note

Current `sarima` is generated as:

\[
Y_t - Y_{t-s} = F_t + \varepsilon_t
\]

So it contains a **seasonal unit root**, but not ordinary stochastic integration.

Therefore:

`seasonal + stochastic_trend`

remains a genuine 2-way combination.

---

## B. Seasonal + Deterministic Trend

| Combination | Status |
|---|---:|
| Seasonal + `linear_trend` | ✅ |
| Seasonal + `quadratic_trend` | ✅ |
| Seasonal + `cubic_trend` | ✅ |
| Seasonal + `exponential_trend` | ✅ |
| Seasonal + `damped_trend` | ✅ |

All deterministic trend types are allowed.

---

## C. Seasonal + Anomaly

| Combination | Status | Rule |
|---|---:|---|
| Seasonal + `point_anomaly` | ✅ | Valid for all seasonal bases. |
| Seasonal + `collective_anomaly` | ✅ | Valid; anomaly generation must remain independent of seasonal context. |
| Seasonal + `contextual_anomaly` | ✅ | Valid only because a seasonal context exists. |

### Collective vs Contextual Rule

A collective anomaly is generated independently of the expected seasonal pattern.

A contextual anomaly explicitly depends on the seasonal pattern and violates the expected behavior at a seasonal context.

Therefore both may exist on seasonal bases, but their generation mechanisms must remain distinct.

---

## D. Seasonal + Structural Break

| Combination | Status | Rule |
|---|---:|---|
| Seasonal + `mean_shift` | ✅ | Valid level-regime change while preserving seasonality. |
| Seasonal + `variance_shift` | ✅ / 🔧 for SARIMA | Valid; SARIMA requires technical review. |
| Seasonal + `trend_shift` | ❌ as 2-way | Requires `linear_trend`. |

### SARIMA variance-shift note

For current SARIMA:

\[
Y_t-Y_{t-s}=F_t+\varepsilon_t
\]

a clean variance shift would ideally modify the variance of \(\varepsilon_t\).

Therefore:

`SARIMA + variance_shift`

is logically valid but technically flagged for implementation review.

---

# Final Seasonal 2-Way Rule

| Second component | Status |
|---|---:|
| Stationary | ❌ duplicate existing seasonal-base logic |
| Stochastic trend | ✅ |
| Volatility | ✅ |
| Fractional | ✅ |
| Linear trend | ✅ |
| Quadratic trend | ✅ |
| Cubic trend | ✅ |
| Exponential trend | ✅ |
| Damped trend | ✅ |
| Point anomaly | ✅ |
| Collective anomaly | ✅ |
| Contextual anomaly | ✅ |
| Mean shift | ✅ |
| Variance shift | ✅ / 🔧 for SARIMA |
| Trend shift | ❌ in 2-way |

### Seasonal 2-Way Principle

Seasonality may coexist with stochastic trend, volatility, fractional dependence, deterministic trends, anomalies, and structural breaks.

`contextual_anomaly` is uniquely valid here because the seasonal base provides the required contextual structure.