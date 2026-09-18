# 2-Way Combination Rules — Fractional Base

### Fractional base type

- `arfima`

Current ARFIMA generation uses:

\[
0.25 \le d < 0.49
\]

so the generated process is treated as a **stationary long-memory fractional process**.

---

## A. Fractional + Base-Like Components

| Combination | Status | Rule |
|---|---:|---|
| Fractional + Stationary | ❌ | ARFIMA already contains AR/MA-type stationary dynamics; do not duplicate as a separate standard 2-way class. |
| Fractional + Stochastic trend | ❌ | Ordinary integration and fractional integration would overlap as persistence mechanisms and make the class harder to interpret. |
| Fractional + Seasonal | ✅ | Long-memory dependence and seasonal structure may coexist. |
| Fractional + Volatility | ✅ | Long-memory dependence and conditional heteroskedasticity may coexist. |

---

## B. Fractional + Deterministic Trend

| Combination | Status |
|---|---:|
| Fractional + `linear_trend` | ✅ |
| Fractional + `quadratic_trend` | ✅ |
| Fractional + `cubic_trend` | ✅ |
| Fractional + `exponential_trend` | ✅ |
| Fractional + `damped_trend` | ✅ |

All deterministic trend types are logically allowed.

---

## C. Fractional + Anomaly

| Combination | Status | Rule |
|---|---:|---|
| Fractional + `point_anomaly` | ✅ | Valid. |
| Fractional + `collective_anomaly` | ✅ | Valid. |
| Fractional + `contextual_anomaly` | ❌ | Current contextual anomaly requires seasonal context. |

---

## D. Fractional + Structural Break

| Combination | Status | Rule |
|---|---:|---|
| Fractional + `mean_shift` | ✅ | Valid structural mean change. |
| Fractional + `variance_shift` | ✅ | Valid structural variance regime change. |
| Fractional + `trend_shift` | ❌ as 2-way | Requires `linear_trend`. |

For the current stationary ARFIMA setting, a mean or variance regime change is still logically meaningful while the underlying long-memory dependence remains present.

---

# Final Fractional 2-Way Rule

| Second component | Status |
|---|---:|
| Stationary | ❌ duplicate ARFIMA internal logic |
| Stochastic trend | ❌ overlapping integration mechanisms |
| Seasonal | ✅ |
| Volatility | ✅ |
| Linear trend | ✅ |
| Quadratic trend | ✅ |
| Cubic trend | ✅ |
| Exponential trend | ✅ |
| Damped trend | ✅ |
| Point anomaly | ✅ |
| Collective anomaly | ✅ |
| Contextual anomaly | ❌ |
| Mean shift | ✅ |
| Variance shift | ✅ |
| Trend shift | ❌ in 2-way |

### Fractional 2-Way Principle

ARFIMA is treated as a stationary long-memory base. It may be combined with seasonal structure, volatility, deterministic trends, supported anomalies, and structural mean/variance breaks, but not with a separate ordinary stochastic-integration component.