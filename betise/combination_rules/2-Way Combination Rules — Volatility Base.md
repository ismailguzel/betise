# 2-Way Combination Rules — Volatility Base

### Volatility base types

- `arch`
- `garch`
- `egarch`
- `aparch`

---

## A. Volatility + Base-Like Components

| Combination | Status | Rule |
|---|---:|---|
| Volatility + Stationary | ✅ | Distinct mechanisms: conditional heteroskedasticity + stationary mean dynamics. |
| Volatility + Stochastic trend | ✅ | Integrated/evolving level with conditional heteroskedasticity is logically meaningful. |
| Volatility + Seasonal | ✅ | Seasonal structure and time-varying conditional variance may coexist. |
| Volatility + Fractional | ✅ | Long-memory dependence and conditional heteroskedasticity may coexist. |

No other volatility model is added simultaneously.

Therefore:

- `GARCH + EGARCH` ❌
- `ARCH + APARCH` ❌

Only one volatility model should govern the volatility component.

---

## B. Volatility + Deterministic Trend

| Combination | Status |
|---|---:|
| Volatility + `linear_trend` | ✅ |
| Volatility + `quadratic_trend` | ✅ |
| Volatility + `cubic_trend` | ✅ |
| Volatility + `exponential_trend` | ✅ |
| Volatility + `damped_trend` | ✅ |

All deterministic trend types are allowed.

The current volatility generators use zero-mean models, so there is no drift-versus-linear-trend redundancy.

---

## C. Volatility + Anomaly

| Combination | Status | Rule |
|---|---:|---|
| Volatility + `point_anomaly` | ✅ | Valid isolated anomaly on top of conditional volatility. |
| Volatility + `collective_anomaly` | ✅ | Valid abnormal interval distinct from ordinary volatility dynamics. |
| Volatility + `contextual_anomaly` | ❌ | Current contextual anomaly requires seasonal context. |

---

## D. Volatility + Structural Break

| Combination | Status | Rule |
|---|---:|---|
| Volatility + `mean_shift` | ✅ | Valid change in mean regime. |
| Volatility + `variance_shift` | ✅ / 🔧 | Logically valid, but technical implementation requires review. |
| Volatility + `trend_shift` | ❌ as 2-way | `trend_shift` requires `linear_trend`. |

### Variance-shift note

Conditional volatility and a structural variance shift are **not the same phenomenon**.

- ARCH/GARCH/etc. → variance changes dynamically over time.
- `variance_shift` → persistent change in the variance regime.

Therefore:

`GARCH + variance_shift`

is a valid 2-way combination.

However, during technical implementation we should decide whether the structural break should modify volatility-model parameters or simply rescale the generated process.

---

# Final Volatility 2-Way Rule

| Second component | Status |
|---|---:|
| Stationary | ✅ |
| Stochastic trend | ✅ |
| Seasonal | ✅ |
| Fractional | ✅ |
| Linear trend | ✅ |
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

### Volatility 2-Way Principle

Volatility may coexist with other dynamic structures because it primarily describes the **conditional variance behavior** of a process rather than its mean, trend, seasonal, or long-memory structure.

Only one volatility-model family should be active at a time.