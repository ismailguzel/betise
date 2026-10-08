"""
JOINT STATISTICAL VALIDATION
SARMA / SARIMA + Volatility

Tests:
    sarma
    sarima

        x

    ARCH
    GARCH
    EGARCH
    APARCH

Goal
----
Validate that BOTH stochastic seasonality and volatility remain
statistically detectable in the SAME generated series.

SARMA:
    phi(B) Phi(B^s) Y_t = theta(B) Theta(B^s) epsilon_t

SARIMA:
    phi(B) Phi(B^s) (1-B)^d (1-B^s)^D Y_t
        = theta(B) Theta(B^s) epsilon_t

where epsilon_t comes from:
    ARCH / GARCH / EGARCH / APARCH

Validation:
1. Detect stochastic seasonal dependence in final Y_t.
2. Undo integration when needed.
3. Undo complete nonseasonal + seasonal ARMA dynamics.
4. Test recovered innovations for volatility.
5. Joint success = seasonality detected AND volatility detected.

Controls:
1. Matched nonseasonal volatility-driven process:
       checks false seasonal detections.

2. Seasonal model + Gaussian innovations:
       checks false volatility detections.
"""

from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.signal import lfilter
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

from betise.core.generator import TimeSeriesGenerator


# ============================================================
# SETTINGS
# ============================================================

MODELS = ["sarma", "sarima"]
VOLATILITY_TYPES = ["arch", "garch", "egarch", "aparch"]
PERIODS = [7, 12, 24, 52]

LENGTH = 400
N_TRIALS = 50
WARMUP = 50
ALPHA = 0.05
TEST_LAG = 10
SEED = 42

OUTPUT_DIR = Path(
    "examples/combination_tests/test_outputs/sarma_sarima__volatility"
)
PLOT_DIR = OUTPUT_DIR / "plots"

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# LABEL
# ============================================================

def case_label(model, period):
    prefix = "SARMA" if model == "sarma" else "SARIMA"
    return f"{prefix}_S{period}"


# ============================================================
# MODEL POLYNOMIALS
# ============================================================

def build_polynomials(info):
    """Reconstruct exact multiplicative SARMA AR/MA polynomials."""

    s = int(info["periods"][0])

    ar_coefs = np.asarray(info["ar_coefs"], dtype=float)
    ma_coefs = np.asarray(info["ma_coefs"], dtype=float)
    seasonal_ar_coefs = np.asarray(info["seasonal_ar_coefs"], dtype=float)
    seasonal_ma_coefs = np.asarray(info["seasonal_ma_coefs"], dtype=float)

    nonseasonal_ar = np.r_[1.0, -ar_coefs]
    nonseasonal_ma = np.r_[1.0, ma_coefs]

    seasonal_ar = np.zeros(len(seasonal_ar_coefs) * s + 1)
    seasonal_ma = np.zeros(len(seasonal_ma_coefs) * s + 1)

    seasonal_ar[0] = 1.0
    seasonal_ma[0] = 1.0

    for i, coef in enumerate(seasonal_ar_coefs, start=1):
        seasonal_ar[i * s] = -coef

    for i, coef in enumerate(seasonal_ma_coefs, start=1):
        seasonal_ma[i * s] = coef

    return {
        "period": s,
        "nonseasonal_ar": nonseasonal_ar,
        "nonseasonal_ma": nonseasonal_ma,
        "total_ar": np.convolve(nonseasonal_ar, seasonal_ar),
        "total_ma": np.convolve(nonseasonal_ma, seasonal_ma),
    }


# ============================================================
# SEASONALITY DETECTION
# ============================================================

def detect_seasonality(series, info):
    """
    Detect stochastic seasonal dependence at the known target period.

    First remove only the NONSEASONAL ARMA dynamics. The remaining
    process should retain the seasonal AR/MA structure and/or seasonal
    integration.

    Seasonal dependence is then tested through a lag-s regression.
    HAC covariance is used because innovations may be heteroskedastic.
    """

    poly = build_polynomials(info)
    s = poly["period"]

    seasonal_process = lfilter(
        poly["nonseasonal_ar"],
        poly["nonseasonal_ma"],
        np.asarray(series, dtype=float),
    )

    seasonal_process = seasonal_process[WARMUP:]

    y = seasonal_process[s:]
    lagged = seasonal_process[:-s]

    X = sm.add_constant(lagged)

    fit = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": TEST_LAG},
    )

    beta = float(fit.params[1])
    p_value = float(fit.pvalues[1])
    rho = float(np.corrcoef(y, lagged)[0, 1])

    return {
        "detected": p_value < ALPHA,
        "beta": beta,
        "p_value": p_value,
        "rho": rho,
    }


# ============================================================
# DIFFERENCING
# ============================================================

def difference_for_model(series, info):
    """Undo seasonal and nonseasonal integration before recovery."""

    x = np.asarray(series, dtype=float)
    s = int(info["periods"][0])

    d = int(info.get("diff", 0) or 0)
    D = int(info.get("seasonal_diff", 0) or 0)

    offset = 0

    for _ in range(D):
        x = x[s:] - x[:-s]
        offset += s

    for _ in range(d):
        x = np.diff(x)
        offset += 1

    return x, offset


# ============================================================
# INNOVATION RECOVERY
# ============================================================

def recover_innovations(series, info):
    """
    Undo integration and complete multiplicative SARMA filtering.

    The result should recover the original external volatility
    innovations.
    """

    stationary, offset = difference_for_model(series, info)
    poly = build_polynomials(info)

    recovered = lfilter(
        poly["total_ar"],
        poly["total_ma"],
        stationary,
    )

    warmup = max(WARMUP, 2 * poly["period"])
    recovered = recovered[warmup:]

    source_start = offset + warmup
    source_end = source_start + len(recovered)

    return recovered, source_start, source_end


# ============================================================
# VOLATILITY TESTS
# Same method as single_multiple_volatility.py
# ============================================================

def volatility_tests(residuals):
    """
    ARCH-LM and Ljung-Box on squared residuals.

    any_detected:
        either test detects volatility

    both_detected:
        both tests detect volatility
    """

    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    arch_p = float(het_arch(x, nlags=TEST_LAG)[1])

    lb = acorr_ljungbox(
        x ** 2,
        lags=[TEST_LAG],
        return_df=True,
    )
    lb_p = float(lb["lb_pvalue"].iloc[0])

    arch_detected = arch_p < ALPHA
    lb_detected = lb_p < ALPHA

    return {
        "arch_detected": arch_detected,
        "lb2_detected": lb_detected,
        "any_detected": arch_detected or lb_detected,
        "both_detected": arch_detected and lb_detected,
        "arch_p": arch_p,
        "lb2_p": lb_p,
    }


# ============================================================
# GENERATION
# ============================================================

def generate_combination(model, period, volatility_kind):
    ts = TimeSeriesGenerator(length=LENGTH)

    innovations, volatility_info = ts.generate_volatility(
        kind=volatility_kind,
        as_innovations=True,
    )
    innovations = np.asarray(innovations, dtype=float)

    if model == "sarma":
        combined_df, seasonal_info = ts.generate_sarma_seasonality(
            period=period,
            innovations=innovations,
        )
    else:
        combined_df, seasonal_info = ts.generate_sarima_seasonality(
            period=period,
            d=0,
            D=1,
            innovations=innovations,
        )

    return combined_df, seasonal_info, innovations, volatility_info


# ============================================================
# MATCHED NONSEASONAL CONTROL
# ============================================================

def build_nonseasonal_control(innovations, info):
    """
    Same external innovations and same nonseasonal ARMA dynamics,
    but without seasonal AR/MA or seasonal integration.
    """

    poly = build_polynomials(info)

    return lfilter(
        poly["nonseasonal_ma"],
        poly["nonseasonal_ar"],
        np.asarray(innovations, dtype=float),
    )


# ============================================================
# MAIN TRIAL
# ============================================================

def run_joint_trial(model, period, volatility_kind):
    """
    Same FINAL combined series is used for both characteristics.

    Seasonality:
        final series -> remove nonseasonal ARMA
        -> stochastic seasonal dependence test

    Volatility:
        final series -> undo integration
        -> undo complete SARMA structure
        -> ARCH-LM / squared Ljung-Box
    """

    combined_df, info, innovations, _ = generate_combination(
        model,
        period,
        volatility_kind,
    )

    final_series = combined_df["data"].to_numpy(dtype=float)

    seasonal = detect_seasonality(final_series, info)

    recovered, source_start, source_end = recover_innovations(
        final_series,
        info,
    )
    source = innovations[source_start:source_end]

    source_vol = volatility_tests(source)
    recovered_vol = volatility_tests(recovered)

    n = min(len(source), len(recovered))
    innovation_corr = float(
        np.corrcoef(source[-n:], recovered[-n:])[0, 1]
    )

    seasonal_success = seasonal["detected"]

    return {
        "seasonal_success": seasonal_success,
        "seasonal_beta": seasonal["beta"],
        "seasonal_rho": seasonal["rho"],
        "seasonal_p": seasonal["p_value"],

        "source_vol_arch": source_vol["arch_detected"],
        "source_vol_lb2": source_vol["lb2_detected"],
        "source_vol_any": source_vol["any_detected"],
        "source_vol_both": source_vol["both_detected"],

        "recovered_vol_arch": recovered_vol["arch_detected"],
        "recovered_vol_lb2": recovered_vol["lb2_detected"],
        "recovered_vol_any": recovered_vol["any_detected"],
        "recovered_vol_both": recovered_vol["both_detected"],

        "innovation_correlation": innovation_corr,

        "joint_any": (
            seasonal_success
            and recovered_vol["any_detected"]
        ),
        "joint_strict": (
            seasonal_success
            and recovered_vol["both_detected"]
        ),
    }


# ============================================================
# NEGATIVE CONTROL 1
# VOLATILITY WITHOUT SEASONALITY
# ============================================================

def run_volatility_only_control(model, period, volatility_kind):
    """
    Same volatility innovations and nonseasonal ARMA dynamics,
    but no stochastic seasonal mechanism.

    Seasonal detection here is a false positive.
    """

    _, info, innovations, _ = generate_combination(
        model,
        period,
        volatility_kind,
    )

    control = build_nonseasonal_control(
        innovations,
        info,
    )

    result = detect_seasonality(
        control,
        info,
    )

    return result["detected"]


# ============================================================
# NEGATIVE CONTROL 2
# SEASONAL MODEL + GAUSSIAN INNOVATIONS
# ============================================================

def run_seasonal_gaussian_control(model, period):
    """
    SARMA/SARIMA seasonal structure is present,
    but conditional heteroskedasticity is absent.

    Recovered volatility detection here is a false positive.
    """

    ts = TimeSeriesGenerator(length=LENGTH)
    innovations = np.random.normal(0, 1, LENGTH)

    if model == "sarma":
        df, info = ts.generate_sarma_seasonality(
            period=period,
            innovations=innovations,
        )
    else:
        df, info = ts.generate_sarima_seasonality(
            period=period,
            d=0,
            D=1,
            innovations=innovations,
        )

    series = df["data"].to_numpy(dtype=float)

    recovered, _, _ = recover_innovations(
        series,
        info,
    )

    return volatility_tests(recovered)


# ============================================================
# RUN MAIN VALIDATION
# ============================================================

def run_main_validation():
    rows = []

    for model in MODELS:
        for period in PERIODS:
            label = case_label(model, period)

            for volatility_kind in VOLATILITY_TYPES:
                print(
                    f"Running {label} + "
                    f"{volatility_kind.upper()} ..."
                )

                for trial in range(N_TRIALS):
                    result = run_joint_trial(
                        model,
                        period,
                        volatility_kind,
                    )

                    rows.append({
                        "model": model,
                        "period": period,
                        "case": label,
                        "volatility": volatility_kind,
                        "trial": trial + 1,
                        **result,
                    })

    return pd.DataFrame(rows)


# ============================================================
# RUN CONTROLS
# ============================================================

def run_seasonal_false_positive_control():
    rows = []

    for model in MODELS:
        for period in PERIODS:
            label = case_label(model, period)

            for volatility_kind in VOLATILITY_TYPES:
                print(
                    f"Seasonal FP control: {label} on "
                    f"{volatility_kind.upper()} ..."
                )

                for trial in range(N_TRIALS):
                    detected = run_volatility_only_control(
                        model,
                        period,
                        volatility_kind,
                    )

                    rows.append({
                        "model": model,
                        "period": period,
                        "case": label,
                        "volatility": volatility_kind,
                        "trial": trial + 1,
                        "seasonal_false_positive": detected,
                    })

    return pd.DataFrame(rows)


def run_volatility_false_positive_control():
    rows = []

    for model in MODELS:
        for period in PERIODS:
            label = case_label(model, period)

            print(
                f"Volatility FP control: "
                f"{label} + Gaussian ..."
            )

            for trial in range(N_TRIALS):
                result = run_seasonal_gaussian_control(
                    model,
                    period,
                )

                rows.append({
                    "model": model,
                    "period": period,
                    "case": label,
                    "trial": trial + 1,
                    "volatility_FP_ARCH": result["arch_detected"],
                    "volatility_FP_LB2": result["lb2_detected"],
                    "volatility_FP_any": result["any_detected"],
                    "volatility_FP_both": result["both_detected"],
                })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================

def summarize_main(results):
    return (
        results
        .groupby(
            ["case", "model", "period", "volatility"],
            as_index=False,
        )
        .agg(
            seasonal_detection_rate=("seasonal_success", "mean"),

            source_vol_ARCH=("source_vol_arch", "mean"),
            source_vol_LB2=("source_vol_lb2", "mean"),
            source_vol_any=("source_vol_any", "mean"),
            source_vol_both=("source_vol_both", "mean"),

            recovered_vol_ARCH=("recovered_vol_arch", "mean"),
            recovered_vol_LB2=("recovered_vol_lb2", "mean"),
            recovered_vol_any=("recovered_vol_any", "mean"),
            recovered_vol_both=("recovered_vol_both", "mean"),

            joint_success_any=("joint_any", "mean"),
            joint_success_strict=("joint_strict", "mean"),

            mean_seasonal_beta=("seasonal_beta", "mean"),
            mean_seasonal_rho=("seasonal_rho", "mean"),
            mean_innovation_correlation=(
                "innovation_correlation",
                "mean",
            ),
        )
    )


def summarize_seasonal_fp(results):
    return (
        results
        .groupby(
            ["case", "model", "period", "volatility"],
            as_index=False,
        )
        .agg(
            seasonal_false_positive_rate=(
                "seasonal_false_positive",
                "mean",
            )
        )
    )


def summarize_volatility_fp(results):
    return (
        results
        .groupby(
            ["case", "model", "period"],
            as_index=False,
        )
        .agg(
            ARCH_false_positive=("volatility_FP_ARCH", "mean"),
            LB2_false_positive=("volatility_FP_LB2", "mean"),
            ANY_false_positive=("volatility_FP_any", "mean"),
            BOTH_false_positive=("volatility_FP_both", "mean"),
        )
    )


# ============================================================
# REPRESENTATIVE PLOTS
# ============================================================

def save_representative_plots():
    """
    Save one representative 3-panel figure for every seasonal + volatility combination.

    Panel 1:
        source volatility innovations

    Panel 2:
        effect of stochastic seasonal filtering relative to
        the matched nonseasonal process

    Panel 3:
        final combined series

    Note:
        SARMA/SARIMA seasonality is not an additive Fourier
        component, so Panel 2 is a diagnostic seasonal EFFECT,
        not an independent additive component.
    """

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        for period in PERIODS:
            label = case_label(model, period)

            for volatility_kind in VOLATILITY_TYPES:
                df, info, innovations, _ = generate_combination(
                    model,
                    period,
                    volatility_kind,
                )

                final_series = df["data"].to_numpy(dtype=float)
                nonseasonal_control = build_nonseasonal_control(
                    innovations,
                    info,
                )

                seasonal_effect = (
                    final_series - nonseasonal_control
                )

                time = np.arange(LENGTH)

                fig, axes = plt.subplots(
                    3,
                    1,
                    figsize=(12, 9),
                    sharex=True,
                )

                axes[0].plot(
                    time,
                    innovations,
                    linewidth=1.1,
                )
                axes[0].set_title(
                    f"{label} + {volatility_kind.upper()} "
                    "— Volatility Innovations"
                )
                axes[0].set_ylabel("Value")
                axes[0].grid(alpha=0.3)

                axes[1].plot(
                    time,
                    seasonal_effect,
                    linewidth=1.1,
                )
                axes[1].set_title(
                    f"{label} + {volatility_kind.upper()} "
                    "— Stochastic Seasonal Effect"
                )
                axes[1].set_ylabel("Value")
                axes[1].grid(alpha=0.3)

                axes[2].plot(
                    time,
                    final_series,
                    linewidth=1.1,
                )
                axes[2].set_title(
                    f"{label} + {volatility_kind.upper()} "
                    "— Final Combined Series"
                )
                axes[2].set_xlabel("Time")
                axes[2].set_ylabel("Value")
                axes[2].grid(alpha=0.3)

                plt.tight_layout()

                filename = (
                    f"{label}_{volatility_kind}_components.png"
                )

                plt.savefig(
                    PLOT_DIR / filename,
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                print(f"Saved plot: {filename}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n========================================")
    print("SARMA / SARIMA + VOLATILITY")
    print("JOINT STATISTICAL VALIDATION")
    print("========================================\n")

    main_trials = run_main_validation()
    seasonal_fp_trials = run_seasonal_false_positive_control()
    volatility_fp_trials = run_volatility_false_positive_control()

    main_summary = summarize_main(main_trials)
    seasonal_fp_summary = summarize_seasonal_fp(
        seasonal_fp_trials
    )
    volatility_fp_summary = summarize_volatility_fp(
        volatility_fp_trials
    )

    main_trials.to_csv(
        OUTPUT_DIR / "trial_results.csv",
        index=False,
    )
    main_summary.to_csv(
        OUTPUT_DIR / "joint_detection_rates.csv",
        index=False,
    )
    seasonal_fp_summary.to_csv(
        OUTPUT_DIR / "seasonal_false_positive_control.csv",
        index=False,
    )
    volatility_fp_summary.to_csv(
        OUTPUT_DIR / "volatility_false_positive_control.csv",
        index=False,
    )

    save_representative_plots()

    print("\n========================================")
    print("JOINT DETECTION RATES")
    print("========================================")
    print(
        main_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n========================================")
    print("SEASONAL FALSE-POSITIVE CONTROL")
    print("========================================")
    print(
        seasonal_fp_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n========================================")
    print("VOLATILITY FALSE-POSITIVE CONTROL")
    print("========================================")
    print(
        volatility_fp_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )