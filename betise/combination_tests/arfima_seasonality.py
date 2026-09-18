"""
JOINT STATISTICAL VALIDATION
ARFIMA + Deterministic Seasonality

Validated candidate combinations
--------------------------------
ARFIMA + single_seasonality
ARFIMA + multiple_seasonality

Model
-----
    Y_t = X_t + F_t

where:
    X_t = stationary long-memory ARFIMA(p, d, q), 0 < d < 0.5
    F_t = deterministic Fourier seasonality

Validation
----------
For the SAME final combined series:

1. Detect the target Fourier seasonality.
2. Estimate and remove Fourier seasonality.
3. Remove the KNOWN short-memory ARMA structure from:
       - source ARFIMA
       - recovered ARFIMA
4. Estimate fractional differencing d on the remaining fractional component.
5. Check whether long memory and d are preserved.

Main composition success:
    seasonality detected
    AND recovered d is close to source d
    AND recovered background is highly correlated with source background

A stricter joint-detection metric is also reported separately.

Controls
--------
1. ARFIMA-only:
       No seasonality exists.
       Measures seasonal false-positive rate.

2. ARMA + seasonality:
       Seasonality exists but there is no fractional long memory.
       Measures long-memory false-positive rate.

Plots
-----
For every case:

    Component 1: ARFIMA background
    Component 2: Fourier seasonality
    Combination
"""

from itertools import combinations
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import lfilter

from betise.core.generator import TimeSeriesGenerator


# ============================================================
# SETTINGS
# ============================================================

LENGTH = 400
N_TRIALS = 50
ALPHA = 0.05
TEST_LAG = 10
NUM_HARMONICS = 1
SEED = 42

D_RANGE = (0.25, 0.49)

# GPH low-frequency bandwidth:
# m = floor(n ** GPH_BANDWIDTH_EXPONENT)
GPH_BANDWIDTH_EXPONENT = 0.60

# Conservative practical long-memory threshold.
LONG_MEMORY_D_THRESHOLD = 0.20

# After removing the known short-memory ARMA part, a threshold of 0.20
# is deliberately conservative for n≈400 and keeps the ARMA-control
# false-positive rate much lower than the previous raw-GPH threshold.

# Preservation is judged against the SOURCE estimate because
# finite-sample d estimation can itself be biased/noisy.
D_PRESERVATION_TOLERANCE = 0.10
BACKGROUND_CORRELATION_THRESHOLD = 0.95

PERIOD_POOL = [7, 12, 24, 52]

SINGLE_CASES = [("single", (p,)) for p in PERIOD_POOL]
MULTIPLE_CASES = [("multiple", pair) for pair in combinations(PERIOD_POOL, 2)]
SEASONAL_CASES = SINGLE_CASES + MULTIPLE_CASES

OUTPUT_DIR = Path("betise/combination_tests/test_outputs/arfima_seasonality")
PLOT_DIR = OUTPUT_DIR / "plots"

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# LABEL
# ============================================================

def case_label(kind, periods):
    if kind == "single":
        return f"S{periods[0]}"
    return "M" + "+".join(str(p) for p in periods)


# ============================================================
# FOURIER DESIGN
# ============================================================

def build_fourier_design(length, periods, num_harmonics=1):
    t = np.arange(length)
    columns = [np.ones(length)]

    for period in periods:
        for k in range(1, num_harmonics + 1):
            columns.append(np.sin(2 * np.pi * k * t / period))
            columns.append(np.cos(2 * np.pi * k * t / period))

    return np.column_stack(columns)


# ============================================================
# SEASONALITY DETECTION / REMOVAL
# ============================================================

def fit_seasonality(series, periods):
    """
    Estimate deterministic Fourier seasonality from the FINAL series.

    HAC covariance is used because ARFIMA has serial dependence.
    True Fourier coefficients are not used.
    """

    y = np.asarray(series, dtype=float)
    X = build_fourier_design(len(y), periods, NUM_HARMONICS)

    fit = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": TEST_LAG}
    )

    terms_per_period = 2 * NUM_HARMONICS
    period_pvalues = {}

    for i, period in enumerate(periods):
        first = 1 + i * terms_per_period
        R = np.zeros((terms_per_period, X.shape[1]))

        for j in range(terms_per_period):
            R[j, first + j] = 1.0

        period_pvalues[period] = float(
            fit.wald_test(R, scalar=True).pvalue
        )

    seasonal_detected = all(
        pvalue < ALPHA
        for pvalue in period_pvalues.values()
    )

    seasonal_fit = X[:, 1:] @ fit.params[1:]
    residuals = np.asarray(fit.resid, dtype=float)

    residual_std = np.std(residuals)
    seasonal_std_ratio = (
        np.std(seasonal_fit) / residual_std
        if residual_std > 1e-12
        else np.inf
    )

    return {
        "detected": seasonal_detected,
        "period_pvalues": period_pvalues,
        "seasonal_fit": seasonal_fit,
        "residuals": residuals,
        "seasonal_std_ratio": seasonal_std_ratio,
    }


# ============================================================
# GPH FRACTIONAL-d ESTIMATOR
# ============================================================

def estimate_d_gph(series, bandwidth_exponent=GPH_BANDWIDTH_EXPONENT):
    """
    Geweke-Porter-Hudak log-periodogram estimate of fractional d.

    For low Fourier frequencies:

        log I(lambda_j)
            = c - d * log(4 sin^2(lambda_j / 2)) + error

    Therefore:
        d_hat = -slope

    The estimator is used comparatively:
        source ARFIMA d_hat
        vs
        recovered ARFIMA d_hat

    This avoids requiring an unrealistically perfect finite-sample
    estimate of the true generated d in every trial.
    """

    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    n = len(x)
    if n < 50:
        raise ValueError("GPH estimation requires at least 50 observations.")

    m = int(np.floor(n ** bandwidth_exponent))
    m = max(8, min(m, n // 4))

    fft_values = np.fft.fft(x)
    j = np.arange(1, m + 1)
    lambdas = 2 * np.pi * j / n

    periodogram = (np.abs(fft_values[j]) ** 2) / (2 * np.pi * n)
    periodogram = np.maximum(periodogram, np.finfo(float).tiny)

    regressor = np.log(4 * np.sin(lambdas / 2) ** 2)
    response = np.log(periodogram)

    X = sm.add_constant(regressor)
    fit = sm.OLS(response, X).fit()

    d_hat = float(-fit.params[1])

    return {
        "d_hat": d_hat,
        "slope_se": float(fit.bse[1]),
        "r_squared": float(fit.rsquared),
        "bandwidth": m,
    }


def prewhiten_short_memory(series, ar_coefs, ma_coefs):
    """
    Remove the KNOWN short-memory ARMA structure before estimating d.

    ARFIMA simulator convention:
        Phi(B) X_t = Theta(B) W_t

    Therefore:
        W_t = Theta(B)^(-1) Phi(B) X_t

    Using the generator's ground-truth AR/MA coefficients here is intentional:
    this is a GENERATOR VALIDATION test, not a blind downstream estimator.

    After this step:
        - ARFIMA should leave fractionally integrated noise
        - ARMA control should leave approximately white noise

    This prevents ordinary short-memory AR/MA dynamics from being mistaken
    for fractional long memory by the low-frequency GPH regression.
    """

    x = np.asarray(series, dtype=float)

    ar_coefs = np.asarray(
        [] if ar_coefs is None else ar_coefs,
        dtype=float
    )

    ma_coefs = np.asarray(
        [] if ma_coefs is None else ma_coefs,
        dtype=float
    )

    ar_poly = np.r_[1.0, -ar_coefs]
    ma_poly = np.r_[1.0, ma_coefs]

    filtered = lfilter(
        ar_poly,
        ma_poly,
        x
    )

    # Remove a small filter start-up transient.
    transient = max(
        20,
        5 * max(len(ar_coefs), len(ma_coefs), 1)
    )

    if transient >= len(filtered) - 50:
        transient = max(0, len(filtered) // 10)

    return filtered[transient:]


def assess_long_memory(
    source_series,
    recovered_series,
    true_d,
    ar_coefs,
    ma_coefs
):
    """
    Validate preservation of the FRACTIONAL component.

    The same known ARMA inverse filter is applied to source and recovered
    backgrounds before estimating d. Therefore any difference in d_hat is
    caused by the seasonality composition/removal step, not by ARMA nuisance
    dynamics.
    """

    source_fractional = prewhiten_short_memory(
        source_series,
        ar_coefs,
        ma_coefs
    )

    recovered_fractional = prewhiten_short_memory(
        recovered_series,
        ar_coefs,
        ma_coefs
    )

    source_est = estimate_d_gph(source_fractional)
    recovered_est = estimate_d_gph(recovered_fractional)

    source_d = source_est["d_hat"]
    recovered_d = recovered_est["d_hat"]

    source_long_memory = source_d >= LONG_MEMORY_D_THRESHOLD
    recovered_long_memory = recovered_d >= LONG_MEMORY_D_THRESHOLD

    d_error_source = abs(source_d - true_d)
    d_error_recovered = abs(recovered_d - true_d)
    d_preserved = abs(recovered_d - source_d) <= D_PRESERVATION_TOLERANCE

    # Fair composition metric:
    # if long memory was detectable in the source, did it remain detectable
    # after Fourier composition + estimated Fourier removal?
    conditional_long_memory_preserved = (
        recovered_long_memory
        if source_long_memory
        else np.nan
    )

    return {
        "source_d_hat": source_d,
        "recovered_d_hat": recovered_d,
        "source_long_memory": source_long_memory,
        "recovered_long_memory": recovered_long_memory,
        "conditional_long_memory_preserved": conditional_long_memory_preserved,
        "d_preserved": d_preserved,
        "source_d_abs_error": d_error_source,
        "recovered_d_abs_error": d_error_recovered,
        "d_source_recovered_abs_diff": abs(recovered_d - source_d),
    }


# ============================================================
# GENERATION
# ============================================================

def generate_combination(kind, periods):
    ts = TimeSeriesGenerator(length=LENGTH)

    background_df, background_info = ts.generate_fractional_process(
        kind="arfima",
        d_range=D_RANGE
    )

    kwargs = {
        "background_df": background_df,
        "kind": kind,
        "difference_order": 0,
        "num_harmonics": NUM_HARMONICS,
    }

    if kind == "single":
        kwargs["period"] = periods[0]
    else:
        kwargs["periods"] = list(periods)

    combined_df, seasonal_info = ts.compose_with_fourier_seasonality(**kwargs)

    return background_df, background_info, combined_df, seasonal_info


# ============================================================
# JOINT TRIAL
# ============================================================

def run_joint_trial(kind, periods):
    background_df, background_info, combined_df, _ = generate_combination(
        kind,
        periods
    )

    source_background = background_df["data"].to_numpy(dtype=float)
    final_series = combined_df["data"].to_numpy(dtype=float)
    true_d = float(background_info["d"])

    # 1. Detect and estimate seasonality in the FINAL series.
    seasonal = fit_seasonality(final_series, periods)

    # 2. Recovered background after estimated Fourier removal.
    recovered_background = seasonal["residuals"]

    # 3. Long-memory preservation.
    long_memory = assess_long_memory(
        source_background,
        recovered_background,
        true_d,
        ar_coefs=background_info.get("ar_coefs"),
        ma_coefs=background_info.get("ma_coefs"),
    )

    # 4. Direct recovery diagnostic.
    source_centered = source_background - np.mean(source_background)
    recovered_centered = recovered_background - np.mean(recovered_background)

    background_correlation = float(
        np.corrcoef(source_centered, recovered_centered)[0, 1]
    )

    background_recovered = (
        background_correlation >= BACKGROUND_CORRELATION_THRESHOLD
    )

    long_memory_success = (
        long_memory["recovered_long_memory"]
        and long_memory["d_preserved"]
    )

    # Main generator-composition criterion.
    # This does not punish the composition when a finite-sample detector
    # already fails on the SOURCE ARFIMA realization.
    composition_success = (
        seasonal["detected"]
        and long_memory["d_preserved"]
        and background_recovered
    )

    # Stricter observational criterion: both source and recovered background
    # must be detected as long-memory in addition to preservation.
    joint_detection_success = (
        seasonal["detected"]
        and long_memory["source_long_memory"]
        and long_memory["recovered_long_memory"]
        and long_memory["d_preserved"]
    )

    return {
        "true_d": true_d,

        "seasonal_detected": seasonal["detected"],
        "seasonal_std_ratio": seasonal["seasonal_std_ratio"],

        **long_memory,

        "long_memory_success": long_memory_success,
        "background_correlation": background_correlation,
        "background_recovered": background_recovered,
        "composition_success": composition_success,
        "joint_detection_success": joint_detection_success,
    }


# ============================================================
# CONTROL 1
# ARFIMA WITHOUT SEASONALITY
# ============================================================

def run_seasonal_false_positive_control(periods):
    """
    No Fourier seasonality exists.

    Detection of all requested target periods is therefore
    a seasonal false positive.
    """

    ts = TimeSeriesGenerator(length=LENGTH)

    df, _ = ts.generate_fractional_process(
        kind="arfima",
        d_range=D_RANGE
    )

    result = fit_seasonality(
        df["data"].to_numpy(dtype=float),
        periods
    )

    return result["detected"]


# ============================================================
# CONTROL 2
# ARMA + SEASONALITY
# ============================================================

def run_long_memory_false_positive_control(kind, periods):
    """
    Generate short-memory ARMA + real Fourier seasonality.

    After Fourier removal, the recovered background should
    NOT normally be classified as fractional long memory.

    This empirically checks the chosen d threshold.
    """

    ts = TimeSeriesGenerator(length=LENGTH)

    background_df, background_info = ts.generate_stationary_base_series(
        distribution="arma"
    )

    kwargs = {
        "background_df": background_df,
        "kind": kind,
        "difference_order": 0,
        "num_harmonics": NUM_HARMONICS,
    }

    if kind == "single":
        kwargs["period"] = periods[0]
    else:
        kwargs["periods"] = list(periods)

    combined_df, _ = ts.compose_with_fourier_seasonality(**kwargs)

    seasonal = fit_seasonality(
        combined_df["data"].to_numpy(dtype=float),
        periods
    )

    recovered_background = seasonal["residuals"]

    recovered_innovations = prewhiten_short_memory(
        recovered_background,
        ar_coefs=background_info.get("ar_coefs"),
        ma_coefs=background_info.get("ma_coefs"),
    )

    d_hat = estimate_d_gph(recovered_innovations)["d_hat"]

    return {
        "d_hat": d_hat,
        "false_long_memory": d_hat >= LONG_MEMORY_D_THRESHOLD,
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def run_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)
        print(f"Running ARFIMA + {label} ...")

        for trial in range(N_TRIALS):
            result = run_joint_trial(kind, periods)

            rows.append({
                "case": label,
                "seasonality": kind,
                "periods": "+".join(map(str, periods)),
                "trial": trial + 1,
                **result,
            })

    return pd.DataFrame(rows)


def run_seasonal_fp_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)
        print(f"Seasonal FP: {label} on ARFIMA-only ...")

        for trial in range(N_TRIALS):
            detected = run_seasonal_false_positive_control(periods)

            rows.append({
                "case": label,
                "trial": trial + 1,
                "seasonal_false_positive": detected,
            })

    return pd.DataFrame(rows)


def run_long_memory_fp_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)
        print(f"Long-memory FP: {label} + ARMA ...")

        for trial in range(N_TRIALS):
            result = run_long_memory_false_positive_control(kind, periods)

            rows.append({
                "case": label,
                "trial": trial + 1,
                **result,
            })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================

def summarize_validation(results):
    return (
        results
        .groupby(["case", "seasonality", "periods"], as_index=False)
        .agg(
            seasonal_detection_rate=("seasonal_detected", "mean"),

            source_long_memory_rate=("source_long_memory", "mean"),
            recovered_long_memory_rate=("recovered_long_memory", "mean"),
            conditional_long_memory_preservation_rate=(
                "conditional_long_memory_preserved",
                "mean"
            ),
            d_preservation_rate=("d_preserved", "mean"),
            long_memory_success_rate=("long_memory_success", "mean"),
            background_recovery_rate=("background_recovered", "mean"),

            composition_success_rate=("composition_success", "mean"),
            joint_detection_success_rate=("joint_detection_success", "mean"),

            mean_true_d=("true_d", "mean"),
            mean_source_d_hat=("source_d_hat", "mean"),
            mean_recovered_d_hat=("recovered_d_hat", "mean"),

            mean_source_d_abs_error=("source_d_abs_error", "mean"),
            mean_recovered_d_abs_error=("recovered_d_abs_error", "mean"),
            mean_source_recovered_d_diff=("d_source_recovered_abs_diff", "mean"),

            mean_background_correlation=("background_correlation", "mean"),
            mean_seasonal_std_ratio=("seasonal_std_ratio", "mean"),
        )
    )


def summarize_seasonal_fp(results):
    return (
        results
        .groupby("case", as_index=False)
        .agg(
            seasonal_false_positive_rate=(
                "seasonal_false_positive",
                "mean"
            )
        )
    )


def summarize_long_memory_fp(results):
    return (
        results
        .groupby("case", as_index=False)
        .agg(
            long_memory_false_positive_rate=(
                "false_long_memory",
                "mean"
            ),
            mean_control_d_hat=("d_hat", "mean"),
        )
    )


# ============================================================
# REPRESENTATIVE PLOTS
# ============================================================

def save_representative_plots():
    """
    Three stacked panels:

        Component 1: ARFIMA background
        Component 2: Fourier seasonality
        Combination: final series
    """

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        background_df, background_info, combined_df, _ = generate_combination(
            kind,
            periods
        )

        time = combined_df["time"].to_numpy()
        background = background_df["data"].to_numpy(dtype=float)
        combined = combined_df["data"].to_numpy(dtype=float)
        fourier = combined - background

        true_d = float(background_info["d"])

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(time, background, linewidth=1.1)
        axes[0].set_title(
            f"ARFIMA + {label} — Component 1: ARFIMA (d={true_d:.3f})"
        )
        axes[0].set_ylabel("Value")
        axes[0].grid(alpha=0.3)

        axes[1].plot(time, fourier, linewidth=1.1)
        axes[1].set_title(
            f"ARFIMA + {label} — Component 2: Fourier"
        )
        axes[1].set_ylabel("Value")
        axes[1].grid(alpha=0.3)

        axes[2].plot(time, combined, linewidth=1.1)
        axes[2].set_title(
            f"ARFIMA + {label} — Combination"
        )
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Value")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()

        filename = f"arfima_{label}_components.png".replace("+", "_")

        plt.savefig(
            PLOT_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved plot: {filename}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("ARFIMA + SEASONALITY")
    print("JOINT STATISTICAL VALIDATION — V2")
    print("Prewhitened GPH long-memory validation")
    print(f"Long-memory d threshold: {LONG_MEMORY_D_THRESHOLD}")
    print(f"d preservation tolerance: {D_PRESERVATION_TOLERANCE}")
    print("========================================\n")

    trial_results = run_validation()
    seasonal_fp_results = run_seasonal_fp_validation()
    long_memory_fp_results = run_long_memory_fp_validation()

    summary = summarize_validation(trial_results)
    seasonal_fp_summary = summarize_seasonal_fp(seasonal_fp_results)
    long_memory_fp_summary = summarize_long_memory_fp(long_memory_fp_results)

    trial_results.to_csv(
        OUTPUT_DIR / "trial_results.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "joint_detection_rates.csv",
        index=False
    )

    seasonal_fp_summary.to_csv(
        OUTPUT_DIR / "seasonal_false_positive_control.csv",
        index=False
    )

    long_memory_fp_summary.to_csv(
        OUTPUT_DIR / "long_memory_false_positive_control.csv",
        index=False
    )

    print("\n========================================")
    print("JOINT DETECTION RATES")
    print("========================================")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\n========================================")
    print("SEASONAL FALSE-POSITIVE CONTROL")
    print("========================================")
    print(
        seasonal_fp_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\n========================================")
    print("LONG-MEMORY FALSE-POSITIVE CONTROL")
    print("========================================")
    print(
        long_memory_fp_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    save_representative_plots()

    print(f"\nResults saved to:\n{OUTPUT_DIR.resolve()}")
