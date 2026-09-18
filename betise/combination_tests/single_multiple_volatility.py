"""
JOINT STATISTICAL VALIDATION
Seasonality + Volatility

Tests:
    single_seasonality
    multiple_seasonality

        x

    ARCH
    GARCH
    EGARCH
    APARCH

Goal
----
Validate that BOTH characteristics remain statistically detectable
in the SAME generated series.

Final model:
    Y_t = F_t + V_t

Validation:
1. Detect deterministic seasonality in final Y_t.
2. Estimate and remove seasonality from Y_t.
3. Test recovered residuals for volatility.
4. Joint success = seasonality detected AND volatility detected.

Controls:
1. Volatility only:
       Checks false seasonal detections.

2. Seasonality + Gaussian noise:
       Checks false volatility detections.
"""

from pathlib import Path
from itertools import combinations
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

from betise.core.generator import TimeSeriesGenerator


# ============================================================
# SETTINGS
# ============================================================

LENGTH = 400
N_TRIALS = 50
WARMUP = 50
ALPHA = 0.05
TEST_LAG = 10
NUM_HARMONICS = 1
SEED = 42

VOLATILITY_TYPES = ["arch", "garch", "egarch", "aparch"]

# All are valid for length=400 with min_cycles=6.
PERIOD_POOL = [7, 12, 24, 52]

SINGLE_CASES = [("single", (p,)) for p in PERIOD_POOL]
MULTIPLE_CASES = [("multiple", pair) for pair in combinations(PERIOD_POOL, 2)]
SEASONAL_CASES = SINGLE_CASES + MULTIPLE_CASES

OUTPUT_DIR = Path("betise/combination_tests/test_outputs/seasonality_volatility")
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
# FOURIER REGRESSION
# ============================================================

def build_fourier_design(length, periods, start_index=0):
    """
    Design matrix:
        intercept
        sin/cos for each target period

    We use the known target periods from metadata, but NOT the
    true generated Fourier coefficients.
    """

    t = np.arange(start_index, start_index + length)
    columns = [np.ones(length)]

    for period in periods:
        for k in range(1, NUM_HARMONICS + 1):
            columns.append(np.sin(2 * np.pi * k * t / period))
            columns.append(np.cos(2 * np.pi * k * t / period))

    return np.column_stack(columns)


def fit_and_test_seasonality(series, periods, start_index=0):
    """
    Fit Fourier regression to the FINAL combined series.

    HC3 robust covariance is used because the background is
    intentionally heteroskedastic.

    Returns:
        - p-value for each target period
        - whether every target period is detected
        - joint seasonal Wald test
        - residuals after estimated seasonality removal
        - seasonal/residual standard-deviation ratio
    """

    y = np.asarray(series, dtype=float)
    X = build_fourier_design(len(y), periods, start_index=start_index)

    fit = sm.OLS(y, X).fit(cov_type="HC3")

    period_pvalues = {}
    n_terms_per_period = 2 * NUM_HARMONICS

    for i, period in enumerate(periods):
        first = 1 + i * n_terms_per_period

        R = np.zeros((n_terms_per_period, X.shape[1]))
        for j in range(n_terms_per_period):
            R[j, first + j] = 1.0

        period_pvalues[period] = float(fit.wald_test(R, scalar=True).pvalue)

    # Test all Fourier coefficients together.
    R_all = np.zeros((X.shape[1] - 1, X.shape[1]))
    R_all[:, 1:] = np.eye(X.shape[1] - 1)
    joint_p = float(fit.wald_test(R_all, scalar=True).pvalue)

    all_targets_detected = all(p < ALPHA for p in period_pvalues.values())

    # Seasonal fitted component WITHOUT the intercept.
    seasonal_fit = X[:, 1:] @ fit.params[1:]
    residuals = np.asarray(fit.resid, dtype=float)

    resid_std = np.std(residuals)
    seasonal_ratio = np.std(seasonal_fit) / resid_std if resid_std > 1e-12 else np.inf

    return {
        "period_pvalues": period_pvalues,
        "all_targets_detected": all_targets_detected,
        "joint_seasonal_detected": joint_p < ALPHA,
        "joint_seasonal_p": joint_p,
        "residuals": residuals,
        "seasonal_std_ratio": seasonal_ratio,
    }


# ============================================================
# VOLATILITY TESTS
# ============================================================

def volatility_tests(residuals):
    """
    ARCH-LM and Ljung-Box on squared residuals.

    any_detected:
        either test detects volatility

    both_detected:
        stricter criterion; both tests detect volatility
    """

    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    arch_p = float(het_arch(x, nlags=TEST_LAG)[1])

    lb = acorr_ljungbox(x ** 2, lags=[TEST_LAG], return_df=True)
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

def compose_seasonal_volatility(kind, periods, volatility_kind):
    ts = TimeSeriesGenerator(length=LENGTH)

    volatility_df, volatility_info = ts.generate_volatility(kind=volatility_kind)

    kwargs = {
        "background_df": volatility_df,
        "kind": kind,
        "difference_order": 0,
        "num_harmonics": NUM_HARMONICS,
    }

    if kind == "single":
        kwargs["period"] = periods[0]
    else:
        kwargs["periods"] = list(periods)

    combined_df, seasonal_info = ts.compose_with_fourier_seasonality(**kwargs)

    return volatility_df, combined_df, volatility_info, seasonal_info


# ============================================================
# MAIN TRIAL
# ============================================================

def run_joint_trial(kind, periods, volatility_kind):
    """
    Same final series is used for BOTH branches.

    Seasonality:
        final series -> Fourier regression -> target detection

    Volatility:
        final series -> estimated seasonal removal -> residuals
        -> ARCH-LM / squared Ljung-Box
    """

    volatility_df, combined_df, _, _ = compose_seasonal_volatility(
        kind, periods, volatility_kind
    )

    source_volatility = volatility_df["data"].to_numpy(dtype=float)[WARMUP:]
    final_series = combined_df["data"].to_numpy(dtype=float)[WARMUP:]

    seasonal = fit_and_test_seasonality(
        final_series,
        periods,
        start_index=WARMUP
    )

    source_vol_test = volatility_tests(source_volatility)
    recovered_vol_test = volatility_tests(seasonal["residuals"])

    seasonal_success = seasonal["all_targets_detected"]

    return {
        "seasonal_success": seasonal_success,
        "joint_seasonal_success": seasonal["joint_seasonal_detected"],

        "source_vol_arch": source_vol_test["arch_detected"],
        "source_vol_lb2": source_vol_test["lb2_detected"],
        "source_vol_any": source_vol_test["any_detected"],
        "source_vol_both": source_vol_test["both_detected"],

        "recovered_vol_arch": recovered_vol_test["arch_detected"],
        "recovered_vol_lb2": recovered_vol_test["lb2_detected"],
        "recovered_vol_any": recovered_vol_test["any_detected"],
        "recovered_vol_both": recovered_vol_test["both_detected"],

        # Main combined metrics
        "joint_any": seasonal_success and recovered_vol_test["any_detected"],
        "joint_strict": seasonal_success and recovered_vol_test["both_detected"],

        "seasonal_std_ratio": seasonal["seasonal_std_ratio"],
    }


# ============================================================
# NEGATIVE CONTROL 1
# VOLATILITY WITHOUT SEASONALITY
# ============================================================

def run_volatility_only_control(periods, volatility_kind):
    """
    There is NO deterministic seasonality.

    We deliberately test the target periods anyway.

    Detection here = false-positive seasonal detection.
    """

    ts = TimeSeriesGenerator(length=LENGTH)
    volatility_df, _ = ts.generate_volatility(kind=volatility_kind)

    series = volatility_df["data"].to_numpy(dtype=float)[WARMUP:]

    result = fit_and_test_seasonality(
        series,
        periods,
        start_index=WARMUP
    )

    return result["all_targets_detected"]


# ============================================================
# NEGATIVE CONTROL 2
# SEASONALITY + GAUSSIAN NOISE
# ============================================================

def run_seasonal_gaussian_control(kind, periods):
    """
    Deterministic seasonality is present, but volatility is not.

    After Fourier regression and seasonal removal, the residual
    should behave like constant-variance Gaussian noise.

    Volatility detection here = false positive.
    """

    ts = TimeSeriesGenerator(length=LENGTH)

    gaussian = np.random.normal(0, 1, LENGTH)
    gaussian_df = pd.DataFrame({
        "time": np.arange(LENGTH),
        "data": gaussian
    })

    kwargs = {
        "background_df": gaussian_df,
        "kind": kind,
        "difference_order": 0,
        "num_harmonics": NUM_HARMONICS,
    }

    if kind == "single":
        kwargs["period"] = periods[0]
    else:
        kwargs["periods"] = list(periods)

    combined_df, _ = ts.compose_with_fourier_seasonality(**kwargs)
    series = combined_df["data"].to_numpy(dtype=float)[WARMUP:]

    seasonal = fit_and_test_seasonality(
        series,
        periods,
        start_index=WARMUP
    )

    return volatility_tests(seasonal["residuals"])


# ============================================================
# RUN MAIN VALIDATION
# ============================================================

def run_main_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for volatility_kind in VOLATILITY_TYPES:
            print(f"Running {label} + {volatility_kind.upper()} ...")

            for trial in range(N_TRIALS):
                result = run_joint_trial(kind, periods, volatility_kind)

                rows.append({
                    "seasonality": kind,
                    "periods": "+".join(map(str, periods)),
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

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for volatility_kind in VOLATILITY_TYPES:
            print(f"Seasonal FP control: {label} on {volatility_kind.upper()} ...")

            for trial in range(N_TRIALS):
                detected = run_volatility_only_control(periods, volatility_kind)

                rows.append({
                    "case": label,
                    "volatility": volatility_kind,
                    "trial": trial + 1,
                    "seasonal_false_positive": detected,
                })

    return pd.DataFrame(rows)


def run_volatility_false_positive_control():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)
        print(f"Volatility FP control: {label} + Gaussian ...")

        for trial in range(N_TRIALS):
            result = run_seasonal_gaussian_control(kind, periods)

            rows.append({
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
        .groupby(["case", "seasonality", "periods", "volatility"], as_index=False)
        .agg(
            seasonal_detection_rate=("seasonal_success", "mean"),
            joint_seasonal_detection=("joint_seasonal_success", "mean"),

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

            mean_seasonal_std_ratio=("seasonal_std_ratio", "mean"),
        )
    )


def summarize_seasonal_fp(results):
    return (
        results
        .groupby(["case", "volatility"], as_index=False)
        .agg(seasonal_false_positive_rate=("seasonal_false_positive", "mean"))
    )


def summarize_volatility_fp(results):
    return (
        results
        .groupby("case", as_index=False)
        .agg(
            ARCH_false_positive=("volatility_FP_ARCH", "mean"),
            LB2_false_positive=("volatility_FP_LB2", "mean"),
            ANY_false_positive=("volatility_FP_any", "mean"),
            BOTH_false_positive=("volatility_FP_both", "mean"),
        )
    )


def save_representative_plots():
    """
    Save one representative 3-panel figure for every
    seasonality + volatility combination.

    Panel 1: Volatility component
    Panel 2: Fourier seasonality component
    Panel 3: Combined series
    """

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for volatility_kind in VOLATILITY_TYPES:
            volatility_df, combined_df, _, _ = compose_seasonal_volatility(
                kind, periods, volatility_kind
            )

            time = combined_df["time"].to_numpy()
            volatility = volatility_df["data"].to_numpy(dtype=float)
            combined = combined_df["data"].to_numpy(dtype=float)

            # Since:
            # combined = volatility + Fourier
            fourier = combined - volatility

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

            # ------------------------------------------------
            # COMPONENT 1 — VOLATILITY
            # ------------------------------------------------
            axes[0].plot(time, volatility, linewidth=1.1)
            axes[0].set_title(
                f"{label} + {volatility_kind.upper()} — Volatility Component"
            )
            axes[0].set_ylabel("Value")
            axes[0].grid(alpha=0.3)

            # ------------------------------------------------
            # COMPONENT 2 — FOURIER SEASONALITY
            # ------------------------------------------------
            axes[1].plot(time, fourier, linewidth=1.1)
            axes[1].set_title(
                f"{label} + {volatility_kind.upper()} — Fourier Component"
            )
            axes[1].set_ylabel("Value")
            axes[1].grid(alpha=0.3)

            # ------------------------------------------------
            # COMBINED SERIES
            # ------------------------------------------------
            axes[2].plot(time, combined, linewidth=1.1)
            axes[2].set_title(
                f"{label} + {volatility_kind.upper()} — Combined Series"
            )
            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Value")
            axes[2].grid(alpha=0.3)

            plt.tight_layout()

            filename = f"{label}_{volatility_kind}_components.png".replace("+", "_")

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
    print("SEASONALITY + VOLATILITY")
    print("JOINT STATISTICAL VALIDATION")
    print("========================================\n")

    main_trials = run_main_validation()
    seasonal_fp_trials = run_seasonal_false_positive_control()
    volatility_fp_trials = run_volatility_false_positive_control()

    main_summary = summarize_main(main_trials)
    seasonal_fp_summary = summarize_seasonal_fp(seasonal_fp_trials)
    volatility_fp_summary = summarize_volatility_fp(volatility_fp_trials)

    main_trials.to_csv(OUTPUT_DIR / "trial_results.csv", index=False)
    main_summary.to_csv(OUTPUT_DIR / "joint_detection_rates.csv", index=False)
    seasonal_fp_summary.to_csv(OUTPUT_DIR / "seasonal_false_positive_control.csv", index=False)
    volatility_fp_summary.to_csv(OUTPUT_DIR / "volatility_false_positive_control.csv", index=False)

    save_representative_plots()

    print("\n========================================")
    print("JOINT DETECTION RATES")
    print("========================================")
    print(main_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n========================================")
    print("SEASONAL FALSE-POSITIVE CONTROL")
    print("========================================")
    print(seasonal_fp_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n========================================")
    print("VOLATILITY FALSE-POSITIVE CONTROL")
    print("========================================")
    print(volatility_fp_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nResults saved to:\n{OUTPUT_DIR.resolve()}")