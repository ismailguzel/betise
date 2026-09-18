"""
JOINT STATISTICAL VALIDATION
Seasonality + Stochastic

Valid combinations
------------------
Single:
    RW / RWD / ARI / IMA
    ARIMA + single excluded: redundant with deterministic SARIMA.

Multiple:
    RW / RWD / ARI / IMA / ARIMA

Model
-----
    Y_t = S_t + F_t

where:
    S_t = stochastic background
    F_t = deterministic Fourier seasonality

Validation
----------
1. Detect target period(s) in the final combined series.
2. Estimate/remove Fourier seasonality.
3. Check that the stochastic integration order is preserved.
4. Difference to recover the pre-integration process.
5. Undo AR/MA/ARMA dynamics.
6. Check that recovered innovations are white noise.

Joint success:
    seasonality detected
    AND integration structure preserved
    AND recovered innovations are white noise.
"""

from itertools import combinations
from pathlib import Path
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.signal import lfilter
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from betise.core.generator import TimeSeriesGenerator


# ============================================================
# SETTINGS
# ============================================================

STOCHASTIC_TYPES = ["rw", "rwd", "ari", "ima", "arima"]

LENGTH = 400
N_TRIALS = 50
ALPHA = 0.05
TEST_LAG = 10
WARMUP = 50
NUM_HARMONICS = 1
SEED = 42

PERIOD_POOL = [7, 12, 24, 52]

SINGLE_CASES = [("single", (p,)) for p in PERIOD_POOL]
MULTIPLE_CASES = [("multiple", pair) for pair in combinations(PERIOD_POOL, 2)]
SEASONAL_CASES = SINGLE_CASES + MULTIPLE_CASES

OUTPUT_DIR = Path("betise/combination_tests/test_outputs/seasonality_stochastic")
PLOT_DIR = OUTPUT_DIR / "plots"

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def case_label(kind, periods):
    if kind == "single":
        return f"S{periods[0]}"
    return "M" + "+".join(str(p) for p in periods)


def is_valid_combination(kind, stochastic_kind):
    # ARIMA + single is already represented by deterministic SARIMA.
    return not (kind == "single" and stochastic_kind == "arima")


def stochastic_difference_order(stochastic_kind, info):
    if stochastic_kind in {"rw", "rwd"}:
        return 1
    return int(info["diff"])


# ============================================================
# FOURIER DESIGN
# ============================================================

def build_fourier_design(length, periods, start_index=0):
    t = np.arange(start_index, start_index + length)
    columns = [np.ones(length)]

    for period in periods:
        for k in range(1, NUM_HARMONICS + 1):
            columns.append(np.sin(2 * np.pi * k * t / period))
            columns.append(np.cos(2 * np.pi * k * t / period))

    return np.column_stack(columns)


# ============================================================
# SEASONALITY DETECTION
# ============================================================

def detect_seasonality(series, periods, start_index=0):
    """
    Detect seasonality using Fourier regression.

    This will normally be applied AFTER differencing because
    stochastic backgrounds contain unit roots.

    Differencing changes Fourier amplitude/phase but not its
    frequency, so the same sine/cosine basis remains valid.
    """

    y = np.asarray(series, dtype=float)
    X = build_fourier_design(len(y), periods, start_index)

    fit = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": TEST_LAG}
    )

    terms_per_period = 2 * NUM_HARMONICS
    pvalues = {}

    for i, period in enumerate(periods):
        first = 1 + i * terms_per_period
        R = np.zeros((terms_per_period, X.shape[1]))

        for j in range(terms_per_period):
            R[j, first + j] = 1.0

        pvalues[period] = float(
            fit.wald_test(R, scalar=True).pvalue
        )

    seasonal_fit = X[:, 1:] @ fit.params[1:]
    residuals = np.asarray(fit.resid, dtype=float)

    residual_std = np.std(residuals)
    seasonal_ratio = (
        np.std(seasonal_fit) / residual_std
        if residual_std > 1e-12 else np.inf
    )

    return {
        "detected": all(p < ALPHA for p in pvalues.values()),
        "period_pvalues": pvalues,
        "seasonal_fit": seasonal_fit,
        "residuals": residuals,
        "seasonal_std_ratio": seasonal_ratio,
    }


# ============================================================
# LEVEL FOURIER REMOVAL
# ============================================================

def remove_level_seasonality(series, periods):
    """
    Estimate Fourier in level space so we can inspect the
    recovered stochastic level process.

    Significance testing is NOT done here. Seasonal inference
    is performed on the differenced series above.
    """

    y = np.asarray(series, dtype=float)
    X = build_fourier_design(len(y), periods)

    fit = sm.OLS(y, X).fit()
    seasonal_fit = X[:, 1:] @ fit.params[1:]

    return y - seasonal_fit


# ============================================================
# STATIONARITY
# ============================================================

def stationarity_test(series):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    adf_p = float(adfuller(x, regression="c", autolag="AIC")[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_p = float(kpss(x, regression="c", nlags="auto")[1])

    stationary = (adf_p < ALPHA) and (kpss_p > ALPHA)

    return {
        "stationary": stationary,
        "adf_p": adf_p,
        "kpss_p": kpss_p,
    }


def integration_order_test(series, d):
    """
    Check an I(d) pattern.

    d=1:
        level nonstationary
        first difference stationary

    d=2:
        level nonstationary
        first difference nonstationary
        second difference stationary
    """

    x = np.asarray(series, dtype=float)

    tests = [stationarity_test(x)]

    current = x.copy()
    for _ in range(d):
        current = np.diff(current)
        tests.append(stationarity_test(current))

    if d == 1:
        valid = (not tests[0]["stationary"]) and tests[1]["stationary"]

    elif d == 2:
        valid = (
            (not tests[0]["stationary"])
            and (not tests[1]["stationary"])
            and tests[2]["stationary"]
        )

    else:
        raise ValueError("This validation currently supports d=1 or d=2.")

    return {
        "valid": valid,
        "tests": tests,
    }


# ============================================================
# PRE-INTEGRATION PROCESS
# ============================================================

def recover_preintegration_process(series, stochastic_kind, info):
    """
    Undo integration.

    RW:
        ΔX_t

    RWD:
        ΔX_t - drift

    ARI / IMA / ARIMA:
        Δ^d X_t
    """

    x = np.asarray(series, dtype=float)

    if stochastic_kind == "rw":
        return np.diff(x)

    if stochastic_kind == "rwd":
        return np.diff(x) - float(info["drift"])

    d = int(info["diff"])
    return np.diff(x, n=d)


# ============================================================
# INNOVATION RECOVERY
# ============================================================

def recover_innovations(stationary_process, stochastic_kind, info):
    y = np.asarray(stationary_process, dtype=float)

    if stochastic_kind in {"rw", "rwd"}:
        return y

    if stochastic_kind == "ari":
        ar = np.r_[1.0, -np.asarray(info["ar_coefs"], dtype=float)]
        ma = np.array([1.0])

    elif stochastic_kind == "ima":
        ar = np.array([1.0])
        ma = np.r_[1.0, np.asarray(info["ma_coefs"], dtype=float)]

    elif stochastic_kind == "arima":
        ar = np.r_[1.0, -np.asarray(info["ar_coefs"], dtype=float)]
        ma = np.r_[1.0, np.asarray(info["ma_coefs"], dtype=float)]

    else:
        raise ValueError(f"Unknown stochastic type: {stochastic_kind}")

    return lfilter(ar, ma, y)


# ============================================================
# WHITENESS
# ============================================================

def whiteness_test(innovations):
    x = np.asarray(innovations, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    result = acorr_ljungbox(x, lags=[TEST_LAG], return_df=True)
    p = float(result["lb_pvalue"].iloc[0])

    return {
        "white": p > ALPHA,
        "lb_p": p,
    }


# ============================================================
# GENERATION
# ============================================================

def generate_stochastic_background(stochastic_kind):
    ts = TimeSeriesGenerator(length=LENGTH)

    if stochastic_kind == "rw":
        df, info = ts.generate_stochastic_trend(kind="rw")
        d = 1

    elif stochastic_kind == "rwd":
        df, info = ts.generate_stochastic_trend(kind="rwd", drift=0.05)
        d = 1

    else:
        d = int(np.random.choice([1, 2]))

        df, info = ts.generate_stochastic_trend(
            kind=stochastic_kind,
            d=d,
            const=False
        )

    return ts, df, info, d


def generate_combination(kind, periods, stochastic_kind):
    ts, background_df, background_info, d = (
        generate_stochastic_background(stochastic_kind)
    )

    kwargs = {
        "background_df": background_df,
        "kind": kind,
        "difference_order": d,
        "num_harmonics": NUM_HARMONICS,
    }

    if kind == "single":
        kwargs["period"] = periods[0]
    else:
        kwargs["periods"] = list(periods)

    combined_df, seasonal_info = ts.compose_with_fourier_seasonality(**kwargs)

    return (
        background_df,
        background_info,
        combined_df,
        seasonal_info,
        d,
    )


# ============================================================
# JOINT TRIAL
# ============================================================

def run_joint_trial(kind, periods, stochastic_kind):
    background_df, info, combined_df, _, d = generate_combination(
        kind, periods, stochastic_kind
    )

    source = background_df["data"].to_numpy(dtype=float)
    combined = combined_df["data"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # 1. SEASONALITY DETECTION
    # --------------------------------------------------------

    differenced_combined = np.diff(combined, n=d)

    seasonal = detect_seasonality(
        differenced_combined,
        periods,
        start_index=d
    )

    # residual here is the recovered stationary
    # pre-integration stochastic process.
    recovered_stationary_process = seasonal["residuals"]

    # --------------------------------------------------------
    # 2. INTEGRATION STRUCTURE
    # --------------------------------------------------------

    recovered_level = remove_level_seasonality(
        combined,
        periods
    )

    source_integration = integration_order_test(source, d)
    recovered_integration = integration_order_test(recovered_level, d)

    # --------------------------------------------------------
    # 3. SOURCE PRE-INTEGRATION PROCESS
    # --------------------------------------------------------

    source_stationary_process = recover_preintegration_process(
        source,
        stochastic_kind,
        info
    )

    # RWD seasonal regression already removes an intercept.
    # For consistency source RWD removes known drift above.

    # --------------------------------------------------------
    # 4. INNOVATION RECOVERY
    # --------------------------------------------------------

    source_innovations = recover_innovations(
        source_stationary_process,
        stochastic_kind,
        info
    )[WARMUP:]

    recovered_innovations = recover_innovations(
        recovered_stationary_process,
        stochastic_kind,
        info
    )[WARMUP:]

    source_white = whiteness_test(source_innovations)
    recovered_white = whiteness_test(recovered_innovations)

    # --------------------------------------------------------
    # 5. RECOVERY CORRELATIONS
    # --------------------------------------------------------

    n = min(
        len(source_stationary_process),
        len(recovered_stationary_process)
    )

    stationary_correlation = float(
        np.corrcoef(
            source_stationary_process[-n:],
            recovered_stationary_process[-n:]
        )[0, 1]
    )

    level_correlation = float(
        np.corrcoef(
            source,
            recovered_level
        )[0, 1]
    )

    seasonal_success = seasonal["detected"]
    integration_success = recovered_integration["valid"]
    structure_success = recovered_white["white"]

    joint_success = (
        seasonal_success
        and integration_success
        and structure_success
    )

    return {
        "d": d,

        "seasonal_detected": seasonal_success,
        "seasonal_std_ratio": seasonal["seasonal_std_ratio"],

        "source_integration_valid": source_integration["valid"],
        "recovered_integration_valid": integration_success,

        "source_white": source_white["white"],
        "recovered_white": structure_success,

        "stationary_process_correlation": stationary_correlation,
        "level_background_correlation": level_correlation,

        "joint_success": joint_success,
    }


# ============================================================
# CONTROL 1 — STOCHASTIC ONLY
# ============================================================

def run_seasonal_false_positive_control(periods, stochastic_kind):
    _, background_df, info, d = generate_stochastic_background(
        stochastic_kind
    )

    source = background_df["data"].to_numpy(dtype=float)
    stationary_process = np.diff(source, n=d)

    result = detect_seasonality(
        stationary_process,
        periods,
        start_index=d
    )

    return result["detected"]


# ============================================================
# CONTROL 2 — STATIONARY + SEASONALITY
# ============================================================

def run_stochastic_false_positive_control(kind, periods):
    """
    Generate AR + Fourier.

    After seasonal removal it should NOT look I(1).

    If it does, our stochastic/integration classifier is
    producing a false positive.
    """

    ts = TimeSeriesGenerator(length=LENGTH)

    background_df, _ = ts.generate_stationary_base_series(
        distribution="ar"
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

    combined = combined_df["data"].to_numpy(dtype=float)
    recovered = remove_level_seasonality(combined, periods)

    return integration_order_test(recovered, d=1)["valid"]


# ============================================================
# MAIN VALIDATION
# ============================================================

def run_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for stochastic_kind in STOCHASTIC_TYPES:
            if not is_valid_combination(kind, stochastic_kind):
                continue

            print(f"Running {label} + {stochastic_kind.upper()} ...")

            for trial in range(N_TRIALS):
                result = run_joint_trial(kind, periods, stochastic_kind)

                rows.append({
                    "case": label,
                    "seasonality": kind,
                    "periods": "+".join(map(str, periods)),
                    "stochastic": stochastic_kind,
                    "trial": trial + 1,
                    **result,
                })

    return pd.DataFrame(rows)


# ============================================================
# CONTROLS
# ============================================================

def run_seasonal_fp_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for stochastic_kind in STOCHASTIC_TYPES:
            if not is_valid_combination(kind, stochastic_kind):
                continue

            print(f"Seasonal FP: {label} on {stochastic_kind.upper()} ...")

            for trial in range(N_TRIALS):
                detected = run_seasonal_false_positive_control(
                    periods,
                    stochastic_kind
                )

                rows.append({
                    "case": label,
                    "stochastic": stochastic_kind,
                    "trial": trial + 1,
                    "seasonal_false_positive": detected,
                })

    return pd.DataFrame(rows)


def run_stochastic_fp_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        print(f"Stochastic FP control: {label} + AR ...")

        for trial in range(N_TRIALS):
            detected = run_stochastic_false_positive_control(
                kind,
                periods
            )

            rows.append({
                "case": label,
                "trial": trial + 1,
                "stochastic_false_positive": detected,
            })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================

def summarize_validation(results):
    return (
        results
        .groupby(
            ["case", "seasonality", "periods", "stochastic"],
            as_index=False
        )
        .agg(
            seasonal_detection_rate=("seasonal_detected", "mean"),

            source_integration_rate=("source_integration_valid", "mean"),
            recovered_integration_rate=("recovered_integration_valid", "mean"),

            source_whiteness_rate=("source_white", "mean"),
            recovered_whiteness_rate=("recovered_white", "mean"),

            joint_success_rate=("joint_success", "mean"),

            mean_stationary_process_correlation=(
                "stationary_process_correlation",
                "mean"
            ),

            mean_level_background_correlation=(
                "level_background_correlation",
                "mean"
            ),

            mean_seasonal_std_ratio=("seasonal_std_ratio", "mean"),

            d1_trials=("d", lambda x: int((x == 1).sum())),
            d2_trials=("d", lambda x: int((x == 2).sum())),
        )
    )


def summarize_seasonal_fp(results):
    return (
        results
        .groupby(["case", "stochastic"], as_index=False)
        .agg(
            seasonal_false_positive_rate=(
                "seasonal_false_positive",
                "mean"
            )
        )
    )


def summarize_stochastic_fp(results):
    return (
        results
        .groupby("case", as_index=False)
        .agg(
            stochastic_false_positive_rate=(
                "stochastic_false_positive",
                "mean"
            )
        )
    )


# ============================================================
# REPRESENTATIVE PLOTS
# ============================================================

def save_representative_plots():
    """
    Component 1: stochastic background
    Component 2: Fourier seasonality
    Combination: final series
    """

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for stochastic_kind in STOCHASTIC_TYPES:
            if not is_valid_combination(kind, stochastic_kind):
                continue

            background_df, _, combined_df, _, d = generate_combination(
                kind,
                periods,
                stochastic_kind
            )

            time = combined_df["time"].to_numpy()
            background = background_df["data"].to_numpy(dtype=float)
            combined = combined_df["data"].to_numpy(dtype=float)
            fourier = combined - background

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

            axes[0].plot(time, background, linewidth=1.1)
            axes[0].set_title(
                f"{label} + {stochastic_kind.upper()} "
                f"— Component 1: Stochastic (d={d})"
            )
            axes[0].set_ylabel("Value")
            axes[0].grid(alpha=0.3)

            axes[1].plot(time, fourier, linewidth=1.1)
            axes[1].set_title(
                f"{label} + {stochastic_kind.upper()} "
                "— Component 2: Fourier"
            )
            axes[1].set_ylabel("Value")
            axes[1].grid(alpha=0.3)

            axes[2].plot(time, combined, linewidth=1.1)
            axes[2].set_title(
                f"{label} + {stochastic_kind.upper()} — Combination"
            )
            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Value")
            axes[2].grid(alpha=0.3)

            plt.tight_layout()

            filename = (
                f"{label}_{stochastic_kind}_components.png"
                .replace("+", "_")
            )

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
    print("SEASONALITY + STOCHASTIC")
    print("JOINT STATISTICAL VALIDATION")
    print("========================================\n")

    trial_results = run_validation()
    seasonal_fp_results = run_seasonal_fp_validation()
    stochastic_fp_results = run_stochastic_fp_validation()

    summary = summarize_validation(trial_results)
    seasonal_fp_summary = summarize_seasonal_fp(seasonal_fp_results)
    stochastic_fp_summary = summarize_stochastic_fp(stochastic_fp_results)

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

    stochastic_fp_summary.to_csv(
        OUTPUT_DIR / "stochastic_false_positive_control.csv",
        index=False
    )

    print("\n========================================")
    print("JOINT DETECTION RATES")
    print("========================================")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\n========================================")
    print("SEASONAL FALSE-POSITIVE CONTROL")
    print("========================================")
    print(
        seasonal_fp_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\n========================================")
    print("STOCHASTIC FALSE-POSITIVE CONTROL")
    print("========================================")
    print(
        stochastic_fp_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    save_representative_plots()

    print(f"\nResults saved to:\n{OUTPUT_DIR.resolve()}")