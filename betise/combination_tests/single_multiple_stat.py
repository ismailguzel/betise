"""
JOINT STATISTICAL VALIDATION
Seasonality + Stationary

Tested combinations
-------------------
single_seasonality / multiple_seasonality

    x

AR / MA / ARMA / White Noise


Model
-----
    Y_t = B_t + F_t

where:
    B_t = stationary background
    F_t = deterministic Fourier seasonality


Validation
----------
For the SAME final combined series:

1. Detect the target Fourier seasonality.
2. Estimate and remove Fourier seasonality.
3. Check whether the recovered background is stationary.
4. Undo the known AR / MA / ARMA filter.
5. Check whether recovered innovations are white noise.

Joint success:
    seasonality detected
    AND background stationary
    AND recovered innovations white


Controls
--------
1. Stationary-only:
       No seasonality exists.
       Measures seasonal false-positive rate.

2. Random-walk + seasonality:
       Seasonality exists but background is non-stationary.
       Measures false stationary classification.


Plots
-----
For every case:

    Component 1: stationary background
    Component 2: Fourier seasonality
    Combination
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

STATIONARY_TYPES = ["ar", "ma", "arma", "white_noise"]

LENGTH = 400
N_TRIALS = 50
ALPHA = 0.05

TEST_LAG = 10
WARMUP = 50
NUM_HARMONICS = 1
SEED = 42

PERIOD_POOL = [7, 12, 24, 52]

SINGLE_CASES = [
    ("single", (p,))
    for p in PERIOD_POOL
]

MULTIPLE_CASES = [
    ("multiple", pair)
    for pair in combinations(PERIOD_POOL, 2)
]

SEASONAL_CASES = SINGLE_CASES + MULTIPLE_CASES

OUTPUT_DIR = Path("betise/combination_tests/test_outputs/seasonality_stationary")
PLOT_DIR = OUTPUT_DIR / "plots"

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# CASE LABEL
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
    Estimate deterministic Fourier seasonality from the
    FINAL observed series.

    HAC covariance is used because AR / MA / ARMA backgrounds
    may contain serial correlation.

    True Fourier coefficients are NOT used here.
    """

    y = np.asarray(series, dtype=float)
    X = build_fourier_design(
        len(y),
        periods,
        num_harmonics=NUM_HARMONICS
    )

    fit = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": TEST_LAG}
    )

    period_pvalues = {}
    terms_per_period = 2 * NUM_HARMONICS

    for i, period in enumerate(periods):
        first = 1 + i * terms_per_period
        R = np.zeros((terms_per_period, X.shape[1]))

        for j in range(terms_per_period):
            R[j, first + j] = 1.0

        period_pvalues[period] = float(
            fit.wald_test(R, scalar=True).pvalue
        )

    seasonal_detected = all(
        p < ALPHA
        for p in period_pvalues.values()
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
# STATIONARITY TESTS
# ============================================================

def stationarity_tests(series):
    """
    Conservative stationarity criterion:

    ADF:
        reject unit root
        p < 0.05

    KPSS:
        fail to reject stationarity
        p > 0.05

    stationary=True only if BOTH agree.
    """

    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    adf_p = float(
        adfuller(
            x,
            regression="c",
            autolag="AIC"
        )[1]
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            InterpolationWarning
        )

        kpss_p = float(
            kpss(
                x,
                regression="c",
                nlags="auto"
            )[1]
        )

    adf_stationary = adf_p < ALPHA
    kpss_stationary = kpss_p > ALPHA

    return {
        "adf_stationary": adf_stationary,
        "kpss_stationary": kpss_stationary,
        "stationary": (
            adf_stationary
            and kpss_stationary
        ),
        "adf_p": adf_p,
        "kpss_p": kpss_p,
    }


# ============================================================
# RECOVER INNOVATIONS
# ============================================================

def recover_innovations(series, stationary_kind, info):
    """
    Undo the known stationary model.

    AR:
        phi(B) X_t = epsilon_t

    MA:
        X_t = theta(B) epsilon_t

    ARMA:
        phi(B) X_t = theta(B) epsilon_t

    White noise:
        the background already is the innovation process.
    """

    y = np.asarray(series, dtype=float)

    if stationary_kind == "white_noise":
        return y

    if stationary_kind == "ar":
        ar_coefs = np.asarray(info["ar_coefs"], dtype=float)

        ar = np.r_[1.0, -ar_coefs]
        ma = np.array([1.0])

    elif stationary_kind == "ma":
        ma_coefs = np.asarray(info["ma_coefs"], dtype=float)

        ar = np.array([1.0])
        ma = np.r_[1.0, ma_coefs]

    elif stationary_kind == "arma":
        ar_coefs = np.asarray(info["ar_coefs"], dtype=float)
        ma_coefs = np.asarray(info["ma_coefs"], dtype=float)

        ar = np.r_[1.0, -ar_coefs]
        ma = np.r_[1.0, ma_coefs]

    else:
        raise ValueError(
            f"Unknown stationary type: {stationary_kind}"
        )

    return lfilter(ar, ma, y)


# ============================================================
# WHITENESS TEST
# ============================================================

def whiteness_test(innovations):
    """
    Recovered innovations should NOT contain significant
    serial autocorrelation.

    Therefore Ljung-Box p > 0.05 is considered success.
    """

    x = np.asarray(innovations, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    lb = acorr_ljungbox(
        x,
        lags=[TEST_LAG],
        return_df=True
    )

    p_value = float(
        lb["lb_pvalue"].iloc[0]
    )

    return {
        "white": p_value > ALPHA,
        "lb_p": p_value,
    }


# ============================================================
# GENERATION
# ============================================================

def generate_combination(kind, periods, stationary_kind):
    ts = TimeSeriesGenerator(length=LENGTH)

    background_df, background_info = (
        ts.generate_stationary_base_series(
            distribution=stationary_kind
        )
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

    combined_df, seasonal_info = (
        ts.compose_with_fourier_seasonality(**kwargs)
    )

    return (
        background_df,
        background_info,
        combined_df,
        seasonal_info,
    )


# ============================================================
# SINGLE JOINT TRIAL
# ============================================================

def run_joint_trial(kind, periods, stationary_kind):
    background_df, background_info, combined_df, _ = (
        generate_combination(
            kind,
            periods,
            stationary_kind
        )
    )

    source_background = (
        background_df["data"].to_numpy(dtype=float)
    )

    final_series = (
        combined_df["data"].to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # 1. SEASONALITY
    # --------------------------------------------------------

    seasonal = fit_seasonality(
        final_series,
        periods
    )

    recovered_background = seasonal["residuals"]

    # --------------------------------------------------------
    # 2. STATIONARITY
    # --------------------------------------------------------

    source_stationarity = stationarity_tests(
        source_background
    )

    recovered_stationarity = stationarity_tests(
        recovered_background
    )

    # --------------------------------------------------------
    # 3. RECOVER INNOVATIONS
    # --------------------------------------------------------

    source_innovations = recover_innovations(
        source_background,
        stationary_kind,
        background_info
    )[WARMUP:]

    recovered_innovations = recover_innovations(
        recovered_background,
        stationary_kind,
        background_info
    )[WARMUP:]

    source_white = whiteness_test(
        source_innovations
    )

    recovered_white = whiteness_test(
        recovered_innovations
    )

    # --------------------------------------------------------
    # 4. BACKGROUND RECOVERY DIAGNOSTIC
    # --------------------------------------------------------

    source_centered = (
        source_background
        - np.mean(source_background)
    )

    recovered_centered = (
        recovered_background
        - np.mean(recovered_background)
    )

    background_correlation = float(
        np.corrcoef(
            source_centered,
            recovered_centered
        )[0, 1]
    )

    seasonal_success = seasonal["detected"]
    stationary_success = recovered_stationarity["stationary"]
    structure_success = recovered_white["white"]

    joint_success = (
        seasonal_success
        and stationary_success
        and structure_success
    )

    return {
        "seasonal_detected": seasonal_success,
        "seasonal_std_ratio": seasonal["seasonal_std_ratio"],

        "source_ADF": source_stationarity["adf_stationary"],
        "source_KPSS": source_stationarity["kpss_stationary"],
        "source_stationary": source_stationarity["stationary"],

        "recovered_ADF": recovered_stationarity["adf_stationary"],
        "recovered_KPSS": recovered_stationarity["kpss_stationary"],
        "recovered_stationary": stationary_success,

        "source_white": source_white["white"],
        "recovered_white": structure_success,

        "background_correlation": background_correlation,

        "joint_success": joint_success,
    }


# ============================================================
# CONTROL 1
# STATIONARY WITHOUT SEASONALITY
# ============================================================

def run_seasonal_false_positive_control(periods, stationary_kind):
    """
    No Fourier exists.

    We nevertheless test the target periods.

    Detection = seasonal false positive.
    """

    ts = TimeSeriesGenerator(length=LENGTH)

    df, _ = ts.generate_stationary_base_series(
        distribution=stationary_kind
    )

    result = fit_seasonality(
        df["data"].to_numpy(dtype=float),
        periods
    )

    return result["detected"]


# ============================================================
# CONTROL 2
# RANDOM WALK + SEASONALITY
# ============================================================

def run_nonstationary_control(kind, periods):
    """
    Seasonality is real, but the background is a random walk.

    After seasonal estimation/removal the residual should
    NOT be classified as stationary.

    stationary=True here is a false positive.
    """

    ts = TimeSeriesGenerator(length=LENGTH)

    rw_df, _ = ts.generate_stochastic_trend(
        kind="rw"
    )

    kwargs = {
        "background_df": rw_df,
        "kind": kind,
        "difference_order": 1,
        "num_harmonics": NUM_HARMONICS,
    }

    if kind == "single":
        kwargs["period"] = periods[0]
    else:
        kwargs["periods"] = list(periods)

    combined_df, _ = (
        ts.compose_with_fourier_seasonality(**kwargs)
    )

    seasonal = fit_seasonality(
        combined_df["data"].to_numpy(dtype=float),
        periods
    )

    result = stationarity_tests(
        seasonal["residuals"]
    )

    return result["stationary"]


# ============================================================
# MAIN VALIDATION
# ============================================================

def run_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for stationary_kind in STATIONARY_TYPES:
            print(
                f"Running {label} + "
                f"{stationary_kind.upper()} ..."
            )

            for trial in range(N_TRIALS):
                result = run_joint_trial(
                    kind,
                    periods,
                    stationary_kind
                )

                rows.append({
                    "case": label,
                    "seasonality": kind,
                    "periods": "+".join(map(str, periods)),
                    "stationary": stationary_kind,
                    "trial": trial + 1,
                    **result,
                })

    return pd.DataFrame(rows)


# ============================================================
# SEASONAL FALSE-POSITIVE CONTROL
# ============================================================

def run_seasonal_fp_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for stationary_kind in STATIONARY_TYPES:
            print(
                f"Seasonal FP: {label} on "
                f"{stationary_kind.upper()} ..."
            )

            for trial in range(N_TRIALS):
                detected = (
                    run_seasonal_false_positive_control(
                        periods,
                        stationary_kind
                    )
                )

                rows.append({
                    "case": label,
                    "stationary": stationary_kind,
                    "trial": trial + 1,
                    "seasonal_false_positive": detected,
                })

    return pd.DataFrame(rows)


# ============================================================
# NON-STATIONARY CONTROL
# ============================================================

def run_nonstationary_validation():
    rows = []

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        print(
            f"Non-stationary control: {label} + RW ..."
        )

        for trial in range(N_TRIALS):
            false_stationary = (
                run_nonstationary_control(
                    kind,
                    periods
                )
            )

            rows.append({
                "case": label,
                "trial": trial + 1,
                "stationary_false_positive": false_stationary,
            })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================

def summarize_validation(results):
    return (
        results
        .groupby(
            [
                "case",
                "seasonality",
                "periods",
                "stationary",
            ],
            as_index=False
        )
        .agg(
            seasonal_detection_rate=(
                "seasonal_detected",
                "mean"
            ),

            source_stationary_rate=(
                "source_stationary",
                "mean"
            ),

            recovered_stationary_rate=(
                "recovered_stationary",
                "mean"
            ),

            source_whiteness_rate=(
                "source_white",
                "mean"
            ),

            recovered_whiteness_rate=(
                "recovered_white",
                "mean"
            ),

            joint_success_rate=(
                "joint_success",
                "mean"
            ),

            mean_background_correlation=(
                "background_correlation",
                "mean"
            ),

            mean_seasonal_std_ratio=(
                "seasonal_std_ratio",
                "mean"
            ),
        )
    )


def summarize_seasonal_fp(results):
    return (
        results
        .groupby(
            ["case", "stationary"],
            as_index=False
        )
        .agg(
            seasonal_false_positive_rate=(
                "seasonal_false_positive",
                "mean"
            )
        )
    )


def summarize_nonstationary(results):
    return (
        results
        .groupby(
            "case",
            as_index=False
        )
        .agg(
            stationary_false_positive_rate=(
                "stationary_false_positive",
                "mean"
            )
        )
    )


# ============================================================
# REPRESENTATIVE PLOTS
# ============================================================

def save_representative_plots():
    """
    Three stacked panels for every combination:

        Component 1 = stationary background
        Component 2 = Fourier seasonality
        Combination = final series
    """

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for kind, periods in SEASONAL_CASES:
        label = case_label(kind, periods)

        for stationary_kind in STATIONARY_TYPES:
            background_df, _, combined_df, _ = (
                generate_combination(
                    kind,
                    periods,
                    stationary_kind
                )
            )

            time = combined_df["time"].to_numpy()

            background = (
                background_df["data"].to_numpy(
                    dtype=float
                )
            )

            combined = (
                combined_df["data"].to_numpy(
                    dtype=float
                )
            )

            fourier = combined - background

            fig, axes = plt.subplots(
                3,
                1,
                figsize=(12, 9),
                sharex=True
            )

            axes[0].plot(
                time,
                background,
                linewidth=1.1
            )

            axes[0].set_title(
                f"{label} + {stationary_kind.upper()} "
                "— Component 1: Stationary"
            )

            axes[0].set_ylabel("Value")
            axes[0].grid(alpha=0.3)

            axes[1].plot(
                time,
                fourier,
                linewidth=1.1
            )

            axes[1].set_title(
                f"{label} + {stationary_kind.upper()} "
                "— Component 2: Fourier"
            )

            axes[1].set_ylabel("Value")
            axes[1].grid(alpha=0.3)

            axes[2].plot(
                time,
                combined,
                linewidth=1.1
            )

            axes[2].set_title(
                f"{label} + {stationary_kind.upper()} "
                "— Combination"
            )

            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Value")
            axes[2].grid(alpha=0.3)

            plt.tight_layout()

            filename = (
                f"{label}_{stationary_kind}_components.png"
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
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========================================")
    print("SEASONALITY + STATIONARY")
    print("JOINT STATISTICAL VALIDATION")
    print("========================================\n")

    trial_results = run_validation()
    seasonal_fp_results = run_seasonal_fp_validation()
    nonstationary_results = run_nonstationary_validation()

    summary = summarize_validation(trial_results)
    seasonal_fp_summary = summarize_seasonal_fp(
        seasonal_fp_results
    )
    nonstationary_summary = summarize_nonstationary(
        nonstationary_results
    )

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

    nonstationary_summary.to_csv(
        OUTPUT_DIR / "nonstationary_control.csv",
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
    print("NON-STATIONARY CONTROL")
    print("========================================")
    print(
        nonstationary_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    save_representative_plots()

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )