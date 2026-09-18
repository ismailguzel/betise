"""
JOINT STATISTICAL VALIDATION
Deterministic SARMA / SARIMA + Volatility

Models
------
SARMA:
    Y_t = F_t + ARMA(epsilon_t)

SARIMA:
    Y_t = F_t + ARIMA(epsilon_t)

where epsilon_t comes from:
    ARCH / GARCH / EGARCH / APARCH

Validation
----------
1. Is the target deterministic seasonality detectable
   in the FINAL combined series?

2. After estimating/removing seasonality and undoing
   ARMA/ARIMA dynamics, is volatility still detectable?

3. Joint success:
       seasonality detected AND volatility detected
   in the same trial.

Controls
--------
Seasonal false positive:
    Test the volatility-driven nonseasonal background alone
    for the target seasonal period.

Volatility false positive:
    Run the same deterministic SARMA/SARIMA pipeline with
    Gaussian innovations instead of volatility innovations.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ALPHA = 0.05
TEST_LAG = 10
WARMUP = 50
NUM_HARMONICS = 1
SEED = 42

OUTPUT_DIR = Path("betise/combination_tests/test_outputs/deterministic_sarma_sarima_volatility")
PLOT_DIR = OUTPUT_DIR / "plots"

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# FOURIER RECONSTRUCTION
# ============================================================

def reconstruct_fourier(info, length):
    """
    Reconstruct the exact deterministic Fourier component
    from stored FINAL Fourier coefficients.
    """

    t = np.arange(length)
    period = int(info["periods"][0])
    fourier = np.zeros(length, dtype=float)

    for coef in info["fourier_coefficients"]:
        k = int(coef["harmonic"])
        fourier += coef["sin_coef"] * np.sin(2 * np.pi * k * t / period)
        fourier += coef["cos_coef"] * np.cos(2 * np.pi * k * t / period)

    return fourier


# ============================================================
# PREPROCESS FOR SEASONALITY / ARMA RECOVERY
# ============================================================

def difference_for_model(series, model, info):
    """
    SARMA:
        no differencing

    SARIMA:
        difference d times

    After this operation the non-seasonal background should
    be stationary ARMA.
    """

    x = np.asarray(series, dtype=float)

    if model == "sarma":
        return x, 0

    d = int(info["diff"])
    return np.diff(x, n=d), d


# ============================================================
# FOURIER REGRESSION
# ============================================================

def build_fourier_design(length, period, num_harmonics=1, start_index=0):
    t = np.arange(start_index, start_index + length)
    columns = [np.ones(length)]

    for k in range(1, num_harmonics + 1):
        columns.append(np.sin(2 * np.pi * k * t / period))
        columns.append(np.cos(2 * np.pi * k * t / period))

    return np.column_stack(columns)


def fit_seasonality(series, period, num_harmonics=1, start_index=0):
    """
    Estimate deterministic seasonality from the observed series.

    HC3 covariance is used because the residual background
    may be heteroskedastic.
    """

    y = np.asarray(series, dtype=float)
    X = build_fourier_design(
        len(y),
        period,
        num_harmonics=num_harmonics,
        start_index=start_index
    )

    fit = sm.OLS(y, X).fit(cov_type="HC3")

    n_terms = 2 * num_harmonics
    R = np.zeros((n_terms, X.shape[1]))

    for j in range(n_terms):
        R[j, j + 1] = 1.0

    p_value = float(fit.wald_test(R, scalar=True).pvalue)

    seasonal_fit = X[:, 1:] @ fit.params[1:]
    residuals = np.asarray(fit.resid, dtype=float)

    resid_std = np.std(residuals)
    seasonal_ratio = (
        np.std(seasonal_fit) / resid_std
        if resid_std > 1e-12 else np.inf
    )

    return {
        "detected": p_value < ALPHA,
        "p_value": p_value,
        "seasonal_fit": seasonal_fit,
        "residuals": residuals,
        "seasonal_std_ratio": seasonal_ratio,
    }


# ============================================================
# RECOVER VOLATILITY INNOVATIONS
# ============================================================

def recover_innovations(arma_process, info):
    """
    Undo the non-seasonal ARMA filter.

    At this stage SARIMA has already been differenced,
    so SARMA and SARIMA use the same inverse-filter logic.
    """

    y = np.asarray(arma_process, dtype=float)

    ar_coefs = np.asarray(info["ar_coefs"], dtype=float)
    ma_coefs = np.asarray(info["ma_coefs"], dtype=float)

    ar = np.r_[1.0, -ar_coefs]
    ma = np.r_[1.0, ma_coefs]

    return lfilter(ar, ma, y)


# ============================================================
# VOLATILITY TESTS
# ============================================================

def volatility_tests(residuals):
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

def generate_combination(model, period, volatility_kind):
    ts = TimeSeriesGenerator(length=LENGTH)

    innovations, volatility_info = ts.generate_volatility(
        kind=volatility_kind,
        as_innovations=True
    )

    innovations = np.asarray(innovations, dtype=float)

    if model == "sarma":
        df, info = ts.generate_deterministic_sarma(
            period=period,
            num_harmonics=NUM_HARMONICS,
            innovations=innovations
        )
    else:
        df, info = ts.generate_deterministic_sarima(
            period=period,
            d=1,
            num_harmonics=NUM_HARMONICS,
            innovations=innovations
        )

    return df, info, innovations, volatility_info


# ============================================================
# SINGLE JOINT TRIAL
# ============================================================

def run_joint_trial(model, period, volatility_kind):
    df, info, innovations, _ = generate_combination(
        model,
        period,
        volatility_kind
    )

    final_series = df["data"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # TRUE decomposition
    # Only used for control / diagnostics.
    # Main recovery still estimates seasonality from final Y_t.
    # --------------------------------------------------------

    true_fourier = reconstruct_fourier(info, LENGTH)
    true_background = final_series - true_fourier

    # --------------------------------------------------------
    # PREPROCESS FINAL SERIES
    # --------------------------------------------------------

    processed, d = difference_for_model(final_series, model, info)

    # np.diff shifts the effective starting index.
    start_index = d

    # --------------------------------------------------------
    # SEASONALITY DETECTION
    # --------------------------------------------------------

    seasonal = fit_seasonality(
        processed,
        period,
        num_harmonics=NUM_HARMONICS,
        start_index=start_index
    )

    # --------------------------------------------------------
    # VOLATILITY RECOVERY
    #
    # residual after Fourier regression should correspond to
    # stationary ARMA background.
    # --------------------------------------------------------

    recovered = recover_innovations(
        seasonal["residuals"],
        info
    )

    recovered = recovered[WARMUP:]
    source = innovations[-len(recovered):]

    source_vol = volatility_tests(source)
    recovered_vol = volatility_tests(recovered)

    # --------------------------------------------------------
    # SEASONAL FALSE POSITIVE CONTROL
    #
    # Remove TRUE Fourier from the same generated trial and
    # ask whether the nonseasonal volatility-driven background
    # falsely appears seasonal at the target period.
    # --------------------------------------------------------

    background_processed, bg_d = difference_for_model(
        true_background,
        model,
        info
    )

    background_seasonal_test = fit_seasonality(
        background_processed,
        period,
        num_harmonics=NUM_HARMONICS,
        start_index=bg_d
    )

    seasonal_success = seasonal["detected"]

    return {
        "seasonal_detected": seasonal_success,
        "seasonal_p": seasonal["p_value"],
        "seasonal_std_ratio": seasonal["seasonal_std_ratio"],

        "background_seasonal_FP": background_seasonal_test["detected"],

        "source_ARCH": source_vol["arch_detected"],
        "source_LB2": source_vol["lb2_detected"],
        "source_any": source_vol["any_detected"],
        "source_both": source_vol["both_detected"],

        "recovered_ARCH": recovered_vol["arch_detected"],
        "recovered_LB2": recovered_vol["lb2_detected"],
        "recovered_any": recovered_vol["any_detected"],
        "recovered_both": recovered_vol["both_detected"],

        "joint_any": seasonal_success and recovered_vol["any_detected"],
        "joint_strict": seasonal_success and recovered_vol["both_detected"],
    }


# ============================================================
# GAUSSIAN CONTROL
# ============================================================

def run_gaussian_control(model, period):
    """
    Same deterministic seasonal model, but innovations are
    ordinary constant-variance Gaussian noise.

    Any recovered volatility detection is a false positive.
    """

    ts = TimeSeriesGenerator(length=LENGTH)
    innovations = np.random.normal(0, 1, LENGTH)

    if model == "sarma":
        df, info = ts.generate_deterministic_sarma(
            period=period,
            num_harmonics=NUM_HARMONICS,
            innovations=innovations
        )
    else:
        df, info = ts.generate_deterministic_sarima(
            period=period,
            d=1,
            num_harmonics=NUM_HARMONICS,
            innovations=innovations
        )

    series = df["data"].to_numpy(dtype=float)
    processed, d = difference_for_model(series, model, info)

    seasonal = fit_seasonality(
        processed,
        period,
        num_harmonics=NUM_HARMONICS,
        start_index=d
    )

    recovered = recover_innovations(
        seasonal["residuals"],
        info
    )[WARMUP:]

    result = volatility_tests(recovered)

    return {
        "seasonal_detected": seasonal["detected"],
        "volatility_ARCH_FP": result["arch_detected"],
        "volatility_LB2_FP": result["lb2_detected"],
        "volatility_any_FP": result["any_detected"],
        "volatility_both_FP": result["both_detected"],
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def run_validation():
    rows = []

    for model in MODELS:
        for period in PERIODS:
            for volatility_kind in VOLATILITY_TYPES:

                print(
                    f"Running {model.upper()} "
                    f"(period={period}) + "
                    f"{volatility_kind.upper()} ..."
                )

                for trial in range(N_TRIALS):
                    result = run_joint_trial(
                        model,
                        period,
                        volatility_kind
                    )

                    rows.append({
                        "model": model,
                        "period": period,
                        "volatility": volatility_kind,
                        "trial": trial + 1,
                        **result,
                    })

    return pd.DataFrame(rows)


# ============================================================
# GAUSSIAN VALIDATION
# ============================================================

def run_gaussian_validation():
    rows = []

    for model in MODELS:
        for period in PERIODS:

            print(
                f"Gaussian control: "
                f"{model.upper()} period={period} ..."
            )

            for trial in range(N_TRIALS):
                result = run_gaussian_control(model, period)

                rows.append({
                    "model": model,
                    "period": period,
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
        .groupby(["model", "period", "volatility"], as_index=False)
        .agg(
            seasonal_detection_rate=("seasonal_detected", "mean"),
            seasonal_false_positive_rate=("background_seasonal_FP", "mean"),

            source_ARCH=("source_ARCH", "mean"),
            source_LB2=("source_LB2", "mean"),
            source_any=("source_any", "mean"),
            source_both=("source_both", "mean"),

            recovered_ARCH=("recovered_ARCH", "mean"),
            recovered_LB2=("recovered_LB2", "mean"),
            recovered_any=("recovered_any", "mean"),
            recovered_both=("recovered_both", "mean"),

            joint_success_any=("joint_any", "mean"),
            joint_success_strict=("joint_strict", "mean"),

            mean_seasonal_std_ratio=("seasonal_std_ratio", "mean"),
        )
    )


def summarize_gaussian(results):
    return (
        results
        .groupby(["model", "period"], as_index=False)
        .agg(
            seasonal_detection_rate=("seasonal_detected", "mean"),
            ARCH_false_positive=("volatility_ARCH_FP", "mean"),
            LB2_false_positive=("volatility_LB2_FP", "mean"),
            ANY_false_positive=("volatility_any_FP", "mean"),
            BOTH_false_positive=("volatility_both_FP", "mean"),
        )
    )


# ============================================================
# REPRESENTATIVE COMPONENT PLOTS
# ============================================================

def save_representative_plots():
    """
    One figure per model + period + volatility combination.

    Panel 1:
        Deterministic Fourier component.

    Panel 2:
        Volatility-driven ARMA / ARIMA background.

    Panel 3:
        Final combined deterministic SARMA / SARIMA series.
    """

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        for period in PERIODS:
            for volatility_kind in VOLATILITY_TYPES:

                df, info, _, _ = generate_combination(
                    model,
                    period,
                    volatility_kind
                )

                combined = df["data"].to_numpy(dtype=float)
                fourier = reconstruct_fourier(info, LENGTH)
                background = combined - fourier
                time = np.arange(LENGTH)

                fig, axes = plt.subplots(
                    3, 1,
                    figsize=(12, 9),
                    sharex=True
                )

                axes[0].plot(time, fourier, linewidth=1.1)
                axes[0].set_title(
                    f"{model.upper()} + {volatility_kind.upper()} "
                    f"(period={period}) — Component 1: Fourier"
                )
                axes[0].set_ylabel("Value")
                axes[0].grid(alpha=0.3)

                axes[1].plot(time, background, linewidth=1.1)
                axes[1].set_title(
                    f"{model.upper()} + {volatility_kind.upper()} "
                    "— Component 2: Volatility-driven "
                    f"{'ARMA' if model == 'sarma' else 'ARIMA'}"
                )
                axes[1].set_ylabel("Value")
                axes[1].grid(alpha=0.3)

                axes[2].plot(time, combined, linewidth=1.1)
                axes[2].set_title(
                    f"{model.upper()} + {volatility_kind.upper()} "
                    "— Combination"
                )
                axes[2].set_xlabel("Time")
                axes[2].set_ylabel("Value")
                axes[2].grid(alpha=0.3)

                plt.tight_layout()

                filename = (
                    f"{model}_p{period}_"
                    f"{volatility_kind}_components.png"
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
    print("DETERMINISTIC SARMA / SARIMA + VOLATILITY")
    print("JOINT STATISTICAL VALIDATION")
    print("========================================\n")

    trial_results = run_validation()
    gaussian_results = run_gaussian_validation()

    summary = summarize_validation(trial_results)
    gaussian_summary = summarize_gaussian(gaussian_results)

    trial_results.to_csv(
        OUTPUT_DIR / "trial_results.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "joint_detection_rates.csv",
        index=False
    )

    gaussian_summary.to_csv(
        OUTPUT_DIR / "gaussian_control.csv",
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
    print("GAUSSIAN FALSE-POSITIVE CONTROL")
    print("========================================")
    print(
        gaussian_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    save_representative_plots()

    print(f"\nResults saved to:\n{OUTPUT_DIR.resolve()}")