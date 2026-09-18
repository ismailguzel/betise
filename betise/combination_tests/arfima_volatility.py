"""
JOINT STATISTICAL VALIDATION — V2
ARFIMA + Volatility

Candidate combinations
----------------------
ARFIMA + ARCH
ARFIMA + GARCH
ARFIMA + EGARCH
ARFIMA + APARCH

Model
-----
    Phi(B) (1-B)^d X_t = Theta(B) epsilon_t

where:
    0 < d < 0.5
    epsilon_t follows an ARCH/GARCH/EGARCH/APARCH volatility process.

This is an innovation-driven composition.
The volatility process is NOT added to ARFIMA in the observation domain.

Why V2?
-------
The first validation used a fixed GPH threshold (d_hat >= 0.15).
Under heteroskedastic short-memory controls, that produced too many
long-memory false positives.

V2 therefore calibrates a SEPARATE empirical long-memory threshold
for each volatility model:

    threshold(model)
        = 95th percentile of d_hat under d = 0

The calibration samples are independent of the evaluation samples.

Validation questions
--------------------
1. Does the final ARFIMA + volatility series preserve fractional long memory?
2. Is d estimated without systematic bias?
3. Can the volatility innovations be approximately recovered from the final
   ARFIMA series using the known generator parameters?
4. Is volatility still statistically detectable after recovery?

Important reporting distinction
-------------------------------
- Long-memory detection:
      empirical model-specific threshold from independent d=0 calibration.

- Parameter preservation:
      bias, MAE, RMSE of d_hat - d_true.
      No arbitrary per-trial +/- tolerance is used as a success criterion.

- Volatility preservation:
      source vs recovered ARCH/volatility detectability and innovation recovery.

Controls
--------
1. Independent d=0 + volatility HOLDOUT control:
       verifies that the calibrated long-memory threshold produces
       approximately 5% false positives.

2. Gaussian innovations + ARFIMA:
       volatility false-positive control.

Representative plots
--------------------
Three stacked panels:
    Component 1 : volatility innovations
    Component 2 : fractionally integrated innovations
    Combination : final ARFIMA + volatility series
"""

from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.signal import fftconvolve, lfilter
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from betise.core.generator import TimeSeriesGenerator


# ============================================================
# SETTINGS
# ============================================================

LENGTH = 400
NUMSEAS = 100

# Evaluation trials for true ARFIMA + volatility combinations.
N_TRIALS = 50

# Independent d=0 trials used ONLY to calibrate the detector threshold.
N_CALIBRATION_TRIALS = 150

# Independent d=0 trials used to verify the calibrated false-positive rate.
N_HOLDOUT_CONTROL_TRIALS = 100

# Gaussian ARFIMA trials for volatility false-positive control.
N_VOLATILITY_FP_TRIALS = 200

SEED = 42
D_RANGE = (0.25, 0.49)

ALPHA = 0.05
ARCH_LAGS = 10
LJUNG_BOX_LAG = 10

GPH_BANDWIDTH_EXPONENT = 0.60
EMPIRICAL_THRESHOLD_QUANTILE = 0.95

# Beginning of the observed series is excluded from inverse-filter diagnostics
# because the final output starts after hidden ARFIMA burn-in.
RECOVERY_TRANSIENT = 75

VOLATILITY_MODELS = [
    "arch",
    "garch",
    "egarch",
    "aparch",
]

OUTPUT_DIR = Path("betise/combination_tests/test_outputs/arfima_volatility")
PLOT_DIR = OUTPUT_DIR / "plots"

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# FRACTIONAL FILTER HELPERS
# ============================================================

def fractional_integration_weights(d, length):
    """
    Coefficients of:

        (1-B)^(-d)

    psi_0 = 1
    psi_k = psi_(k-1) * (k - 1 + d) / k
    """

    weights = np.empty(length, dtype=float)
    weights[0] = 1.0

    for k in range(1, length):
        weights[k] = (
            weights[k - 1]
            * (k - 1 + d)
            / k
        )

    return weights


def fractional_difference_weights(d, length):
    """
    Coefficients of:

        (1-B)^d

    pi_0 = 1
    pi_k = pi_(k-1) * (k - 1 - d) / k
    """

    weights = np.empty(length, dtype=float)
    weights[0] = 1.0

    for k in range(1, length):
        weights[k] = (
            weights[k - 1]
            * (k - 1 - d)
            / k
        )

    return weights


def fractionally_integrate(innovations, d):
    innovations = np.asarray(innovations, dtype=float)

    weights = fractional_integration_weights(
        d,
        len(innovations)
    )

    return fftconvolve(
        innovations,
        weights,
        mode="full"
    )[:len(innovations)]


def fractionally_difference(series, d):
    series = np.asarray(series, dtype=float)

    weights = fractional_difference_weights(
        d,
        len(series)
    )

    return fftconvolve(
        series,
        weights,
        mode="full"
    )[:len(series)]


# ============================================================
# KNOWN-ARMA INVERSE FILTER
# ============================================================

def remove_short_memory_arma(series, ar_coefs, ma_coefs):
    """
    Given:

        Phi(B) X_t = Theta(B) W_t

    recover approximately:

        W_t = Theta(B)^(-1) Phi(B) X_t

    Ground-truth AR/MA parameters are intentionally used because this is
    generator validation, not blind downstream parameter estimation.
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

    return lfilter(
        ar_poly,
        ma_poly,
        x
    )


def recover_innovations(final_series, d, ar_coefs, ma_coefs):
    """
    Approximate inverse of the external-innovation ARFIMA path:

        final X
          -> remove ARMA short-memory filter
          -> fractional difference by d
          -> recovered epsilon
    """

    fractional_component = remove_short_memory_arma(
        final_series,
        ar_coefs,
        ma_coefs
    )

    return fractionally_difference(
        fractional_component,
        d
    )


# ============================================================
# GPH d ESTIMATOR
# ============================================================

def estimate_d_gph(series, bandwidth_exponent=GPH_BANDWIDTH_EXPONENT):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    n = len(x)

    if n < 50:
        raise ValueError(
            "GPH estimation requires at least 50 observations."
        )

    m = int(np.floor(n ** bandwidth_exponent))
    m = max(8, min(m, n // 4))

    fft_values = np.fft.fft(x)

    j = np.arange(1, m + 1)
    lambdas = 2 * np.pi * j / n

    periodogram = (
        np.abs(fft_values[j]) ** 2
        / (2 * np.pi * n)
    )

    periodogram = np.maximum(
        periodogram,
        np.finfo(float).tiny
    )

    regressor = np.log(
        4 * np.sin(lambdas / 2) ** 2
    )

    response = np.log(periodogram)

    X = sm.add_constant(regressor)
    fit = sm.OLS(response, X).fit()

    return {
        "d_hat": float(-fit.params[1]),
        "slope_se": float(fit.bse[1]),
        "r_squared": float(fit.rsquared),
        "bandwidth": m,
    }


# ============================================================
# VOLATILITY DETECTION
# ============================================================

def detect_volatility(innovations):
    """
    Detect conditional heteroskedasticity with:

    1. Engle ARCH-LM
    2. Ljung-Box on squared innovations

    Volatility is considered detected if either test is significant.
    """

    x = np.asarray(innovations, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)

    if len(x) <= max(ARCH_LAGS, LJUNG_BOX_LAG) + 10:
        raise ValueError(
            "Not enough observations for volatility diagnostics."
        )

    _, arch_lm_pvalue, _, _ = het_arch(
        x,
        nlags=ARCH_LAGS
    )

    lb = acorr_ljungbox(
        x ** 2,
        lags=[LJUNG_BOX_LAG],
        return_df=True
    )

    squared_lb_pvalue = float(
        lb["lb_pvalue"].iloc[-1]
    )

    detected = (
        arch_lm_pvalue < ALPHA
        or squared_lb_pvalue < ALPHA
    )

    return {
        "detected": bool(detected),
        "arch_lm_pvalue": float(arch_lm_pvalue),
        "squared_lb_pvalue": squared_lb_pvalue,
    }


# ============================================================
# GENERATORS
# ============================================================

def generate_volatility_innovations(volatility_model):
    total = LENGTH + NUMSEAS

    vol_ts = TimeSeriesGenerator(
        length=total
    )

    innovations, info = vol_ts.generate_volatility(
        kind=volatility_model,
        as_innovations=True
    )

    return np.asarray(innovations, dtype=float), info


def generate_combination(volatility_model):
    innovations, volatility_info = (
        generate_volatility_innovations(
            volatility_model
        )
    )

    ts = TimeSeriesGenerator(
        length=LENGTH
    )

    df, arfima_info = ts.generate_fractional_process(
        kind="arfima",
        d_range=D_RANGE,
        numseas=NUMSEAS,
        innovations=innovations
    )

    return (
        innovations,
        volatility_info,
        df,
        arfima_info
    )


def generate_d0_control(volatility_model):
    """
    Same external-volatility pipeline, but d is fixed to zero.

    This is the correct null model for empirical long-memory calibration.
    """

    innovations, volatility_info = (
        generate_volatility_innovations(
            volatility_model
        )
    )

    ts = TimeSeriesGenerator(
        length=LENGTH
    )

    df, info = ts.generate_fractional_process(
        kind="arfima",
        d_range=(0.0, 0.0),
        numseas=NUMSEAS,
        innovations=innovations
    )

    return (
        innovations,
        volatility_info,
        df,
        info
    )


# ============================================================
# LONG-MEMORY NULL STATISTIC
# ============================================================

def get_d_hat_from_final(df, info):
    """
    Remove only the known short-memory ARMA part.

    If d > 0, fractional dependence remains.
    If d = 0, the result should contain only the volatility innovations
    up to finite-sample/filter transients.
    """

    final_series = df["data"].to_numpy(dtype=float)

    fractional_component = remove_short_memory_arma(
        final_series,
        info.get("ar_coefs"),
        info.get("ma_coefs")
    )

    return estimate_d_gph(
        fractional_component[
            RECOVERY_TRANSIENT:
        ]
    )["d_hat"]


# ============================================================
# STEP 1 — EMPIRICAL THRESHOLD CALIBRATION
# ============================================================

def calibrate_long_memory_thresholds():
    """
    For each volatility model, estimate the 95th percentile of d_hat
    under the null:

        d = 0 + same volatility model.

    These samples are used ONLY for calibration.
    """

    rows = []
    threshold_rows = []

    for volatility_model in VOLATILITY_MODELS:
        print(
            f"Calibrating d=0 threshold for "
            f"{volatility_model.upper()} ..."
        )

        model_d_hats = []

        for trial in range(N_CALIBRATION_TRIALS):
            _, _, df, info = generate_d0_control(
                volatility_model
            )

            d_hat = get_d_hat_from_final(
                df,
                info
            )

            model_d_hats.append(d_hat)

            rows.append({
                "volatility_model": volatility_model,
                "trial": trial + 1,
                "d_hat": d_hat,
            })

        threshold = float(
            np.quantile(
                model_d_hats,
                EMPIRICAL_THRESHOLD_QUANTILE
            )
        )

        threshold_rows.append({
            "volatility_model": volatility_model,
            "quantile": EMPIRICAL_THRESHOLD_QUANTILE,
            "n_calibration": N_CALIBRATION_TRIALS,
            "empirical_d_threshold": threshold,
            "null_mean_d_hat": float(np.mean(model_d_hats)),
            "null_std_d_hat": float(np.std(model_d_hats, ddof=1)),
        })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(threshold_rows)
    )


# ============================================================
# STEP 2 — INDEPENDENT d=0 HOLDOUT CONTROL
# ============================================================

def run_long_memory_holdout_control(thresholds):
    """
    Independent null samples verify the actual false-positive rate
    of the calibrated threshold.
    """

    threshold_map = dict(
        zip(
            thresholds["volatility_model"],
            thresholds["empirical_d_threshold"]
        )
    )

    rows = []

    for volatility_model in VOLATILITY_MODELS:
        threshold = threshold_map[
            volatility_model
        ]

        print(
            f"Running independent d=0 holdout for "
            f"{volatility_model.upper()} ..."
        )

        for trial in range(N_HOLDOUT_CONTROL_TRIALS):
            _, _, df, info = generate_d0_control(
                volatility_model
            )

            d_hat = get_d_hat_from_final(
                df,
                info
            )

            rows.append({
                "volatility_model": volatility_model,
                "trial": trial + 1,
                "empirical_d_threshold": threshold,
                "d_hat": d_hat,
                "false_long_memory": d_hat > threshold,
            })

    return pd.DataFrame(rows)


# ============================================================
# STEP 3 — TRUE ARFIMA + VOLATILITY EVALUATION
# ============================================================

def run_joint_trial(volatility_model, d_threshold):
    (
        source_innovations,
        volatility_info,
        df,
        arfima_info
    ) = generate_combination(
        volatility_model
    )

    final_series = df["data"].to_numpy(dtype=float)

    true_d = float(
        arfima_info["d"]
    )

    ar_coefs = arfima_info.get(
        "ar_coefs"
    )

    ma_coefs = arfima_info.get(
        "ma_coefs"
    )

    # --------------------------------------------------------
    # LONG MEMORY
    # --------------------------------------------------------

    d_hat = get_d_hat_from_final(
        df,
        arfima_info
    )

    d_error = d_hat - true_d

    long_memory_detected = (
        d_hat > d_threshold
    )

    # --------------------------------------------------------
    # RECOVER VOLATILITY INNOVATIONS
    # --------------------------------------------------------

    recovered_innovations = recover_innovations(
        final_series,
        true_d,
        ar_coefs,
        ma_coefs
    )

    source_segment = source_innovations[
        NUMSEAS:NUMSEAS + LENGTH
    ]

    source_eval = source_segment[
        RECOVERY_TRANSIENT:
    ]

    recovered_eval = recovered_innovations[
        RECOVERY_TRANSIENT:
    ]

    source_volatility = detect_volatility(
        source_eval
    )

    recovered_volatility = detect_volatility(
        recovered_eval
    )

    conditional_volatility_preserved = (
        recovered_volatility["detected"]
        if source_volatility["detected"]
        else np.nan
    )

    innovation_correlation = float(
        np.corrcoef(
            source_eval,
            recovered_eval
        )[0, 1]
    )

    squared_innovation_correlation = float(
        np.corrcoef(
            source_eval ** 2,
            recovered_eval ** 2
        )[0, 1]
    )

    # Strict observational joint detection.
    strict_joint_detection = (
        long_memory_detected
        and recovered_volatility["detected"]
    )

    # Fair composition success:
    # only score volatility preservation when volatility was detectable
    # in its own source innovation sequence.
    conditional_joint_success = (
        bool(
            long_memory_detected
            and recovered_volatility["detected"]
        )
        if source_volatility["detected"]
        else np.nan
    )

    return {
        "volatility_model": volatility_model,

        "empirical_d_threshold": d_threshold,

        "true_d": true_d,
        "d_hat": d_hat,
        "d_error": d_error,
        "d_abs_error": abs(d_error),
        "d_squared_error": d_error ** 2,

        "long_memory_detected":
            long_memory_detected,

        "source_volatility_detected":
            source_volatility["detected"],

        "recovered_volatility_detected":
            recovered_volatility["detected"],

        "conditional_volatility_preserved":
            conditional_volatility_preserved,

        "source_arch_lm_pvalue":
            source_volatility["arch_lm_pvalue"],

        "recovered_arch_lm_pvalue":
            recovered_volatility["arch_lm_pvalue"],

        "source_squared_lb_pvalue":
            source_volatility["squared_lb_pvalue"],

        "recovered_squared_lb_pvalue":
            recovered_volatility["squared_lb_pvalue"],

        "innovation_correlation":
            innovation_correlation,

        "squared_innovation_correlation":
            squared_innovation_correlation,

        "strict_joint_detection":
            strict_joint_detection,

        "conditional_joint_success":
            conditional_joint_success,

        "external_innovations_used":
            arfima_info.get(
                "external_innovations_used"
            ),

        "volatility_subtype":
            volatility_info.get(
                "subtype"
            ),
    }


def run_validation(thresholds):
    threshold_map = dict(
        zip(
            thresholds["volatility_model"],
            thresholds["empirical_d_threshold"]
        )
    )

    rows = []

    for volatility_model in VOLATILITY_MODELS:
        threshold = threshold_map[
            volatility_model
        ]

        print(
            f"Evaluating ARFIMA + "
            f"{volatility_model.upper()} "
            f"(threshold={threshold:.3f}) ..."
        )

        for trial in range(N_TRIALS):
            result = run_joint_trial(
                volatility_model,
                threshold
            )

            rows.append({
                "trial": trial + 1,
                **result,
            })

    return pd.DataFrame(rows)


# ============================================================
# VOLATILITY FALSE-POSITIVE CONTROL
# ============================================================

def run_volatility_false_positive_control():
    """
    Gaussian innovations + genuine ARFIMA.

    After exact generator-aware inverse filtering there should be no
    conditional heteroskedasticity.
    """

    total = LENGTH + NUMSEAS

    gaussian_innovations = np.random.normal(
        0,
        1,
        total
    )

    ts = TimeSeriesGenerator(
        length=LENGTH
    )

    df, arfima_info = ts.generate_fractional_process(
        kind="arfima",
        d_range=D_RANGE,
        numseas=NUMSEAS,
        innovations=gaussian_innovations
    )

    recovered = recover_innovations(
        df["data"].to_numpy(dtype=float),
        float(arfima_info["d"]),
        arfima_info.get("ar_coefs"),
        arfima_info.get("ma_coefs")
    )

    result = detect_volatility(
        recovered[
            RECOVERY_TRANSIENT:
        ]
    )

    return {
        "false_positive": result["detected"],
        "arch_lm_pvalue": result["arch_lm_pvalue"],
        "squared_lb_pvalue": result["squared_lb_pvalue"],
    }


def run_volatility_fp_validation():
    rows = []

    print(
        "Running Gaussian-innovation "
        "volatility false-positive control ..."
    )

    for trial in range(N_VOLATILITY_FP_TRIALS):
        result = run_volatility_false_positive_control()

        rows.append({
            "trial": trial + 1,
            **result,
        })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================

def summarize_validation(results):
    summary = (
        results
        .groupby(
            "volatility_model",
            as_index=False
        )
        .agg(
            empirical_d_threshold=(
                "empirical_d_threshold",
                "first"
            ),

            long_memory_detection_rate=(
                "long_memory_detected",
                "mean"
            ),

            source_volatility_detection_rate=(
                "source_volatility_detected",
                "mean"
            ),

            recovered_volatility_detection_rate=(
                "recovered_volatility_detected",
                "mean"
            ),

            conditional_volatility_preservation_rate=(
                "conditional_volatility_preserved",
                "mean"
            ),

            strict_joint_detection_rate=(
                "strict_joint_detection",
                "mean"
            ),

            conditional_joint_success_rate=(
                "conditional_joint_success",
                "mean"
            ),

            mean_true_d=(
                "true_d",
                "mean"
            ),

            mean_d_hat=(
                "d_hat",
                "mean"
            ),

            d_bias=(
                "d_error",
                "mean"
            ),

            d_mae=(
                "d_abs_error",
                "mean"
            ),

            mean_squared_error=(
                "d_squared_error",
                "mean"
            ),

            mean_innovation_correlation=(
                "innovation_correlation",
                "mean"
            ),

            mean_squared_innovation_correlation=(
                "squared_innovation_correlation",
                "mean"
            ),
        )
    )

    summary["d_rmse"] = np.sqrt(
        summary["mean_squared_error"]
    )

    summary = summary.drop(
        columns=["mean_squared_error"]
    )

    return summary


def summarize_holdout_control(results):
    return (
        results
        .groupby(
            "volatility_model",
            as_index=False
        )
        .agg(
            empirical_d_threshold=(
                "empirical_d_threshold",
                "first"
            ),

            holdout_long_memory_false_positive_rate=(
                "false_long_memory",
                "mean"
            ),

            holdout_mean_d_hat=(
                "d_hat",
                "mean"
            ),
        )
    )


def summarize_volatility_fp(results):
    return pd.DataFrame({
        "volatility_false_positive_rate": [
            results[
                "false_positive"
            ].mean()
        ]
    })


# ============================================================
# REPRESENTATIVE PLOTS
# ============================================================

def save_representative_plots():
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for volatility_model in VOLATILITY_MODELS:
        (
            innovations,
            _,
            df,
            info
        ) = generate_combination(
            volatility_model
        )

        true_d = float(
            info["d"]
        )

        fractional_full = fractionally_integrate(
            innovations,
            true_d
        )

        source_innovations = innovations[
            NUMSEAS:NUMSEAS + LENGTH
        ]

        fractional_component = fractional_full[
            NUMSEAS:NUMSEAS + LENGTH
        ]

        final_series = df[
            "data"
        ].to_numpy(dtype=float)

        time = np.arange(
            LENGTH
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 9),
            sharex=True
        )

        axes[0].plot(
            time,
            source_innovations,
            linewidth=1.0
        )

        axes[0].set_title(
            f"ARFIMA + {volatility_model.upper()} "
            f"— Component 1: Volatility innovations"
        )

        axes[0].set_ylabel("Value")
        axes[0].grid(alpha=0.3)

        axes[1].plot(
            time,
            fractional_component,
            linewidth=1.0
        )

        axes[1].set_title(
            f"Component 2: Fractionally integrated "
            f"innovations (d={true_d:.3f})"
        )

        axes[1].set_ylabel("Value")
        axes[1].grid(alpha=0.3)

        axes[2].plot(
            time,
            final_series,
            linewidth=1.0
        )

        axes[2].set_title(
            f"Combination: ARFIMA + "
            f"{volatility_model.upper()}"
        )

        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Value")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()

        filename = (
            f"arfima_{volatility_model}_components.png"
        )

        plt.savefig(
            PLOT_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved plot: {filename}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========================================")
    print("ARFIMA + VOLATILITY")
    print("JOINT STATISTICAL VALIDATION — V2")
    print("========================================")
    print(
        f"Length={LENGTH}, "
        f"numseas={NUMSEAS}"
    )
    print(
        f"evaluation trials/model={N_TRIALS}"
    )
    print(
        f"calibration trials/model="
        f"{N_CALIBRATION_TRIALS}"
    )
    print(
        f"holdout d=0 trials/model="
        f"{N_HOLDOUT_CONTROL_TRIALS}"
    )
    print(
        f"empirical threshold quantile="
        f"{EMPIRICAL_THRESHOLD_QUANTILE:.2f}"
    )
    print("========================================\n")

    # --------------------------------------------------------
    # 1. Calibrate model-specific long-memory thresholds.
    # --------------------------------------------------------

    calibration_results, thresholds = (
        calibrate_long_memory_thresholds()
    )

    # --------------------------------------------------------
    # 2. Verify thresholds on independent d=0 controls.
    # --------------------------------------------------------

    holdout_results = (
        run_long_memory_holdout_control(
            thresholds
        )
    )

    holdout_summary = (
        summarize_holdout_control(
            holdout_results
        )
    )

    # --------------------------------------------------------
    # 3. Evaluate genuine ARFIMA + volatility.
    # --------------------------------------------------------

    trial_results = run_validation(
        thresholds
    )

    joint_summary = summarize_validation(
        trial_results
    )

    # --------------------------------------------------------
    # 4. Independent volatility false-positive control.
    # --------------------------------------------------------

    volatility_fp_results = (
        run_volatility_fp_validation()
    )

    volatility_fp_summary = (
        summarize_volatility_fp(
            volatility_fp_results
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    calibration_results.to_csv(
        OUTPUT_DIR
        / "long_memory_calibration_trials.csv",
        index=False
    )

    thresholds.to_csv(
        OUTPUT_DIR
        / "long_memory_empirical_thresholds.csv",
        index=False
    )

    holdout_results.to_csv(
        OUTPUT_DIR
        / "long_memory_holdout_trials.csv",
        index=False
    )

    holdout_summary.to_csv(
        OUTPUT_DIR
        / "long_memory_holdout_control.csv",
        index=False
    )

    trial_results.to_csv(
        OUTPUT_DIR
        / "trial_results.csv",
        index=False
    )

    joint_summary.to_csv(
        OUTPUT_DIR
        / "joint_detection_rates.csv",
        index=False
    )

    volatility_fp_results.to_csv(
        OUTPUT_DIR
        / "volatility_false_positive_trials.csv",
        index=False
    )

    volatility_fp_summary.to_csv(
        OUTPUT_DIR
        / "volatility_false_positive_control.csv",
        index=False
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\n========================================")
    print("EMPIRICAL LONG-MEMORY THRESHOLDS")
    print("========================================")

    print(
        thresholds.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\n========================================")
    print("INDEPENDENT d=0 HOLDOUT CONTROL")
    print("========================================")

    print(
        holdout_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\n========================================")
    print("JOINT DETECTION RATES")
    print("========================================")

    print(
        joint_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\n========================================")
    print("VOLATILITY FALSE-POSITIVE CONTROL")
    print("========================================")

    print(
        volatility_fp_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    save_representative_plots()

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )
