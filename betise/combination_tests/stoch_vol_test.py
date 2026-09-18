"""
FINAL STATISTICAL VALIDATION
Stochastic + Volatility

Tested combinations:

    Random Walk
    Random Walk with Drift
    ARI
    IMA
    ARIMA

    x

    ARCH
    GARCH
    EGARCH
    APARCH


Mathematical idea
-----------------

RW:
    ΔX_t = epsilon_t

RWD:
    ΔX_t = drift + epsilon_t

ARI / IMA / ARIMA:

    volatility innovations
            ↓
       AR / MA / ARMA
            ↓
        integration
            ↓
       stochastic series


Validation idea
---------------

We test volatility:

1. directly in the original SOURCE innovations;

2. after generating the full stochastic process,
   differencing the final series and inverse-filtering
   AR/MA/ARMA dynamics to obtain RECOVERED innovations.

If source and recovered detection rates are nearly equal,
the stochastic/integration process did not destroy the
volatility information.

Gaussian innovations are used as a negative control.

Saved outputs
-------------
test_outputs/stochastic_volatility/
    detection_rates.csv
    gaussian_control.csv
    plots/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import lfilter
from statsmodels.stats.diagnostic import (
    het_arch,
    acorr_ljungbox,
)

from betise.core.generator import TimeSeriesGenerator


# =========================================================
# SETTINGS
# =========================================================

STOCHASTIC_TYPES = [
    "rw",
    "rwd",
    "ari",
    "ima",
    "arima",
]

VOLATILITY_TYPES = [
    "arch",
    "garch",
    "egarch",
    "aparch",
]

LENGTH = 400
N_TRIALS = 50

TEST_LAG = 10
ALPHA = 0.05

WARMUP = 50


OUTPUT_DIR = Path(
    "betise/combination_tests/test_outputs/stochastic_volatility"
)

PLOT_DIR = OUTPUT_DIR / "plots"


# =========================================================
# RECOVER PRE-INTEGRATION PROCESS
# =========================================================

def recover_stationary_process(
    series,
    stochastic_kind,
    info,
):
    """
    Undo integration.

    RW:
        first difference

    RWD:
        first difference - drift

    ARI / IMA / ARIMA:
        difference d times

    The result is the stationary process that existed
    immediately before integration.
    """

    series = np.asarray(
        series,
        dtype=float
    )

    if stochastic_kind == "rw":

        return np.diff(
            series
        )

    if stochastic_kind == "rwd":

        drift = float(
            info["drift"]
        )

        return (
            np.diff(series)
            - drift
        )

    d = int(
        info["diff"]
    )

    return np.diff(
        series,
        n=d
    )


# =========================================================
# RECOVER VOLATILITY INNOVATIONS
# =========================================================

def recover_innovations(
    stationary_process,
    stochastic_kind,
    info,
):
    """
    RW/RWD:
        differenced series already equals innovations.

    ARI/IMA/ARIMA:
        inverse-filter the AR / MA / ARMA process
        to recover the volatility innovations.
    """

    y = np.asarray(
        stationary_process,
        dtype=float
    )

    if stochastic_kind in {
        "rw",
        "rwd",
    }:

        return y

    if stochastic_kind == "ari":

        ar_coefs = np.asarray(
            info["ar_coefs"],
            dtype=float
        )

        ar = np.r_[
            1.0,
            -ar_coefs
        ]

        ma = np.array([
            1.0
        ])

        return lfilter(
            ar,
            ma,
            y
        )

    if stochastic_kind == "ima":

        ma_coefs = np.asarray(
            info["ma_coefs"],
            dtype=float
        )

        ar = np.array([
            1.0
        ])

        ma = np.r_[
            1.0,
            ma_coefs
        ]

        return lfilter(
            ar,
            ma,
            y
        )

    if stochastic_kind == "arima":

        ar_coefs = np.asarray(
            info["ar_coefs"],
            dtype=float
        )

        ma_coefs = np.asarray(
            info["ma_coefs"],
            dtype=float
        )

        ar = np.r_[
            1.0,
            -ar_coefs
        ]

        ma = np.r_[
            1.0,
            ma_coefs
        ]

        return lfilter(
            ar,
            ma,
            y
        )

    raise ValueError(
        f"Unknown stochastic type: "
        f"{stochastic_kind}"
    )


# =========================================================
# VOLATILITY TESTS
# =========================================================

def volatility_tests(
    residuals,
):
    """
    ARCH-LM and squared Ljung-Box are used to detect
    volatility clustering.

    Small p-values indicate that squared innovations are
    not behaving like independent constant-variance noise.

    any_detected=True if either test detects volatility.
    """

    x = np.asarray(
        residuals,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    x = (
        x
        - np.mean(x)
    )

    arch_result = het_arch(
        x,
        nlags=TEST_LAG
    )

    arch_p = float(
        arch_result[1]
    )

    lb_result = acorr_ljungbox(
        x ** 2,
        lags=[TEST_LAG],
        return_df=True
    )

    lb_p = float(
        lb_result[
            "lb_pvalue"
        ].iloc[0]
    )

    arch_detected = (
        arch_p < ALPHA
    )

    lb_detected = (
        lb_p < ALPHA
    )

    return {
        "arch_detected":
            arch_detected,

        "lb_detected":
            lb_detected,

        "any_detected":
            (
                arch_detected
                or lb_detected
            ),
    }


# =========================================================
# GENERATE STOCHASTIC SERIES
# =========================================================

def generate_stochastic_from_innovations(
    stochastic_kind,
    innovations,
):

    ts = TimeSeriesGenerator(
        length=LENGTH
    )

    if stochastic_kind == "rw":

        df, info = (
            ts.generate_stochastic_trend(
                kind="rw",
                innovations=innovations
            )
        )

        d_used = 1

    elif stochastic_kind == "rwd":

        df, info = (
            ts.generate_stochastic_trend(
                kind="rwd",
                drift=0.05,
                innovations=innovations
            )
        )

        d_used = 1

    else:

        # Test both supported integration orders.
        d_used = int(
            np.random.choice(
                [1, 2]
            )
        )

        df, info = (
            ts.generate_stochastic_trend(
                kind=stochastic_kind,
                d=d_used,
                const=False,
                innovations=innovations
            )
        )

    return (
        df,
        info,
        d_used
    )


# =========================================================
# SINGLE VOLATILITY TRIAL
# =========================================================

def run_volatility_trial(
    stochastic_kind,
    volatility_kind,
):

    ts = TimeSeriesGenerator(
        length=LENGTH
    )

    innovations, _ = (
        ts.generate_volatility(
            kind=volatility_kind,
            as_innovations=True
        )
    )

    innovations = np.asarray(
        innovations,
        dtype=float
    )

    df, info, d_used = (
        generate_stochastic_from_innovations(
            stochastic_kind,
            innovations
        )
    )

    series = df[
        "data"
    ].to_numpy(
        dtype=float
    )

    stationary_process = (
        recover_stationary_process(
            series,
            stochastic_kind,
            info
        )
    )

    recovered = recover_innovations(
        stationary_process,
        stochastic_kind,
        info
    )

    recovered = recovered[
        WARMUP:
    ]

    source = innovations[
        -len(recovered):
    ]

    source_test = volatility_tests(
        source
    )

    recovered_test = volatility_tests(
        recovered
    )

    return {
        "d":
            d_used,

        "source_ARCH":
            source_test[
                "arch_detected"
            ],

        "source_LB2":
            source_test[
                "lb_detected"
            ],

        "source_any":
            source_test[
                "any_detected"
            ],

        "recovered_ARCH":
            recovered_test[
                "arch_detected"
            ],

        "recovered_LB2":
            recovered_test[
                "lb_detected"
            ],

        "recovered_any":
            recovered_test[
                "any_detected"
            ],
    }


# =========================================================
# GAUSSIAN CONTROL
# =========================================================

def run_gaussian_control(
    stochastic_kind,
):
    """
    Negative control.

    The same stochastic pipeline is generated using
    ordinary Gaussian innovations.

    False-positive rates tell us how often the statistical
    tests incorrectly report volatility where none was
    deliberately generated.
    """

    innovations = np.random.normal(
        0,
        1,
        LENGTH
    )

    df, info, _ = (
        generate_stochastic_from_innovations(
            stochastic_kind,
            innovations
        )
    )

    series = df[
        "data"
    ].to_numpy(
        dtype=float
    )

    stationary_process = (
        recover_stationary_process(
            series,
            stochastic_kind,
            info
        )
    )

    recovered = recover_innovations(
        stationary_process,
        stochastic_kind,
        info
    )

    recovered = recovered[
        WARMUP:
    ]

    return volatility_tests(
        recovered
    )


# =========================================================
# REPEATED VALIDATION
# =========================================================

def run_validation():

    rows = []

    print(
        "\n========================================"
    )
    print(
        "STOCHASTIC + VOLATILITY"
    )
    print(
        "FINAL STATISTICAL VALIDATION"
    )
    print(
        "========================================\n"
    )

    for stochastic_kind in STOCHASTIC_TYPES:

        for volatility_kind in VOLATILITY_TYPES:

            print(
                f"Running "
                f"{stochastic_kind.upper()} + "
                f"{volatility_kind.upper()} ..."
            )

            source_arch = 0
            source_lb = 0
            source_any = 0

            recovered_arch = 0
            recovered_lb = 0
            recovered_any = 0

            d1_count = 0
            d2_count = 0

            for _ in range(
                N_TRIALS
            ):

                result = run_volatility_trial(
                    stochastic_kind,
                    volatility_kind
                )

                if result["d"] == 1:
                    d1_count += 1

                elif result["d"] == 2:
                    d2_count += 1

                source_arch += int(
                    result[
                        "source_ARCH"
                    ]
                )

                source_lb += int(
                    result[
                        "source_LB2"
                    ]
                )

                source_any += int(
                    result[
                        "source_any"
                    ]
                )

                recovered_arch += int(
                    result[
                        "recovered_ARCH"
                    ]
                )

                recovered_lb += int(
                    result[
                        "recovered_LB2"
                    ]
                )

                recovered_any += int(
                    result[
                        "recovered_any"
                    ]
                )

            rows.append({
                "stochastic":
                    stochastic_kind,

                "volatility":
                    volatility_kind,

                "source_ARCH":
                    source_arch / N_TRIALS,

                "source_LB2":
                    source_lb / N_TRIALS,

                "source_any":
                    source_any / N_TRIALS,

                "recovered_ARCH":
                    recovered_arch / N_TRIALS,

                "recovered_LB2":
                    recovered_lb / N_TRIALS,

                "recovered_any":
                    recovered_any / N_TRIALS,

                "d1_trials":
                    d1_count,

                "d2_trials":
                    d2_count,
            })

    return pd.DataFrame(
        rows
    )


# =========================================================
# GAUSSIAN VALIDATION
# =========================================================

def run_gaussian_validation():

    rows = []

    for stochastic_kind in STOCHASTIC_TYPES:

        print(
            f"Running Gaussian "
            f"{stochastic_kind.upper()} ..."
        )

        arch_count = 0
        lb_count = 0
        any_count = 0

        for _ in range(
            N_TRIALS
        ):

            result = run_gaussian_control(
                stochastic_kind
            )

            arch_count += int(
                result[
                    "arch_detected"
                ]
            )

            lb_count += int(
                result[
                    "lb_detected"
                ]
            )

            any_count += int(
                result[
                    "any_detected"
                ]
            )

        rows.append({
            "stochastic":
                stochastic_kind,

            "ARCH_false_positive":
                arch_count / N_TRIALS,

            "LB2_false_positive":
                lb_count / N_TRIALS,

            "ANY_false_positive":
                any_count / N_TRIALS,
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# REPRESENTATIVE PLOTS
# =========================================================

def save_representative_plots():
    """
    One representative 3-panel plot for each combination.

    Component 1:
        Standalone stochastic process.

    Component 2:
        Volatility innovations.

    Combination:
        Stochastic process driven by those volatility innovations.

    NOTE:
        This is NOT an additive combination.
        Volatility is used as the innovation process.
    """

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for stochastic_kind in STOCHASTIC_TYPES:
        for volatility_kind in VOLATILITY_TYPES:

            # -------------------------------------------------
            # Choose integration order
            # -------------------------------------------------
            if stochastic_kind in {"rw", "rwd"}:
                d_used = 1
            else:
                d_used = int(np.random.choice([1, 2]))

            # -------------------------------------------------
            # COMPONENT 1 — standalone stochastic
            # -------------------------------------------------
            ts_base = TimeSeriesGenerator(length=LENGTH)

            if stochastic_kind == "rw":
                stochastic_df, _ = ts_base.generate_stochastic_trend(
                    kind="rw"
                )

            elif stochastic_kind == "rwd":
                stochastic_df, _ = ts_base.generate_stochastic_trend(
                    kind="rwd",
                    drift=0.05
                )

            else:
                stochastic_df, _ = ts_base.generate_stochastic_trend(
                    kind=stochastic_kind,
                    d=d_used,
                    const=False
                )

            stochastic_component = stochastic_df["data"].to_numpy(dtype=float)

            # -------------------------------------------------
            # COMPONENT 2 — volatility innovations
            # -------------------------------------------------
            ts_combined = TimeSeriesGenerator(length=LENGTH)

            innovations, _ = ts_combined.generate_volatility(
                kind=volatility_kind,
                as_innovations=True
            )

            innovations = np.asarray(innovations, dtype=float)

            # -------------------------------------------------
            # COMBINATION
            # -------------------------------------------------
            if stochastic_kind == "rw":
                combined_df, _ = ts_combined.generate_stochastic_trend(
                    kind="rw",
                    innovations=innovations
                )

            elif stochastic_kind == "rwd":
                combined_df, _ = ts_combined.generate_stochastic_trend(
                    kind="rwd",
                    drift=0.05,
                    innovations=innovations
                )

            else:
                combined_df, _ = ts_combined.generate_stochastic_trend(
                    kind=stochastic_kind,
                    d=d_used,
                    const=False,
                    innovations=innovations
                )

            combined = combined_df["data"].to_numpy(dtype=float)
            time = np.arange(LENGTH)

            # -------------------------------------------------
            # PLOT
            # -------------------------------------------------
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

            axes[0].plot(time, stochastic_component, linewidth=1.1)
            axes[0].set_title(
                f"{stochastic_kind.upper()} + {volatility_kind.upper()} "
                "— Component 1: Stochastic"
            )
            axes[0].set_ylabel("Value")
            axes[0].grid(alpha=0.3)

            axes[1].plot(time, innovations, linewidth=1.1)
            axes[1].set_title(
                f"{stochastic_kind.upper()} + {volatility_kind.upper()} "
                "— Component 2: Volatility Innovations"
            )
            axes[1].set_ylabel("Innovation")
            axes[1].grid(alpha=0.3)

            axes[2].plot(time, combined, linewidth=1.1)
            axes[2].set_title(
                f"{stochastic_kind.upper()} + {volatility_kind.upper()} "
                f"— Combination (d={d_used})"
            )
            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Value")
            axes[2].grid(alpha=0.3)

            plt.tight_layout()

            filename = (
                f"{stochastic_kind}_{volatility_kind}_components.png"
            )

            plt.savefig(
                PLOT_DIR / filename,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(f"Saved plot: {filename}")



# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detection_results = (
        run_validation()
    )

    gaussian_results = (
        run_gaussian_validation()
    )

    detection_results.to_csv(
        OUTPUT_DIR
        / "detection_rates.csv",
        index=False
    )

    gaussian_results.to_csv(
        OUTPUT_DIR
        / "gaussian_control.csv",
        index=False
    )

    print(
        "\n========================================"
    )
    print(
        "VOLATILITY DETECTION RATES"
    )
    print(
        "========================================"
    )

    print(
        detection_results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}"
        )
    )

    print(
        "\n========================================"
    )
    print(
        "GAUSSIAN FALSE-POSITIVE RATES"
    )
    print(
        "========================================"
    )

    print(
        gaussian_results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}"
        )
    )

    save_representative_plots()

    print(
        "\n========================================"
    )
    print(
        "INTERPRETATION"
    )
    print(
        "========================================"
    )

    print(
        "source_any      : volatility detection "
        "in the original innovations"
    )

    print(
        "recovered_any   : volatility detection "
        "after the complete stochastic pipeline"
    )

    print(
        "Similar values mean integration and "
        "AR/MA/ARMA dynamics preserved volatility."
    )

    print(
        "Gaussian false-positive rates show how "
        "often ordinary constant-variance noise "
        "is incorrectly classified as volatile."
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )