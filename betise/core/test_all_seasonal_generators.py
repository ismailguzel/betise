
"""
Final seasonal-family validation.

Generates 15 series from each:
- generate_pure_sarma
- generate_pure_sarima
- generate_deterministic_sarma
- generate_deterministic_sarima
- generate_seasonal_unit_root_fourier

Length: random 500-2500.

Outputs under test_outputs/<model_name>/:
- plots/
- <model_name>_summary.csv

Also:
- test_outputs/all_seasonal_generators_summary.csv
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import periodogram, find_peaks
from statsmodels.tsa.stattools import acf

from .generator import TimeSeriesGenerator


SEED = 42
N_PER_MODEL = 15
MIN_LENGTH = 500
MAX_LENGTH = 2500

MIN_PERIOD = 2.0
MIN_CYCLES_FOR_SEARCH = 6
TOP_K_PERIODS = 5

ROOT_OUTPUT = Path("test_outputs")
ROOT_OUTPUT.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
rng = np.random.default_rng(SEED)


def safe_acf_values(series, period):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 3:
        return {
            "acf_s": np.nan,
            "acf_2s": np.nan,
            "acf_3s": np.nan,
        }

    max_lag = min(int(3 * period), len(x) - 1)

    values = acf(
        x,
        nlags=max_lag,
        fft=True
    )

    result = {}

    for multiplier in (1, 2, 3):
        lag = int(multiplier * period)
        key = "acf_s" if multiplier == 1 else f"acf_{multiplier}s"

        result[key] = (
            float(values[lag])
            if lag < len(values)
            else np.nan
        )

    return result


def calculate_acf_curve(series, period):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    max_lag = min(
        max(int(3 * period), 50),
        len(x) - 1,
        600
    )

    values = acf(
        x,
        nlags=max_lag,
        fft=True
    )

    return np.arange(len(values)), values


def periodogram_diagnostics(series, true_period, top_k=TOP_K_PERIODS):
    """
    Returns both:

    global_dominant_period:
        strongest non-zero periodogram peak with no period restriction.

    candidate_dominant_period:
        strongest local peak inside:
            2 <= period <= n / 6

    The second value is usually more useful for manual seasonal review,
    while the unrestricted value is kept so low-frequency domination is
    never hidden.
    """

    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    frequencies, powers = periodogram(
        x,
        detrend="linear"
    )

    positive = frequencies > 0
    f_pos = frequencies[positive]
    p_pos = powers[positive]

    empty = {
        "frequencies": frequencies,
        "powers": powers,
        "global_dominant_frequency": np.nan,
        "global_dominant_period": np.nan,
        "candidate_dominant_period": np.nan,
        "top_candidate_periods": [],
        "true_frequency": 1.0 / true_period,
        "nearest_true_frequency": np.nan,
        "true_period_power": np.nan,
        "true_frequency_percentile": np.nan,
        "candidate_relative_error": np.nan,
    }

    if len(f_pos) == 0:
        return empty

    global_idx = int(np.argmax(p_pos))
    global_frequency = float(f_pos[global_idx])
    global_period = float(1.0 / global_frequency)

    true_frequency = 1.0 / float(true_period)
    true_idx = int(np.argmin(np.abs(f_pos - true_frequency)))

    nearest_true_frequency = float(f_pos[true_idx])
    true_period_power = float(p_pos[true_idx])

    true_frequency_percentile = float(
        np.mean(p_pos <= true_period_power)
    )

    periods_pos = 1.0 / f_pos
    max_candidate_period = n / MIN_CYCLES_FOR_SEARCH

    candidate_mask = (
        (periods_pos >= MIN_PERIOD)
        &
        (periods_pos <= max_candidate_period)
    )

    p_candidate = p_pos[candidate_mask]
    periods_candidate = periods_pos[candidate_mask]

    top_periods = []
    candidate_dominant_period = np.nan

    if len(p_candidate) > 0:
        peak_indices, _ = find_peaks(p_candidate)

        if len(peak_indices) == 0:
            peak_indices = np.array([int(np.argmax(p_candidate))])

        ranked_peaks = peak_indices[
            np.argsort(p_candidate[peak_indices])[::-1]
        ][:top_k]

        top_periods = [
            float(periods_candidate[i])
            for i in ranked_peaks
        ]

        if top_periods:
            candidate_dominant_period = top_periods[0]

    if np.isfinite(candidate_dominant_period):
        candidate_relative_error = float(
            abs(candidate_dominant_period - true_period)
            / true_period
        )
    else:
        candidate_relative_error = np.nan

    return {
        "frequencies": frequencies,
        "powers": powers,
        "global_dominant_frequency": global_frequency,
        "global_dominant_period": global_period,
        "candidate_dominant_period": candidate_dominant_period,
        "top_candidate_periods": top_periods,
        "true_frequency": true_frequency,
        "nearest_true_frequency": nearest_true_frequency,
        "true_period_power": true_period_power,
        "true_frequency_percentile": true_frequency_percentile,
        "candidate_relative_error": candidate_relative_error,
    }


def stringify_periods(periods):
    return "; ".join(
        f"{p:.4f}"
        for p in periods
    )


def get_analysis_series(model_name, df, info):
    """
    Raw metrics are always calculated separately.

    Analysis representation:
    - pure_sarma: raw
    - deterministic_sarma: raw
    - pure_sarima:
        d=0,D=1 -> raw
        d=1,D=1 -> first difference
    - deterministic_sarima:
        d=1 -> first difference
    - seasonal_unit_root_fourier:
        seasonal difference Y_t - Y_{t-s}
    """

    raw = df["data"].to_numpy(dtype=float)

    if model_name == "pure_sarima":
        d = int(info.get("diff", 0))

        if d == 1:
            return np.diff(raw), "first_difference"

        return raw.copy(), "raw"

    if model_name == "deterministic_sarima":
        d = int(info.get("diff", 0))

        if d == 1:
            return np.diff(raw), "first_difference"

        return raw.copy(), "raw"

    if model_name == "seasonal_unit_root_fourier":
        period = int(info["periods"][0])

        if "seasonal_difference" in df.columns:
            values = df["seasonal_difference"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            return values, "seasonal_difference"

        return (
            raw[period:] - raw[:-period],
            "seasonal_difference"
        )

    return raw.copy(), "raw"


def add_acf_plot(ax, series, true_period, title):
    lags, values = calculate_acf_curve(
        series,
        true_period
    )

    ax.vlines(
        lags,
        0,
        values
    )

    ax.axhline(
        0,
        linewidth=1
    )

    for multiplier in (1, 2, 3):
        seasonal_lag = multiplier * true_period

        if seasonal_lag <= lags[-1]:
            ax.axvline(
                seasonal_lag,
                linestyle="--",
                linewidth=1.2
            )

    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")


def add_periodogram_plot(
    ax,
    series,
    true_period,
    diagnostics,
    title
):
    """
    Shows periodogram power in period-domain for easier visual review.
    """

    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    frequencies = diagnostics["frequencies"]
    powers = diagnostics["powers"]

    positive = frequencies > 0
    f_pos = frequencies[positive]
    p_pos = powers[positive]

    periods = 1.0 / f_pos
    max_candidate_period = len(x) / MIN_CYCLES_FOR_SEARCH

    display_mask = (
        (periods >= MIN_PERIOD)
        &
        (periods <= max_candidate_period)
    )

    display_periods = periods[display_mask]
    display_powers = p_pos[display_mask]

    order = np.argsort(display_periods)

    if len(order) > 0:
        ax.plot(
            display_periods[order],
            display_powers[order]
        )

    ax.axvline(
        true_period,
        linestyle="--",
        linewidth=1.5,
        label=f"True period = {true_period}"
    )

    detected = diagnostics["candidate_dominant_period"]

    if np.isfinite(detected):
        ax.axvline(
            detected,
            linestyle=":",
            linewidth=1.5,
            label=f"Detected = {detected:.2f}"
        )

    ax.set_title(title)
    ax.set_xlabel("Period")
    ax.set_ylabel("Periodogram power")
    ax.legend()


def plot_diagnostics(
    model_name,
    series_id,
    raw_series,
    analysis_series,
    analysis_type,
    true_period,
    info,
    raw_pg,
    analysis_pg,
    output_path
):
    raw_series = np.asarray(raw_series, dtype=float)
    analysis_series = np.asarray(analysis_series, dtype=float)

    same_representation = (
        analysis_type == "raw"
        and len(raw_series) == len(analysis_series)
    )

    d = info.get("diff", 0)
    D = info.get("seasonal_diff", 0)
    p = info.get("ar_order", 0)
    q = info.get("ma_order", 0)
    P = info.get("seasonal_ar_order", 0)
    Q = info.get("seasonal_ma_order", 0)

    header = (
        f"{model_name} | Series {series_id:02d} | "
        f"n={len(raw_series)} | true s={true_period} | "
        f"d={d}, D={D} | p={p}, q={q}, P={P}, Q={Q}"
    )

    if same_representation:
        fig = plt.figure(
            figsize=(14, 12)
        )

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(
            np.arange(len(raw_series)),
            raw_series
        )
        ax1.set_title(header)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")

        ax2 = fig.add_subplot(3, 1, 2)
        add_acf_plot(
            ax2,
            raw_series,
            true_period,
            "ACF (dashed = s, 2s, 3s)"
        )

        ax3 = fig.add_subplot(3, 1, 3)
        add_periodogram_plot(
            ax3,
            raw_series,
            true_period,
            raw_pg,
            (
                "Periodogram | "
                f"global={raw_pg['global_dominant_period']:.2f} | "
                f"candidate={raw_pg['candidate_dominant_period']:.2f}"
            )
        )

    else:
        fig = plt.figure(
            figsize=(16, 18)
        )

        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(
            np.arange(len(raw_series)),
            raw_series
        )
        ax1.set_title(header + " | RAW")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")

        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(
            np.arange(len(analysis_series)),
            analysis_series
        )
        ax2.set_title(
            f"Analysis: {analysis_type}"
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Value")

        ax3 = fig.add_subplot(3, 2, 3)
        add_acf_plot(
            ax3,
            raw_series,
            true_period,
            "RAW ACF"
        )

        ax4 = fig.add_subplot(3, 2, 4)
        add_acf_plot(
            ax4,
            analysis_series,
            true_period,
            f"{analysis_type} ACF"
        )

        ax5 = fig.add_subplot(3, 2, 5)
        add_periodogram_plot(
            ax5,
            raw_series,
            true_period,
            raw_pg,
            (
                "RAW periodogram | "
                f"global={raw_pg['global_dominant_period']:.2f} | "
                f"candidate={raw_pg['candidate_dominant_period']:.2f}"
            )
        )

        ax6 = fig.add_subplot(3, 2, 6)
        add_periodogram_plot(
            ax6,
            analysis_series,
            true_period,
            analysis_pg,
            (
                f"{analysis_type} periodogram | "
                f"global={analysis_pg['global_dominant_period']:.2f} | "
                f"candidate={analysis_pg['candidate_dominant_period']:.2f}"
            )
        )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def generate_one(
    model_name,
    generator,
    pure_sarima_d
):
    if model_name == "pure_sarma":
        return generator.generate_pure_sarma()

    if model_name == "pure_sarima":
        return generator.generate_pure_sarima(
            d=pure_sarima_d,
            D=1
        )

    if model_name == "deterministic_sarma":
        return generator.generate_deterministic_sarma()

    if model_name == "deterministic_sarima":
        return generator.generate_deterministic_sarima(
            d=1
        )

    if model_name == "seasonal_unit_root_fourier":
        return generator.generate_seasonal_unit_root_fourier()

    raise ValueError(
        f"Unknown model: {model_name}"
    )


MODEL_NAMES = [
    "pure_sarma",
    "pure_sarima",
    "deterministic_sarma",
    "deterministic_sarima",
    "seasonal_unit_root_fourier",
]


# 15 pure SARIMA samples:
# 8 x d=0,D=1
# 7 x d=1,D=1
pure_sarima_d_values = (
    [0] * 8
    +
    [1] * 7
)

rng.shuffle(
    pure_sarima_d_values
)


all_rows = []


for model_name in MODEL_NAMES:

    print("\n" + "=" * 68)
    print(f"MODEL: {model_name}")
    print("=" * 68)

    model_dir = ROOT_OUTPUT / model_name
    plot_dir = model_dir / "plots"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_rows = []

    for sample_index in range(
        1,
        N_PER_MODEL + 1
    ):

        length = int(
            rng.integers(
                MIN_LENGTH,
                MAX_LENGTH + 1
            )
        )

        if model_name == "pure_sarima":
            sarima_d = int(
                pure_sarima_d_values[
                    sample_index - 1
                ]
            )
        else:
            sarima_d = 0

        print(
            f"[{sample_index:02d}/{N_PER_MODEL}] "
            f"length={length}"
        )

        generator = TimeSeriesGenerator(
            length=length
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            df, info = generate_one(
                model_name=model_name,
                generator=generator,
                pure_sarima_d=sarima_d
            )

        raw_series = (
            df["data"]
            .to_numpy(
                dtype=float
            )
        )

        true_period = int(
            info["periods"][0]
        )

        analysis_series, analysis_type = (
            get_analysis_series(
                model_name=model_name,
                df=df,
                info=info
            )
        )

        raw_acf = safe_acf_values(
            raw_series,
            true_period
        )

        raw_pg = periodogram_diagnostics(
            raw_series,
            true_period
        )

        analysis_acf = safe_acf_values(
            analysis_series,
            true_period
        )

        analysis_pg = periodogram_diagnostics(
            analysis_series,
            true_period
        )

        plot_filename = (
            f"{model_name}_"
            f"{sample_index:02d}_"
            f"n{length}_"
            f"s{true_period}.png"
        )

        plot_path = (
            plot_dir
            / plot_filename
        )

        plot_diagnostics(
            model_name=model_name,
            series_id=sample_index,
            raw_series=raw_series,
            analysis_series=analysis_series,
            analysis_type=analysis_type,
            true_period=true_period,
            info=info,
            raw_pg=raw_pg,
            analysis_pg=analysis_pg,
            output_path=plot_path
        )

        row = {
            "model":
                model_name,

            "series_id":
                sample_index,

            "length":
                length,

            "true_period":
                true_period,

            "analysis_type":
                analysis_type,

            "d":
                info.get("diff", 0),

            "D":
                info.get("seasonal_diff", 0),

            "p":
                info.get("ar_order", 0),

            "q":
                info.get("ma_order", 0),

            "P":
                info.get("seasonal_ar_order", 0),

            "Q":
                info.get("seasonal_ma_order", 0),

            "seasonality_source":
                info.get("seasonality_source", ""),

            "stochastic_background":
                info.get("stochastic_background", ""),

            # RAW ACF
            "raw_acf_s":
                raw_acf["acf_s"],

            "raw_acf_2s":
                raw_acf["acf_2s"],

            "raw_acf_3s":
                raw_acf["acf_3s"],

            # RAW PERIODOGRAM
            "raw_global_dominant_period":
                raw_pg["global_dominant_period"],

            "raw_candidate_dominant_period":
                raw_pg["candidate_dominant_period"],

            "raw_top5_candidate_periods":
                stringify_periods(
                    raw_pg["top_candidate_periods"]
                ),

            "raw_true_frequency_percentile":
                raw_pg["true_frequency_percentile"],

            "raw_candidate_relative_error":
                raw_pg["candidate_relative_error"],

            # ANALYSIS ACF
            "analysis_acf_s":
                analysis_acf["acf_s"],

            "analysis_acf_2s":
                analysis_acf["acf_2s"],

            "analysis_acf_3s":
                analysis_acf["acf_3s"],

            # ANALYSIS PERIODOGRAM
            "analysis_global_dominant_period":
                analysis_pg["global_dominant_period"],

            "analysis_candidate_dominant_period":
                analysis_pg["candidate_dominant_period"],

            "analysis_top5_candidate_periods":
                stringify_periods(
                    analysis_pg["top_candidate_periods"]
                ),

            "analysis_true_frequency_percentile":
                analysis_pg["true_frequency_percentile"],

            "analysis_candidate_relative_error":
                analysis_pg["candidate_relative_error"],

            # EASY TRUE-vs-DETECTED COMPARISON
            "true_minus_raw_candidate":
                (
                    true_period
                    - raw_pg["candidate_dominant_period"]
                    if np.isfinite(
                        raw_pg["candidate_dominant_period"]
                    )
                    else np.nan
                ),

            "true_minus_analysis_candidate":
                (
                    true_period
                    - analysis_pg[
                        "candidate_dominant_period"
                    ]
                    if np.isfinite(
                        analysis_pg[
                            "candidate_dominant_period"
                        ]
                    )
                    else np.nan
                ),

            "plot_file":
                str(plot_path),
        }

        model_rows.append(row)
        all_rows.append(row)

        print(
            "    "
            f"true={true_period} | "
            f"raw candidate="
            f"{raw_pg['candidate_dominant_period']:.2f} | "
            f"analysis candidate="
            f"{analysis_pg['candidate_dominant_period']:.2f} | "
            f"analysis={analysis_type}"
        )

    model_df = pd.DataFrame(
        model_rows
    )

    model_csv = (
        model_dir
        / f"{model_name}_summary.csv"
    )

    model_df.to_csv(
        model_csv,
        index=False
    )

    print(
        f"\nSaved CSV: {model_csv}"
    )

    print(
        f"Saved plots: {plot_dir}"
    )


combined_df = pd.DataFrame(
    all_rows
)

combined_csv = (
    ROOT_OUTPUT
    / "all_seasonal_generators_summary.csv"
)

combined_df.to_csv(
    combined_csv,
    index=False
)


print("\n" + "=" * 68)
print("ALL SEASONAL GENERATOR TESTS COMPLETE")
print("=" * 68)

print(
    f"\nCombined summary:\n{combined_csv}"
)

print(
    "\nTRUE PERIOD vs DETECTED ANALYSIS PERIOD"
)

overview_columns = [
    "model",
    "series_id",
    "length",
    "true_period",
    "analysis_type",
    "analysis_candidate_dominant_period",
    "analysis_top5_candidate_periods",
    "analysis_acf_s",
    "analysis_true_frequency_percentile",
]

print(
    combined_df[
        overview_columns
    ].to_string(
        index=False
    )
)
