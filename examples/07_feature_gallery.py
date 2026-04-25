"""
BeTiSe Feature Gallery
=======================

Generates a two-section PDF for visual inspection:

  Section 1 — Base Series (15 plots)
      One clean series for each base type, no features applied.
      Bases: white_noise, ar, ma, arma,
             random_walk, random_walk_drift, ari, ima, arima,
             sarma, sarima,
             arch, garch, egarch, aparch

  Section 2 — Feature Showcase (12 plots)
      Each feature applied individually on an AR base so the effect
      is clearly visible without interference from another feature.
      Features: linear_trend, quadratic_trend, cubic_trend,
                exponential_trend, single_seasonality,
                multiple_seasonality, point_anomaly, collective_anomaly,
                contextual_anomaly, mean_shift, variance_shift, trend_shift

Output:
  generated-dataset/visual_report/feature_gallery.pdf

Run:
    python examples/08_feature_gallery.py
"""

import copy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from betise import dataset_generation as dg
from betise.config import load_config

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("generated-dataset") / "visual_report"
OUTPUT_PDF = OUTPUT_DIR / "feature_gallery.pdf"
SEED       = 42
LENGTH     = 700

# All 15 base series in display order
BASE_SERIES = [
    # Stationary
    ("White Noise",        "white_noise"),
    ("AR",                 "ar"),
    ("MA",                 "ma"),
    ("ARMA",               "arma"),
    # Stochastic
    ("Random Walk",        "random_walk"),
    ("Random Walk + Drift","random_walk_drift"),
    ("ARI",                "ari"),
    ("IMA",                "ima"),
    ("ARIMA",              "arima"),
    # Seasonal
    ("SARMA",              "sarma"),
    ("SARIMA",             "sarima"),
    # Volatility
    ("ARCH",               "arch"),
    ("GARCH",              "garch"),
    ("EGARCH",             "egarch"),
    ("APARCH",             "aparch"),
]

# All 12 features — (display name, feature key, base to use)
# Anomaly features use white_noise base: flatter signal makes spikes stand out.
# Trend/seasonality/break features use ar base.
FEATURES = [
    ("Trend: Linear",                "linear_trend",        "ar",          {}),
    ("Trend: Quadratic",             "quadratic_trend",     "ar",          {}),
    ("Trend: Cubic",                 "cubic_trend",         "ar",          {}),
    ("Trend: Exponential",           "exponential_trend",   "ar",          {}),
    ("Seasonality: Single",          "single_seasonality",  "ar",          {}),
    ("Seasonality: Multiple",        "multiple_seasonality","ar",          {}),
    ("Anomaly: Point",               "point_anomaly",       "ar",          {"is_spike": True, "scale_factor": 1.2}),
    ("Anomaly: Collective",          "collective_anomaly",  "white_noise", {}),
    ("Anomaly: Contextual",          "contextual_anomaly",  "white_noise", {}),
    ("Break: Mean Shift",            "mean_shift",          "ar",          {}),
    ("Break: Variance Shift",        "variance_shift",      "ar",          {}),
    ("Break: Trend Shift",           "trend_shift",         "ar",          {}),
]

# Anomaly index columns in the output dataframe
ANOMALY_INDEX_COL  = "anomaly_indices"
BREAK_INDEX_COL    = "break_indices"

# ── Helpers ───────────────────────────────────────────────────────────────────

ALL_FEATURES_OFF = {
    "linear_trend":         {"enabled": False},
    "quadratic_trend":      {"enabled": False},
    "cubic_trend":          {"enabled": False},
    "exponential_trend":    {"enabled": False},
    "single_seasonality":   {"enabled": False},
    "multiple_seasonality": {"enabled": False},
    "sarma":                {"enabled": False},
    "sarima":               {"enabled": False},
    "arch":                 {"enabled": False},
    "garch":                {"enabled": False},
    "egarch":               {"enabled": False},
    "aparch":               {"enabled": False},
    "mean_shift":           {"enabled": False},
    "variance_shift":       {"enabled": False},
    "trend_shift":          {"enabled": False},
    "point_anomaly":        {"enabled": False},
    "collective_anomaly":   {"enabled": False},
    "contextual_anomaly":   {"enabled": False},
}


def _parse_flat_indices(val):
    """Return a flat list of int indices from various storage formats."""
    import ast, re
    if val is None:
        return []
    if isinstance(val, float) and __import__("pandas").isna(val):
        return []
    if isinstance(val, (list, tuple)):
        flat = []
        for v in val:
            if isinstance(v, (list, tuple)):
                flat.extend(int(x) for x in v)
            else:
                flat.append(int(v))
        return flat
    s = str(val).strip()
    s = re.sub(r"(?:np|numpy)\.int\d+\((-?\d+)\)", r"\1", s)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            flat = []
            for v in obj:
                if isinstance(v, (list, tuple)):
                    flat.extend(int(x) for x in v)
                else:
                    flat.append(int(v))
            return flat
        return [int(obj)]
    except Exception:
        nums = re.findall(r"\b-?\d+\b", s)
        return [int(n) for n in nums]


def _make_cfg(base_series, features_override, seed):
    """Build a minimal config for one series."""
    feats = copy.deepcopy(ALL_FEATURES_OFF)
    feats.update(features_override)
    return load_config(dataset={
        "base_series":  base_series,
        "num_series":   1,
        "length_range": [LENGTH, LENGTH],
        "random_seed":  seed,
        "output_dir":   str(OUTPUT_DIR),
        "output_name":  "gallery_tmp.parquet",
        "features":     feats,
    })


def _plot(ax, df, title, color="steelblue", feat_key=None):
    series = df.sort_values("time").reset_index(drop=True)
    ax.plot(series["time"], series["data"], linewidth=1.4, color=color)

    # ── Annotate anomaly / break positions ────────────────────────────────
    if feat_key in ("point_anomaly", "collective_anomaly", "contextual_anomaly"):
        if ANOMALY_INDEX_COL in series.columns:
            raw = series[ANOMALY_INDEX_COL].iloc[0]
            indices = _parse_flat_indices(raw)
            for idx in indices:
                if 0 <= idx < len(series):
                    t_val = series["time"].iloc[idx]
                    y_val = series["data"].iloc[idx]
                    # Red dot on the anomaly point
                    ax.scatter([t_val], [y_val], color="red", zorder=5,
                               s=60, label="_nolegend_")
                    # Arrow annotation from above/below
                    y_range   = series["data"].max() - series["data"].min()
                    arrow_dir = -0.18 * y_range if y_val > series["data"].median() else 0.18 * y_range
                    ax.annotate("",
                        xy=(t_val, y_val),
                        xytext=(t_val, y_val + arrow_dir),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.8))

    elif feat_key in ("mean_shift", "variance_shift", "trend_shift"):
        if BREAK_INDEX_COL in series.columns:
            raw = series[BREAK_INDEX_COL].iloc[0]
            indices = _parse_flat_indices(raw)
            for idx in indices:
                if 0 <= idx < len(series):
                    ax.axvline(x=series["time"].iloc[idx],
                               color="red", linewidth=1.8,
                               linestyle="--", alpha=0.9)
                    ax.text(series["time"].iloc[idx] + 3,
                            ax.get_ylim()[1] * 0.88,
                            "break", color="red", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(alpha=0.25)
    data_range = series["data"].max() - series["data"].min()
    if data_range > 100:
        ax.set_yscale("symlog")
    else:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))


def _section_title_page(pdf, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    ax.text(0.5, 0.6, title,   ha="center", va="center",
            fontsize=22, fontweight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.35, subtitle, ha="center", va="center",
                fontsize=12, color="gray", transform=ax.transAxes)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    with PdfPages(OUTPUT_PDF) as pdf:

        # ── Section 1: Base Series ─────────────────────────────────────────
        _section_title_page(pdf,
            "Section 1 — Base Series",
            "15 base types, no features applied")

        print("\n[Section 1] Base Series")
        for display_name, base_key in BASE_SERIES:
            seed = int(rng.integers(1, 999_999))
            cfg  = _make_cfg(base_key, {}, seed)
            df, _ = dg.generate_dataframe(cfg)

            fig, ax = plt.subplots(figsize=(14, 4.5))
            _plot(ax, df, f"Base: {display_name}  ({base_key})", color="steelblue")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  ✓  {display_name}")

        # ── Section 2: Feature Showcase ────────────────────────────────────
        _section_title_page(pdf,
            "Section 2 — Feature Showcase",
            "Trend/seasonality/break on AR base  ·  Anomaly features on White Noise base")

        print("\n[Section 2] Features")
        for display_name, feat_key, feat_base, feat_extra in FEATURES:
            seed = int(rng.integers(1, 999_999))
            feat_override = {"enabled": True, **feat_extra}
            cfg  = _make_cfg(feat_base, {feat_key: feat_override}, seed)
            df, _ = dg.generate_dataframe(cfg)

            fig, ax = plt.subplots(figsize=(14, 4.5))
            _plot(ax, df,
                  f"Feature: {display_name}  (base: {feat_base})",
                  color="darkorange",
                  feat_key=feat_key)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  ✓  {display_name}")

    total = len(BASE_SERIES) + len(FEATURES) + 2   # +2 section title pages
    print(f"\nPDF saved → {OUTPUT_PDF.resolve()}")
    print(f"Total pages : {total}  "
          f"(2 title + {len(BASE_SERIES)} base + {len(FEATURES)} features)")
    print("Anomaly plots use white_noise base + red dashed markers at anomaly positions.")
    print("Break plots use ar base + red solid markers at break positions.")


if __name__ == "__main__":
    main()
