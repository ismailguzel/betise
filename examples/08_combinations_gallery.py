"""
BeTiSe Combinations Gallery
=============================

Generates one time series plot for every row in data/combinations.csv
and saves them all into a single PDF.

Unlike 07_visual_report.py (which randomly samples), this script renders
every combination exactly once — useful for a complete visual audit.

Output:
  generated-dataset/visual_report/combinations_gallery.pdf

Run:
    python examples/09_combinations_gallery.py
"""

import copy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from betise import dataset_generation as dg
from betise.config import load_config

# ── Constants ─────────────────────────────────────────────────────────────────

COMBINATIONS_CSV = Path(__file__).parent / "data" / "combinations.csv"
OUTPUT_DIR       = Path("generated-dataset") / "visual_report"
OUTPUT_PDF       = OUTPUT_DIR / "combinations_gallery.pdf"
SEED             = 42
LENGTH           = 700

BASE_SERIES_MAP = {
    "AR":            "ar",
    "MA":            "ma",
    "ARMA":          "arma",
    "ARIMA":         "arima",
    "ARI":           "ari",
    "IMA":           "ima",
    "RW":            "random_walk",
    "RW with Drift": "random_walk_drift",
    "White Noise":   "white_noise",
    "SARMA":         "sarma",
    "SARIMA":        "sarima",
    "ARCH":          "arch",
    "GARCH":         "garch",
    "EGARCH":        "egarch",
    "APARCH":        "aparch",
}

FEATURE_MAP = {
    "Trend: Linear":                     "linear_trend",
    "Trend: Quadratic":                  "quadratic_trend",
    "Trend: Cubic":                      "cubic_trend",
    "Trend: Exponential":                "exponential_trend",
    "Seasonality: Single Seasonality":   "single_seasonality",
    "Seasonality: Multiple Seasonality": "multiple_seasonality",
    "Anomaly: Point":                    "point_anomaly",
    "Anomaly: Collective":               "collective_anomaly",
    "Anomaly: Contextual":               "contextual_anomaly",
    "Break: Mean Shift":                 "mean_shift",
    "Break: Variance Shift":             "variance_shift",
    "Break: Trend Shift":                "trend_shift",
}

ALL_FEATURES_OFF = {k: {"enabled": False} for k in FEATURE_MAP.values()}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(val):
    if val is None:
        return None
    s = str(val).strip()
    return None if not s or s.lower() == "nan" else s


def _make_cfg(base_series, feat_keys, seed, cfg_root):
    feats = copy.deepcopy(ALL_FEATURES_OFF)
    for k in feat_keys:
        feats[k] = {"enabled": True}
    return load_config(str(cfg_root), dataset={
        "base_series":  base_series,
        "num_series":   1,
        "length_range": [LENGTH, LENGTH],
        "random_seed":  seed,
        "output_dir":   str(OUTPUT_DIR),
        "output_name":  "combo_tmp.parquet",
        "features":     feats,
    })


def _plot_series(ax, df, title):
    s = df.sort_values("time")
    ax.plot(s["time"], s["data"], linewidth=1.3, color="steelblue")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.grid(alpha=0.25)
    data_range = s["data"].max() - s["data"].min()
    if data_range > 100:
        ax.set_yscale("symlog")
    else:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))


def _section_page(pdf, text):
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    ax.text(0.5, 0.55, text, ha="center", va="center",
            fontsize=18, fontweight="bold", transform=ax.transAxes)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    root     = Path.cwd().resolve()
    cfg_root = root / "betise" / "config"

    combo_df = pd.read_csv(COMBINATIONS_CSV, sep="\t")
    # Strip BOM from column names if present
    combo_df.columns = [c.strip().lstrip("\ufeff") for c in combo_df.columns]

    rng   = np.random.default_rng(SEED)
    total = len(combo_df)
    print(f"Toplam kombinasyon: {total}")

    # Group by Type for section dividers
    current_type = None
    rendered = 0
    errors   = 0

    with PdfPages(OUTPUT_PDF) as pdf:
        for idx, row in combo_df.iterrows():
            base_display = str(row.get("Base", "")).strip()
            series_type  = str(row.get("Type", "")).strip()
            c1 = _clean(row.get("Characteristic 1"))
            c2 = _clean(row.get("Characteristic 2"))

            # Section divider when type changes
            if series_type != current_type:
                current_type = series_type
                _section_page(pdf, f"── {series_type} ──")
                print(f"\n[{series_type}]")

            base_key  = BASE_SERIES_MAP.get(base_display, base_display.lower())
            feat_keys = [FEATURE_MAP[c] for c in [c1, c2] if c and c in FEATURE_MAP]

            features_str = " + ".join([c1] + ([c2] if c2 else []))
            title = f"{base_display}  |  {features_str}"

            seed = int(rng.integers(1, 999_999))
            try:
                cfg = _make_cfg(base_key, feat_keys, seed, cfg_root)
                df_series, _ = dg.generate_dataframe(cfg)

                fig, ax = plt.subplots(figsize=(14, 4))
                _plot_series(ax, df_series, title)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                rendered += 1

            except Exception as e:
                print(f"  ✗  [{idx}] {title}  →  {e}")
                errors += 1
                plt.close("all")
                continue

            if rendered % 50 == 0:
                print(f"  {rendered}/{total} tamamlandı...")

    print(f"\nPDF kaydedildi → {OUTPUT_PDF.resolve()}")
    print(f"Toplam: {rendered} başarılı, {errors} hata")


if __name__ == "__main__":
    main()
