"""
BeTiSe — Full Variant-Aware Combination Gallery Test
=====================================================

Goal
----
Generate EXACTLY ONE series for every canonical composition template currently
listed in ``full_dataset_variant_aware.json`` (standalone + pair + triple),
while exercising the categorical feature variants in a deterministic round-robin
fashion.

IMPORTANT
---------
This is NOT a full Cartesian expansion of every feature variant. That would be
~millions of series. Instead:

    one composition template -> one generated series

and whenever a feature is present, the script rotates through that feature's
declared variants so all variant classes are exercised repeatedly across the
full gallery.

Outputs
-------
generated-dataset/full_variant_test/
    full_variant_gallery.pdf
    full_variant_test_summary.csv
    full_variant_failures.csv

Recommended location
--------------------
Save this file as:

    betise/core/test_full_variant_aware_gallery.py

Run from repository root:

    python -m betise.core.test_full_variant_aware_gallery

Optional:
    python -m betise.core.test_full_variant_aware_gallery --length 400
    python -m betise.core.test_full_variant_aware_gallery --max-compositions 100
    python -m betise.core.test_full_variant_aware_gallery --plots-per-page 12
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import textwrap
import traceback
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from betise.full_dataset_generation import generate_full_series


VALID_COLLECTIVE_SHAPES = [
    "rectangular",
    "gaussian",
    "triangular",
    "ramp",
    "decay",
]


# ============================================================================
# CONFIG
# ============================================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_variant_aware_config(
    config_dir: Path,
    variant_filename: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params_cfg = load_json(
        config_dir / "params.json"
    )

    full_cfg = load_json(
        config_dir / variant_filename
    )

    if "feature_variants" not in full_cfg:
        raise ValueError(
            f"{variant_filename} does not contain 'feature_variants'. "
            "Use the variant-aware full dataset config."
        )

    if "compositions" not in full_cfg:
        raise ValueError(
            f"{variant_filename} does not contain 'compositions'."
        )

    return params_cfg, full_cfg


# ============================================================================
# VARIANT MATERIALIZATION
# ============================================================================

def _normalize_variant_params(
    feature_name: str,
    raw_params: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """Translate variant-schema fields into apply_feature() config fields."""

    params = deepcopy(raw_params)

    # ------------------------------------------------------------
    # Multiple mean / variance shift:
    # directions are intentionally random per break.
    # apply_feature() already does that whenever mode == "multiple".
    # ------------------------------------------------------------

    if (
        feature_name in {"mean_shift", "variance_shift"}
        and params.get("mode") == "multiple"
    ):
        params.pop("direction", None)
        params.pop("direction_strategy", None)
        params.pop("location", None)

    # ------------------------------------------------------------
    # Multiple point anomaly:
    # count is currently computed automatically by generator.py.
    # ------------------------------------------------------------

    if feature_name == "point_anomaly":
        params.pop("count_strategy", None)

        if params.get("mode") == "multiple":
            params.pop("location", None)
            params.pop("num_anomalies", None)
            params.pop("is_spike", None)

    # ------------------------------------------------------------
    # Collective anomaly:
    # JSON uses shape_strategy; apply_feature expects anomaly_shapes.
    # ------------------------------------------------------------

    if feature_name == "collective_anomaly":
        strategy = params.pop(
            "shape_strategy",
            None,
        )

        count = int(
            params.get(
                "num_anomalies",
                1,
            )
        )

        if strategy == "mixed":
            params["anomaly_shapes"] = [
                rng.choice(
                    VALID_COLLECTIVE_SHAPES
                )
                for _ in range(count)
            ]

        elif strategy is not None:
            # A one-item list is intentionally accepted by generator.py;
            # it repeats the same shape for all anomalies.
            params["anomaly_shapes"] = [
                strategy
            ]

        if params.get("mode") == "multiple":
            params.pop("location", None)

    # ------------------------------------------------------------
    # Contextual anomaly:
    # multiple event cases do not use a single location.
    # ------------------------------------------------------------

    if (
        feature_name == "contextual_anomaly"
        and params.get("mode") == "multiple"
    ):
        params.pop("location", None)

    # ------------------------------------------------------------
    # Trend shift:
    # multiple cases do not have one fixed location.
    # "mixed" is handled by the patched apply_feature() logic.
    # ------------------------------------------------------------

    if (
        feature_name == "trend_shift"
        and params.get("mode") == "multiple"
    ):
        params.pop("location", None)

    return params


def materialize_composition(
    composition: Dict[str, Any],
    feature_variants: Dict[str, List[Dict[str, Any]]],
    variant_cursors: Dict[str, int],
    rng: random.Random,
) -> Tuple[
    Dict[str, Any],
    Dict[str, str],
]:
    """Attach exactly one deterministic categorical variant per active feature."""

    concrete = deepcopy(
        composition
    )

    feature_overrides = {}
    selected_variant_ids = {}

    for feature_name in concrete.get(
        "features",
        [],
    ):
        variants = feature_variants.get(
            feature_name,
            [],
        )

        if not variants:
            continue

        cursor = variant_cursors.get(
            feature_name,
            0,
        )

        variant = variants[
            cursor % len(variants)
        ]

        variant_cursors[
            feature_name
        ] = cursor + 1

        variant_id = variant.get(
            "variant_id",
            f"{feature_name}__variant-{cursor}",
        )

        params = _normalize_variant_params(
            feature_name=feature_name,
            raw_params=variant.get(
                "params",
                {},
            ),
            rng=rng,
        )

        feature_overrides[
            feature_name
        ] = params

        selected_variant_ids[
            feature_name
        ] = variant_id

    concrete[
        "feature_overrides"
    ] = feature_overrides

    return (
        concrete,
        selected_variant_ids,
    )


# ============================================================================
# PDF
# ============================================================================

def _grid_shape(
    plots_per_page: int,
) -> Tuple[int, int]:
    if plots_per_page <= 1:
        return 1, 1

    if plots_per_page <= 4:
        return 2, 2

    if plots_per_page <= 6:
        return 3, 2

    if plots_per_page <= 9:
        return 3, 3

    # Default / recommended: 12
    return 4, 3


def _new_page(
    plots_per_page: int,
):
    rows, cols = _grid_shape(
        plots_per_page
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            16,
            3.4 * rows,
        ),
        squeeze=False,
    )

    return (
        fig,
        axes.ravel(),
    )


def _short_title(
    composition: Dict[str, Any],
    selected_variants: Dict[str, str],
) -> str:
    bases = " + ".join(
        composition.get(
            "base_components",
            [],
        )
    )

    features = composition.get(
        "features",
        [],
    )

    feature_text = (
        " + ".join(features)
        if features
        else "base only"
    )

    variants = [
        variant_id
        for variant_id
        in selected_variants.values()
    ]

    variant_text = (
        " | ".join(variants)
        if variants
        else ""
    )

    raw = (
        f"{composition['id']} [{composition.get('group', '?')}]\n"
        f"{bases}\n"
        f"{feature_text}"
    )

    if variant_text:
        raw += (
            "\n"
            + variant_text
        )

    return "\n".join(
        textwrap.fill(
            line,
            width=65,
        )
        for line in raw.splitlines()
    )


def _plot_success(
    ax,
    df: pd.DataFrame,
    title: str,
):
    series = df.sort_values(
        "time"
    )

    ax.plot(
        series["time"],
        series["data"],
        linewidth=0.8,
    )

    ax.set_title(
        title,
        fontsize=7,
        pad=4,
    )

    ax.tick_params(
        labelsize=6,
    )

    ax.grid(
        alpha=0.20,
    )


def _plot_failure(
    ax,
    title: str,
    error: Exception,
):
    ax.axis("off")

    message = (
        f"{title}\n\n"
        f"FAILED\n"
        f"{type(error).__name__}: {error}"
    )

    ax.text(
        0.02,
        0.98,
        textwrap.fill(
            message,
            width=70,
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
    )


def _flush_page(
    pdf: PdfPages,
    fig,
    axes,
    used_axes: int,
):
    for ax in axes[
        used_axes:
    ]:
        ax.axis("off")

    fig.tight_layout(
        h_pad=1.2,
        w_pad=0.8,
    )

    pdf.savefig(
        fig
    )

    plt.close(
        fig
    )


def _write_cover_page(
    pdf: PdfPages,
    total: int,
    length: int,
    config_name: str,
):
    fig, ax = plt.subplots(
        figsize=(16, 10)
    )

    ax.axis("off")

    text = (
        "BeTiSe Full Variant-Aware Combination Gallery\n\n"
        f"Config: {config_name}\n"
        f"Canonical composition templates: {total}\n"
        f"Series per composition: 1\n"
        f"Series length: {length}\n\n"
        "Variant policy:\n"
        "Each composition template receives exactly one categorical feature "
        "variant. Variant choices rotate deterministically per feature so the "
        "variant catalog is exercised across the full run.\n\n"
        "This intentionally does NOT perform the full Cartesian expansion of "
        "every parameter-level variant."
    )

    ax.text(
        0.05,
        0.90,
        textwrap.fill(
            text,
            width=100,
        ),
        va="top",
        ha="left",
        fontsize=14,
    )

    pdf.savefig(
        fig
    )

    plt.close(
        fig
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-dir",
        default="betise/config",
    )

    parser.add_argument(
        "--variant-config",
        default="full_dataset_variant_aware.json",
    )

    parser.add_argument(
        "--output-dir",
        default="generated-dataset/full_variant_test",
    )

    parser.add_argument(
        "--length",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--plots-per-page",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-compositions",
        type=int,
        default=None,
        help="Optional quick-test limit. Omit to run ALL compositions.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write summary/failure CSV checkpoints every N compositions.",
    )

    args = parser.parse_args()

    config_dir = Path(
        args.config_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    params_cfg, full_cfg = (
        load_variant_aware_config(
            config_dir=config_dir,
            variant_filename=args.variant_config,
        )
    )

    compositions = [
        composition
        for composition
        in full_cfg["compositions"]
        if composition.get(
            "enabled",
            True,
        )
    ]

    if args.max_compositions is not None:
        compositions = compositions[
            :args.max_compositions
        ]

    total = len(
        compositions
    )

    if total == 0:
        raise ValueError(
            "No compositions selected."
        )

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    variant_rng = random.Random(
        args.seed + 991
    )

    variant_cursors = {
        feature_name: 0
        for feature_name
        in full_cfg["feature_variants"]
    }

    summary_rows = []

    pdf_path = (
        output_dir
        / "full_variant_gallery.pdf"
    )

    summary_path = (
        output_dir
        / "full_variant_test_summary.csv"
    )

    failures_path = (
        output_dir
        / "full_variant_failures.csv"
    )

    print(
        "=" * 80
    )
    print(
        "BETISE FULL VARIANT-AWARE GALLERY TEST"
    )
    print(
        "=" * 80
    )
    print(
        f"Compositions : {total}"
    )
    print(
        f"Length       : {args.length}"
    )
    print(
        f"PDF          : {pdf_path}"
    )
    print(
        "=" * 80
    )

    success_count = 0
    failure_count = 0

    fig = None
    axes = None
    page_slot = 0

    with PdfPages(
        pdf_path
    ) as pdf:

        _write_cover_page(
            pdf=pdf,
            total=total,
            length=args.length,
            config_name=args.variant_config,
        )

        for index, composition in enumerate(
            compositions,
            start=1,
        ):
            concrete, selected_variants = (
                materialize_composition(
                    composition=composition,
                    feature_variants=full_cfg[
                        "feature_variants"
                    ],
                    variant_cursors=variant_cursors,
                    rng=variant_rng,
                )
            )

            if fig is None:
                fig, axes = _new_page(
                    args.plots_per_page
                )

                page_slot = 0

            title = _short_title(
                concrete,
                selected_variants,
            )

            status = "PASS"
            error_type = ""
            error_message = ""

            try:
                df = generate_full_series(
                    composition=concrete,
                    full_cfg=full_cfg,
                    params_cfg=params_cfg,
                    series_id=index,
                    length=args.length,
                )

                if len(df) != args.length:
                    raise AssertionError(
                        f"Expected length {args.length}, got {len(df)}."
                    )

                if not np.isfinite(
                    df["data"]
                ).all():
                    raise AssertionError(
                        "Generated data contains non-finite values."
                    )

                _plot_success(
                    axes[page_slot],
                    df,
                    title,
                )

                success_count += 1

            except Exception as exc:
                status = "FAIL"
                error_type = type(
                    exc
                ).__name__
                error_message = str(
                    exc
                )

                _plot_failure(
                    axes[page_slot],
                    title,
                    exc,
                )

                failure_count += 1

            summary_rows.append({
                "index": index,
                "composition_id": composition["id"],
                "composition_name": composition["name"],
                "group": composition.get("group"),
                "base_components": "|".join(
                    composition.get(
                        "base_components",
                        [],
                    )
                ),
                "features": "|".join(
                    composition.get(
                        "features",
                        [],
                    )
                ),
                "selected_variants": json.dumps(
                    selected_variants,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "feature_overrides": json.dumps(
                    concrete.get(
                        "feature_overrides",
                        {},
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "status": status,
                "error_type": error_type,
                "error_message": error_message,
            })

            if (
                args.checkpoint_every > 0
                and index % args.checkpoint_every == 0
            ):
                checkpoint_df = pd.DataFrame(
                    summary_rows
                )

                checkpoint_df.to_csv(
                    summary_path,
                    index=False,
                )

                checkpoint_df[
                    checkpoint_df["status"] == "FAIL"
                ].to_csv(
                    failures_path,
                    index=False,
                )

            page_slot += 1

            if (
                page_slot
                == len(axes)
            ):
                _flush_page(
                    pdf=pdf,
                    fig=fig,
                    axes=axes,
                    used_axes=page_slot,
                )

                fig = None
                axes = None
                page_slot = 0

            if (
                index % 100 == 0
                or index == total
            ):
                print(
                    f"{index:5d}/{total} | "
                    f"PASS={success_count} | "
                    f"FAIL={failure_count}"
                )

        if fig is not None:
            _flush_page(
                pdf=pdf,
                fig=fig,
                axes=axes,
                used_axes=page_slot,
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    failures_df = summary_df[
        summary_df["status"]
        == "FAIL"
    ].copy()

    failures_df.to_csv(
        failures_path,
        index=False,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "FULL VARIANT-AWARE GALLERY TEST COMPLETE"
    )
    print(
        "=" * 80
    )
    print(
        f"Total   : {total}"
    )
    print(
        f"PASS    : {success_count}"
    )
    print(
        f"FAIL    : {failure_count}"
    )
    print(
        f"PDF     : {pdf_path.resolve()}"
    )
    print(
        f"Summary : {summary_path.resolve()}"
    )
    print(
        f"Failures: {failures_path.resolve()}"
    )

    if failure_count == 0:
        print()
        print(
            "PASS — every canonical composition generated successfully."
        )
    else:
        print()
        print(
            "COMPLETE WITH FAILURES — inspect full_variant_failures.csv."
        )


if __name__ == "__main__":
    main()
