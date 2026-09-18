"""Canonical combination rules for BeTiSe.

This module is the single source of truth for:

1. Base-family / base-subtype combinations.
2. Overlay feature dependencies and exclusions.
3. Canonical composition order.
4. Pre-generation validation gates.
5. Lightweight post-generation sanity / metadata gates.

Important project conventions
-----------------------------
- Seasonality is a BASE family, not an overlay feature.
- Volatility may still be configured under ``dataset.features`` for legacy
  config compatibility, but when it participates in a validated combination
  it is treated as a mathematical BASE component.
- ``pure_sarma``, ``pure_sarima`` and ``seasonal_unit_root_fourier`` are
  experimental/reference generators and are intentionally outside the
  canonical combination table.
- Component-preservation statistics are validated by dedicated combination
  tests. They are not re-run inside the dataset-generation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


# ============================================================================
# CANONICAL VOCABULARY
# ============================================================================

ALLOW_VALIDATED = "ALLOW_VALIDATED"
ALLOW_REQUIRES_VALIDATION = "ALLOW_REQUIRES_VALIDATION"

EXCLUDE_REDUNDANT = "EXCLUDE_REDUNDANT"
EXCLUDE_DOMINATED = "EXCLUDE_DOMINATED"
EXCLUDE_NON_IDENTIFIABLE = "EXCLUDE_NON_IDENTIFIABLE"
EXCLUDE_CONFLICT = "EXCLUDE_CONFLICT"

REQUIRE_COMPONENT = "REQUIRE_COMPONENT"

ALLOWED_STATUSES = {
    ALLOW_VALIDATED,
    ALLOW_REQUIRES_VALIDATION,
}

EXCLUDED_STATUSES = {
    EXCLUDE_REDUNDANT,
    EXCLUDE_DOMINATED,
    EXCLUDE_NON_IDENTIFIABLE,
    EXCLUDE_CONFLICT,
    REQUIRE_COMPONENT,
}


# ============================================================================
# COMPOSITION OPERATIONS
# ============================================================================

STANDALONE = "standalone"
VOLATILITY_AS_INNOVATIONS = "volatility_as_innovations"
ADDITIVE_FOURIER = "additive_fourier"

COMPOSITION_ORDER = {
    VOLATILITY_AS_INNOVATIONS: 0,
    ADDITIVE_FOURIER: 1,
    STANDALONE: 99,
}


# ============================================================================
# BASE FAMILIES
# ============================================================================

STATIONARY_BASE_SERIES = {
    "ar",
    "ma",
    "arma",
    "white_noise",
}

STOCHASTIC_BASE_SERIES = {
    "random_walk",
    "random_walk_drift",
    "ari",
    "ima",
    "arima",
}

SEASONAL_BASE_SERIES = {
    "single_seasonality",
    "multiple_seasonality",
    "sarma",
    "sarima",
}

VOLATILITY_BASE_SERIES = {
    "arch",
    "garch",
    "egarch",
    "aparch",
}

FRACTIONAL_BASE_SERIES = {
    "arfima",
}

CANONICAL_BASE_SERIES = (
    STATIONARY_BASE_SERIES
    | STOCHASTIC_BASE_SERIES
    | SEASONAL_BASE_SERIES
    | VOLATILITY_BASE_SERIES
    | FRACTIONAL_BASE_SERIES
)

EXPERIMENTAL_BASE_SERIES = {
    "pure_sarma",
    "pure_sarima",
    "seasonal_unit_root_fourier",
}


# ============================================================================
# OVERLAY FEATURES
# ============================================================================

TREND_FEATURES = {
    "linear_trend",
    "quadratic_trend",
    "cubic_trend",
    "exponential_trend",
    "damped_trend",
}

BREAK_FEATURES = {
    "mean_shift",
    "variance_shift",
    "trend_shift",
}

ANOMALY_FEATURES = {
    "point_anomaly",
    "collective_anomaly",
    "contextual_anomaly",
}

OVERLAY_FEATURES = (
    TREND_FEATURES
    | BREAK_FEATURES
    | ANOMALY_FEATURES
)

# Volatility remains a legacy config "feature", but NOT an overlay feature.
LEGACY_BASE_FEATURES = VOLATILITY_BASE_SERIES

# Canonical post-base application order.
FEATURE_STAGE_ORDER = {
    "trend": 0,
    "structural_break": 1,
    "anomaly": 2,
}


# ============================================================================
# VALIDATION GATES
# ============================================================================

VALIDATION_GATES = {
    "GATE_1_RULE_VALIDITY": (
        "Every requested pair of base components must be allowed by the "
        "canonical base-combination table."
    ),
    "GATE_2_DEPENDENCIES": (
        "Feature prerequisites must exist, e.g. contextual anomaly requires "
        "seasonality and trend shift requires linear trend."
    ),
    "GATE_3_EXCLUSIONS": (
        "Explicit feature conflicts must be rejected, e.g. variance shift "
        "with any volatility base component."
    ),
    "GATE_4_IMPLEMENTATION": (
        "Every allowed requested component must have an implemented generator "
        "path before generation begins."
    ),
    "GATE_5_NUMERICAL_SANITY": (
        "Generated output must have the expected length and only finite values."
    ),
    "GATE_6_COMPONENT_PRESERVATION": (
        "Dedicated statistical validation tests must show that intended "
        "characteristics remain detectable after composition."
    ),
    "GATE_7_METADATA": (
        "base_components, base_families, composition_steps and feature "
        "metadata must match the requested canonical composition."
    ),
    "GATE_8_LABEL_INTEGRITY": (
        "Requested anomaly/break localization labels must exist and contain "
        "valid event/regime labels."
    ),
}


# ============================================================================
# RULE RECORD
# ============================================================================

@dataclass(frozen=True)
class Rule:
    status: str
    reason: str
    composition_steps: Tuple[str, ...] = ()
    validation: str = "validated"

    @property
    def allowed(self) -> bool:
        return self.status in ALLOWED_STATUSES


# ============================================================================
# FAMILY HELPERS
# ============================================================================

def base_family(base_series: str) -> str:
    """Return the canonical family of a base component."""

    if base_series in STATIONARY_BASE_SERIES:
        return "stationary"

    if base_series in STOCHASTIC_BASE_SERIES:
        return "stochastic"

    if base_series in SEASONAL_BASE_SERIES:
        return "seasonality"

    if base_series in VOLATILITY_BASE_SERIES:
        return "volatility"

    if base_series in FRACTIONAL_BASE_SERIES:
        return "fractional"

    if base_series in EXPERIMENTAL_BASE_SERIES:
        return "experimental"

    raise ValueError(
        f"Unknown base series '{base_series}'. "
        f"Canonical choices: {sorted(CANONICAL_BASE_SERIES)}"
    )


def feature_family(feature_name: str) -> str:
    """Return the canonical overlay-feature family."""

    if feature_name in TREND_FEATURES:
        return "trend"

    if feature_name in BREAK_FEATURES:
        return "structural_break"

    if feature_name in ANOMALY_FEATURES:
        return "anomaly"

    if feature_name in LEGACY_BASE_FEATURES:
        return "volatility"

    raise ValueError(
        f"Unknown feature '{feature_name}'. "
        f"Overlay choices: {sorted(OVERLAY_FEATURES)}"
    )


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


# ============================================================================
# BASE-COMBINATION RULE RESOLVER
# ============================================================================

def resolve_base_pair_rule(component_a: str, component_b: str) -> Rule:
    """Resolve the canonical rule for two base components.

    Rules are symmetric; component order does not matter.
    """

    if component_a == component_b:
        return Rule(
            EXCLUDE_CONFLICT,
            "The same base component cannot be stacked with itself.",
        )

    if component_a not in CANONICAL_BASE_SERIES:
        if component_a in EXPERIMENTAL_BASE_SERIES:
            return Rule(
                EXCLUDE_CONFLICT,
                f"'{component_a}' is experimental/reference-only and is "
                "outside the canonical combination table.",
            )
        raise ValueError(f"Unknown base component: {component_a}")

    if component_b not in CANONICAL_BASE_SERIES:
        if component_b in EXPERIMENTAL_BASE_SERIES:
            return Rule(
                EXCLUDE_CONFLICT,
                f"'{component_b}' is experimental/reference-only and is "
                "outside the canonical combination table.",
            )
        raise ValueError(f"Unknown base component: {component_b}")

    family_a = base_family(component_a)
    family_b = base_family(component_b)

    # ------------------------------------------------------------------
    # Same-family stacking
    # ------------------------------------------------------------------

    if family_a == family_b:
        return Rule(
            EXCLUDE_CONFLICT,
            "Canonical generation uses at most one subtype from each base "
            f"family; both components belong to '{family_a}'.",
        )

    family_pair = frozenset((family_a, family_b))

    # ------------------------------------------------------------------
    # Stationary + Stochastic
    # ------------------------------------------------------------------

    if family_pair == {"stationary", "stochastic"}:
        return Rule(
            EXCLUDE_REDUNDANT,
            "Adding an I(0) stationary process to an integer-integrated "
            "stochastic process does not define a separate identifiable "
            "dataset characteristic; the result is representable by a richer "
            "stochastic/ARIMA-type process.",
        )

    # ------------------------------------------------------------------
    # Stationary + Volatility
    # ------------------------------------------------------------------

    if family_pair == {"stationary", "volatility"}:
        stationary = (
            component_a
            if family_a == "stationary"
            else component_b
        )

        if stationary == "white_noise":
            return Rule(
                EXCLUDE_REDUNDANT,
                "White noise + volatility is not kept as a separate canonical "
                "combination; the volatility process already supplies the "
                "innovation mechanism.",
            )

        return Rule(
            ALLOW_VALIDATED,
            "Volatility innovations can drive the stationary AR/MA/ARMA "
            "filter while both short-memory dynamics and heteroskedasticity "
            "remain detectable.",
            (VOLATILITY_AS_INNOVATIONS,),
        )

    # ------------------------------------------------------------------
    # Stationary + Seasonality
    # ------------------------------------------------------------------

    if family_pair == {"stationary", "seasonality"}:
        seasonal = (
            component_a
            if family_a == "seasonality"
            else component_b
        )

        if seasonal in {"single_seasonality", "multiple_seasonality"}:
            return Rule(
                ALLOW_VALIDATED,
                "Deterministic Fourier seasonality can be added to a "
                "stationary background while preserving both components.",
                (ADDITIVE_FOURIER,),
            )

        return Rule(
            EXCLUDE_REDUNDANT,
            "Deterministic SARMA/SARIMA already contains an internal "
            "non-seasonal ARMA/ARIMA background; adding another stationary "
            "base duplicates that structure.",
        )

    # ------------------------------------------------------------------
    # Stationary + Fractional
    # ------------------------------------------------------------------

    if family_pair == {"stationary", "fractional"}:
        return Rule(
            EXCLUDE_REDUNDANT,
            "ARFIMA already contains a stationary ARMA short-memory component; "
            "an additional stationary base is redundant.",
        )

    # ------------------------------------------------------------------
    # Stochastic + Volatility
    # ------------------------------------------------------------------

    if family_pair == {"stochastic", "volatility"}:
        return Rule(
            ALLOW_VALIDATED,
            "Volatility can act as the innovation mechanism of the stochastic "
            "process while stochastic integration and heteroskedasticity are "
            "both preserved.",
            (VOLATILITY_AS_INNOVATIONS,),
        )

    # ------------------------------------------------------------------
    # Stochastic + Seasonality
    # ------------------------------------------------------------------

    if family_pair == {"stochastic", "seasonality"}:
        stochastic = (
            component_a
            if family_a == "stochastic"
            else component_b
        )

        seasonal = (
            component_a
            if family_a == "seasonality"
            else component_b
        )

        if seasonal in {"sarma", "sarima"}:
            return Rule(
                EXCLUDE_NON_IDENTIFIABLE,
                "Deterministic SARMA/SARIMA already contains a non-seasonal "
                "ARMA/ARIMA background, so stacking another stochastic base "
                "creates overlapping/non-identifiable dynamics.",
            )

        if stochastic == "arima" and seasonal == "single_seasonality":
            return Rule(
                EXCLUDE_REDUNDANT,
                "ARIMA + single deterministic Fourier seasonality is already "
                "represented by the project's deterministic SARIMA generator.",
            )

        if seasonal in {"single_seasonality", "multiple_seasonality"}:
            return Rule(
                ALLOW_VALIDATED,
                "Deterministic Fourier seasonality can be added to the "
                "stochastic background and both characteristics remain "
                "detectable.",
                (ADDITIVE_FOURIER,),
            )

    # ------------------------------------------------------------------
    # Stochastic + Fractional
    # ------------------------------------------------------------------

    if family_pair == {"stochastic", "fractional"}:
        return Rule(
            EXCLUDE_DOMINATED,
            "Integer integration I(1)/I(2) dominates the low-frequency "
            "behaviour of stationary long-memory ARFIMA with 0 < d < 0.5, "
            "so the fractional characteristic is not cleanly identifiable.",
        )

    # ------------------------------------------------------------------
    # Seasonality + Volatility
    # ------------------------------------------------------------------

    if family_pair == {"seasonality", "volatility"}:
        seasonal = (
            component_a
            if family_a == "seasonality"
            else component_b
        )

        if seasonal in {"single_seasonality", "multiple_seasonality"}:
            return Rule(
                ALLOW_VALIDATED,
                "The volatility realization can be used as the background and "
                "deterministic Fourier seasonality can be added on top.",
                (ADDITIVE_FOURIER,),
            )

        return Rule(
            ALLOW_VALIDATED,
            "For deterministic SARMA/SARIMA, volatility innovations drive the "
            "internal ARMA/ARIMA core while deterministic Fourier seasonality "
            "remains the seasonal component.",
            (VOLATILITY_AS_INNOVATIONS,),
        )

    # ------------------------------------------------------------------
    # Seasonality + Fractional
    # ------------------------------------------------------------------

    if family_pair == {"seasonality", "fractional"}:
        seasonal = (
            component_a
            if family_a == "seasonality"
            else component_b
        )

        if seasonal in {"single_seasonality", "multiple_seasonality"}:
            return Rule(
                ALLOW_VALIDATED,
                "Deterministic Fourier seasonality and stationary fractional "
                "long memory coexist and were jointly validated.",
                (ADDITIVE_FOURIER,),
            )

        if seasonal == "sarma":
            return Rule(
                EXCLUDE_REDUNDANT,
                "Deterministic SARMA contains an ARMA background that overlaps "
                "with ARFIMA's own short-memory ARMA structure.",
            )

        if seasonal == "sarima":
            return Rule(
                EXCLUDE_DOMINATED,
                "Deterministic SARIMA contains integer-integrated ARIMA "
                "background dynamics that dominate stationary fractional "
                "long memory.",
            )

    # ------------------------------------------------------------------
    # Volatility + Fractional
    # ------------------------------------------------------------------

    if family_pair == {"volatility", "fractional"}:
        return Rule(
            ALLOW_VALIDATED,
            "ARCH/GARCH/EGARCH/APARCH innovations can drive the ARFIMA "
            "fractional filter; long memory and volatility were jointly "
            "validated.",
            (VOLATILITY_AS_INNOVATIONS,),
        )

    raise RuntimeError(
        "No canonical rule was defined for "
        f"{component_a} + {component_b}."
    )


# Materialized lookup table for inspection/export/testing.
BASE_COMBINATION_RULES: Dict[Tuple[str, str], Rule] = {
    _pair_key(a, b): resolve_base_pair_rule(a, b)
    for a, b in combinations(sorted(CANONICAL_BASE_SERIES), 2)
}


# ============================================================================
# FEATURE RULES
# ============================================================================

FEATURE_RULES: Dict[str, Dict[str, Any]] = {
    "linear_trend": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": (),
        "reason": (
            "Deterministic linear trend may be overlaid on any valid base "
            "composition. Visibility/preservation is validated separately."
        ),
    },
    "quadratic_trend": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": (),
        "reason": (
            "Deterministic quadratic trend may be overlaid on any valid base "
            "composition."
        ),
    },
    "cubic_trend": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": (),
        "reason": (
            "Deterministic cubic trend may be overlaid on any valid base "
            "composition."
        ),
    },
    "exponential_trend": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": (),
        "reason": (
            "Deterministic exponential trend may be overlaid on any valid base "
            "composition."
        ),
    },
    "damped_trend": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": (),
        "reason": (
            "Deterministic damped trend may be overlaid on any valid base "
            "composition."
        ),
    },
    "point_anomaly": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": (),
        "reason": (
            "Point anomalies may be added to any valid current series."
        ),
    },
    "collective_anomaly": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": (),
        "reason": (
            "Collective anomalies may be added to any valid current series."
        ),
    },
    "contextual_anomaly": {
        "status": REQUIRE_COMPONENT,
        "requires": ("seasonality",),
        "forbids_base_families": (),
        "reason": (
            "Contextual anomalies require an existing deterministic seasonal "
            "context."
        ),
    },
    "mean_shift": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": (),
        "reason": (
            "Mean shift may be applied to any valid current series; visibility "
            "is validated separately for difficult backgrounds."
        ),
    },
    "variance_shift": {
        "status": ALLOW_REQUIRES_VALIDATION,
        "requires": (),
        "forbids_base_families": ("volatility",),
        "reason": (
            "Variance shift is allowed except when a volatility component is "
            "already present, where the two variance mechanisms are difficult "
            "to identify separately."
        ),
    },
    "trend_shift": {
        "status": REQUIRE_COMPONENT,
        "requires": ("linear_trend",),
        "forbids_base_families": (),
        "reason": (
            "Trend shift is defined as a change in an existing linear trend, "
            "so linear_trend must be applied first."
        ),
    },
}


# ============================================================================
# STACKING RULES
# ============================================================================

MAX_OVERLAY_SUBTYPES_PER_FAMILY = {
    "trend": 1,
    "structural_break": 1,
    "anomaly": 1,
}


# ============================================================================
# VALIDATION RESULT
# ============================================================================

@dataclass
class ValidationReport:
    valid: bool
    base_components: List[str]
    base_families: List[str]
    feature_components: List[str]
    feature_families: List[str]
    composition_steps: List[str]
    gates: Dict[str, Dict[str, Any]]
    errors: List[str]
    warnings: List[str]

    def raise_for_errors(self) -> None:
        if self.valid:
            return

        details = "\n".join(
            f"- {message}"
            for message in self.errors
        )

        raise ValueError(
            "Invalid BeTiSe combination:\n"
            f"{details}"
        )


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def _ordered_composition_steps(
    base_components: Sequence[str],
) -> List[str]:
    if len(base_components) <= 1:
        return [STANDALONE]

    steps = []

    for a, b in combinations(base_components, 2):
        rule = resolve_base_pair_rule(a, b)

        if rule.allowed:
            steps.extend(rule.composition_steps)

    steps = _unique_preserve_order(steps)

    return sorted(
        steps,
        key=lambda step: COMPOSITION_ORDER.get(
            step,
            50
        ),
    )


def _feature_sort_key(feature_name: str) -> int:
    return FEATURE_STAGE_ORDER[
        feature_family(feature_name)
    ]


# ============================================================================
# PRE-GENERATION CANONICAL VALIDATION
# ============================================================================

def validate_requested_combination(
    base_components: Sequence[str],
    feature_components: Sequence[str] = (),
    implemented_base_components: Iterable[str] | None = None,
    implemented_features: Iterable[str] | None = None,
) -> ValidationReport:
    """Validate a requested BeTiSe combination before generation.

    Parameters
    ----------
    base_components:
        Mathematical base components. Volatility belongs here when it is used
        as a validated innovation/background mechanism.

        Examples:
            ["ar"]
            ["ar", "garch"]
            ["arima", "garch", "multiple_seasonality"]
            ["arfima", "garch", "single_seasonality"]

    feature_components:
        True overlays only: deterministic trends, structural breaks, anomalies.

    implemented_base_components / implemented_features:
        Optional implementation registry used by GATE_4. If omitted, all
        canonical components are assumed to have an implementation path.

    Returns
    -------
    ValidationReport
    """

    base_components = list(base_components)
    feature_components = list(feature_components)

    errors: List[str] = []
    warnings: List[str] = []

    gates: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Basic input validation
    # ------------------------------------------------------------------

    if not base_components:
        errors.append(
            "At least one base component is required."
        )

    duplicate_bases = {
        component
        for component in base_components
        if base_components.count(component) > 1
    }

    if duplicate_bases:
        errors.append(
            "Duplicate base components are not allowed: "
            f"{sorted(duplicate_bases)}"
        )

    duplicate_features = {
        feature
        for feature in feature_components
        if feature_components.count(feature) > 1
    }

    if duplicate_features:
        errors.append(
            "Duplicate overlay features are not allowed: "
            f"{sorted(duplicate_features)}"
        )

    unknown_bases = [
        component
        for component in base_components
        if component not in CANONICAL_BASE_SERIES
    ]

    if unknown_bases:
        errors.append(
            "Unknown/non-canonical base components: "
            f"{unknown_bases}"
        )

    unknown_features = [
        feature
        for feature in feature_components
        if feature not in OVERLAY_FEATURES
    ]

    if unknown_features:
        errors.append(
            "Unknown/non-overlay feature components: "
            f"{unknown_features}"
        )

    # Family lists can only be built safely for known items.
    known_bases = [
        component
        for component in base_components
        if component in CANONICAL_BASE_SERIES
    ]

    known_features = [
        feature
        for feature in feature_components
        if feature in OVERLAY_FEATURES
    ]

    base_families = [
        base_family(component)
        for component in known_bases
    ]

    feature_families = [
        feature_family(feature)
        for feature in known_features
    ]

    # ------------------------------------------------------------------
    # GATE 1 — base rule validity
    # ------------------------------------------------------------------

    gate_1_errors = []

    for a, b in combinations(known_bases, 2):
        rule = resolve_base_pair_rule(a, b)

        if not rule.allowed:
            gate_1_errors.append(
                f"{a} + {b}: {rule.status} — {rule.reason}"
            )

    if gate_1_errors:
        errors.extend(gate_1_errors)

    gates["GATE_1_RULE_VALIDITY"] = {
        "passed": not gate_1_errors,
        "details": gate_1_errors,
    }

    # ------------------------------------------------------------------
    # Canonical max-one-subtype-per-base-family rule
    # ------------------------------------------------------------------

    repeated_base_families = sorted({
        family
        for family in base_families
        if base_families.count(family) > 1
    })

    if repeated_base_families:
        family_errors = [
            "Only one subtype per base family is allowed. "
            f"Repeated families: {repeated_base_families}"
        ]
        errors.extend(family_errors)

        gates["GATE_1_RULE_VALIDITY"]["passed"] = False
        gates["GATE_1_RULE_VALIDITY"]["details"].extend(
            family_errors
        )

    # ------------------------------------------------------------------
    # Max one overlay subtype per semantic family
    # ------------------------------------------------------------------

    stacking_errors = []

    for family, maximum in MAX_OVERLAY_SUBTYPES_PER_FAMILY.items():
        selected = [
            feature
            for feature in known_features
            if feature_family(feature) == family
        ]

        if len(selected) > maximum:
            stacking_errors.append(
                f"At most {maximum} '{family}' overlay subtype is allowed; "
                f"requested {selected}."
            )

    if stacking_errors:
        errors.extend(stacking_errors)

    # ------------------------------------------------------------------
    # GATE 2 — dependencies
    # ------------------------------------------------------------------

    dependency_errors = []

    for feature in known_features:
        rule = FEATURE_RULES[feature]

        for requirement in rule["requires"]:
            if requirement == "seasonality":
                if "seasonality" not in base_families:
                    dependency_errors.append(
                        "contextual_anomaly requires an existing seasonal "
                        "base component."
                    )

            elif requirement in OVERLAY_FEATURES:
                if requirement not in known_features:
                    dependency_errors.append(
                        f"{feature} requires '{requirement}' to be selected."
                    )

            else:
                dependency_errors.append(
                    f"{feature} has unknown canonical requirement "
                    f"'{requirement}'."
                )

    if dependency_errors:
        errors.extend(dependency_errors)

    gates["GATE_2_DEPENDENCIES"] = {
        "passed": not dependency_errors,
        "details": dependency_errors,
    }

    # ------------------------------------------------------------------
    # GATE 3 — explicit exclusions
    # ------------------------------------------------------------------

    exclusion_errors = list(stacking_errors)

    for feature in known_features:
        forbidden_families = FEATURE_RULES[
            feature
        ]["forbids_base_families"]

        conflicts = sorted(
            set(base_families)
            & set(forbidden_families)
        )

        if conflicts:
            exclusion_errors.append(
                f"{feature} is not allowed when base families "
                f"{conflicts} are present."
            )

    if exclusion_errors:
        # stacking errors may already be in errors; only append new conflicts.
        for message in exclusion_errors:
            if message not in errors:
                errors.append(message)

    gates["GATE_3_EXCLUSIONS"] = {
        "passed": not exclusion_errors,
        "details": exclusion_errors,
    }

    # ------------------------------------------------------------------
    # GATE 4 — implementation availability
    # ------------------------------------------------------------------

    implementation_errors = []

    if implemented_base_components is not None:
        implemented_base_components = set(
            implemented_base_components
        )

        missing = [
            component
            for component in known_bases
            if component not in implemented_base_components
        ]

        if missing:
            implementation_errors.append(
                "Missing base implementation(s): "
                f"{missing}"
            )

    if implemented_features is not None:
        implemented_features = set(
            implemented_features
        )

        missing = [
            feature
            for feature in known_features
            if feature not in implemented_features
        ]

        if missing:
            implementation_errors.append(
                "Missing feature implementation(s): "
                f"{missing}"
            )

    if implementation_errors:
        errors.extend(implementation_errors)

    gates["GATE_4_IMPLEMENTATION"] = {
        "passed": not implementation_errors,
        "details": implementation_errors,
    }

    # GATE 6 is an offline test-suite requirement.
    gates["GATE_6_COMPONENT_PRESERVATION"] = {
        "passed": None,
        "details": [
            "Not evaluated in the generation loop. Use dedicated joint "
            "statistical validation tests for each canonical composition."
        ],
    }

    composition_steps = (
        _ordered_composition_steps(known_bases)
        if not gate_1_errors
        else []
    )

    # Warn when selected overlays are allowed logically but still require
    # empirical preservation validation across the final intended dataset.
    for feature in known_features:
        if FEATURE_RULES[feature]["status"] == ALLOW_REQUIRES_VALIDATION:
            warnings.append(
                f"{feature}: logically allowed; feature visibility/preservation "
                "must be covered by validation tests."
            )

    return ValidationReport(
        valid=len(errors) == 0,
        base_components=known_bases,
        base_families=base_families,
        feature_components=sorted(
            known_features,
            key=_feature_sort_key
        ),
        feature_families=[
            feature_family(feature)
            for feature in sorted(
                known_features,
                key=_feature_sort_key
            )
        ],
        composition_steps=composition_steps,
        gates=gates,
        errors=errors,
        warnings=warnings,
    )


# ============================================================================
# POST-GENERATION GATES
# ============================================================================

def validate_numerical_output(
    data: Sequence[float],
    expected_length: int,
) -> Dict[str, Any]:
    """GATE 5: length + finite-value sanity."""

    values = np.asarray(
        data,
        dtype=float
    )

    errors = []

    if len(values) != int(expected_length):
        errors.append(
            f"Expected length {expected_length}, "
            f"got {len(values)}."
        )

    if not np.all(np.isfinite(values)):
        errors.append(
            "Generated series contains NaN or infinite values."
        )

    return {
        "gate": "GATE_5_NUMERICAL_SANITY",
        "passed": not errors,
        "details": errors,
    }


def validate_metadata_contract(
    metadata: Mapping[str, Any],
    report: ValidationReport,
) -> Dict[str, Any]:
    """GATE 7: compare persisted metadata with canonical request."""

    errors = []

    actual_base_components = metadata.get(
        "base_components"
    )

    actual_base_families = metadata.get(
        "base_families"
    )

    actual_steps = metadata.get(
        "composition_steps"
    )

    actual_feature_components = metadata.get(
        "feature_components"
    )

    actual_feature_families = metadata.get(
        "feature_families"
    )

    expected_pairs = {
        "base_components": report.base_components,
        "base_families": report.base_families,
        "composition_steps": report.composition_steps,
        "feature_components": report.feature_components,
        "feature_families": report.feature_families,
    }

    actual_pairs = {
        "base_components": actual_base_components,
        "base_families": actual_base_families,
        "composition_steps": actual_steps,
        "feature_components": actual_feature_components,
        "feature_families": actual_feature_families,
    }

    for key, expected in expected_pairs.items():
        actual = actual_pairs[key]

        if actual is None:
            errors.append(
                f"Missing metadata field '{key}'."
            )
            continue

        if list(actual) != list(expected):
            errors.append(
                f"{key} mismatch: expected {expected}, "
                f"got {actual}."
            )

    return {
        "gate": "GATE_7_METADATA",
        "passed": not errors,
        "details": errors,
    }


def validate_label_integrity(
    dataframe: Any,
    feature_components: Sequence[str],
) -> Dict[str, Any]:
    """GATE 8: lightweight anomaly/break localization-label contract.

    This validates label presence and simple value sanity when localization
    labels are present/requested. It does not replace dedicated event-level
    validation tests.
    """

    columns = set(
        getattr(dataframe, "columns", [])
    )

    errors = []

    label_map = {
        "point_anomaly": "point_anom_label",
        "collective_anomaly": "collect_anom_label",
        "contextual_anomaly": "context_anom_label",
        "mean_shift": "mean_shift_label",
        "variance_shift": "variance_shift_label",
        "trend_shift": "trend_shift_label",
    }

    for feature in feature_components:
        label_column = label_map.get(feature)

        if label_column is None:
            continue

        # Labels are optional at generator level because is_loc may be False.
        # If present, validate them. Dedicated dataset configs may choose to
        # make their presence mandatory separately.
        if label_column not in columns:
            continue

        values = np.asarray(
            dataframe[label_column]
        )

        if not np.all(np.isfinite(values)):
            errors.append(
                f"{label_column} contains non-finite values."
            )

        unique_values = set(
            np.unique(values).tolist()
        )

        if feature in {
            "point_anomaly",
            "collective_anomaly",
            "contextual_anomaly",
        }:
            if not unique_values.issubset({0, 1}):
                errors.append(
                    f"{label_column} must be binary; "
                    f"got {sorted(unique_values)}."
                )

        if feature in {
            "mean_shift",
            "variance_shift",
            "trend_shift",
        }:
            if any(
                int(value) != value or value < 0
                for value in unique_values
            ):
                errors.append(
                    f"{label_column} must contain non-negative integer "
                    "regime labels."
                )

    return {
        "gate": "GATE_8_LABEL_INTEGRITY",
        "passed": not errors,
        "details": errors,
    }


# ============================================================================
# CONFIG / PIPELINE CONVENIENCE
# ============================================================================

def promote_legacy_volatility_features(
    base_components: Sequence[str],
    enabled_features: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Move legacy volatility config entries into mathematical base components.

    Example
    -------
    base_components = ["arima", "multiple_seasonality"]
    enabled_features = ["garch", "linear_trend", "mean_shift"]

    returns:
        base_components =
            ["arima", "multiple_seasonality", "garch"]

        overlay_features =
            ["linear_trend", "mean_shift"]
    """

    promoted_bases = list(
        base_components
    )

    overlays = []

    for feature in enabled_features:
        if feature in VOLATILITY_BASE_SERIES:
            promoted_bases.append(
                feature
            )
        else:
            overlays.append(
                feature
            )

    return (
        _unique_preserve_order(
            promoted_bases
        ),
        _unique_preserve_order(
            overlays
        ),
    )


# ============================================================================
# SMALL BUILT-IN SELF TEST
# ============================================================================

def _self_test() -> None:
    valid_cases = [
        (
            ["ar", "garch"],
            [],
        ),
        (
            ["arima", "garch", "multiple_seasonality"],
            ["linear_trend", "mean_shift", "point_anomaly"],
        ),
        (
            ["arfima", "garch", "single_seasonality"],
            ["linear_trend", "trend_shift", "collective_anomaly"],
        ),
        (
            ["single_seasonality"],
            ["contextual_anomaly"],
        ),
    ]

    for bases, features in valid_cases:
        report = validate_requested_combination(
            bases,
            features
        )

        assert report.valid, (
            bases,
            features,
            report.errors,
        )

    invalid_cases = [
        (
            ["arima", "single_seasonality"],
            [],
        ),
        (
            ["arfima", "ar"],
            [],
        ),
        (
            ["garch"],
            ["variance_shift"],
        ),
        (
            ["ar"],
            ["contextual_anomaly"],
        ),
        (
            ["ar"],
            ["trend_shift"],
        ),
        (
            ["ar"],
            ["linear_trend", "quadratic_trend"],
        ),
        (
            ["ar"],
            ["point_anomaly", "collective_anomaly"],
        ),
    ]

    for bases, features in invalid_cases:
        report = validate_requested_combination(
            bases,
            features
        )

        assert not report.valid, (
            bases,
            features,
        )


if __name__ == "__main__":
    _self_test()

    print(
        "PASS — canonical BeTiSe rule self-test"
    )
