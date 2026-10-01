"""
Module purpose:

Coerce mixed-unit clinical columns into a single canonical unit per channel.

Three unit problems occur in the raw admission table, each requiring different
treatment:

``temperature``
    Values are written as ``"37.5 C"`` or ``"98.6 F"``. Conversion is required, using

    $$ T_{\\text{C}} = (T_{\\text{F}} - 32) \\times \\frac{5}{9} $$

    A unit-blind ``float()`` cast would place both populations on one axis and
    produce a spurious bimodal distribution centred near 37 and 99 - a shape that
    looks like two patient cohorts and is in fact one cohort and two thermometers.

``sbp`` / ``mbp``
    Values may carry a ``"mmHg"`` suffix. The unit is already canonical, so the
    suffix is stripped and the number kept unchanged.

``glucose``
    Values may be expressed in mmol/L rather than mg/dL. These differ by a molar mass
    factor, so the numbers must be scaled:

    $$ G_{\\text{mg/dL}} = G_{\\text{mmol/L}} \\times 18.0182 $$

When a value is a bare number with no unit token, its unit is **inferred from the
plausible range** rather than assumed. A bare ``99.1`` in the temperature column is
far outside the human Celsius range but sits squarely in the Fahrenheit range, so it
is treated as Fahrenheit and the inference is recorded in the audit trail.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["convert_units", "fahrenheit_to_celsius", "parse_measurement"]

#: Molar conversion factor between mmol/L and mg/dL for glucose.
MMOL_PER_L_TO_MG_PER_DL: float = 18.0182

_MEASUREMENT_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z/%°]*)\s*$")

#: Bare-number ranges used to infer a missing unit token.
_CELSIUS_RANGE = (25.0, 45.0)  # retained for documentation of the inference bands
_FAHRENHEIT_RANGE = (77.0, 113.0)
_GLUCOSE_MMOL_RANGE = (1.0, 35.0)


def fahrenheit_to_celsius(value: float) -> float:
    """Convert a Fahrenheit temperature to Celsius.

    Parameters
    ----------
    value:
        Temperature in degrees Fahrenheit.

    Returns
    -------
    float
        Temperature in degrees Celsius.

    Example
    -------
    >>> round(fahrenheit_to_celsius(98.6), 1)
    37.0
    """
    return (value - 32.0) * 5.0 / 9.0


def parse_measurement(value: Any) -> tuple[float | None, str]:
    """Split a raw cell into its numeric magnitude and its unit token.

    Parameters
    ----------
    value:
        Raw cell contents - a float, an int, or a string such as ``"98.6 F"``.

    Returns
    -------
    (float | None, str)
        Magnitude (``None`` when the cell holds no parseable number) and the unit
        token in lower case (``""`` when the cell is a bare number).

    Example
    -------
    >>> parse_measurement("6.6 mmol/L")
    (6.6, 'mmol/l')
    >>> parse_measurement(120.0)
    (120.0, '')
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, ""
    match = _MEASUREMENT_PATTERN.match(str(value))
    if not match:
        return None, ""
    return float(match.group(1)), match.group(2).strip().lower()


def _convert_temperature(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Normalise a temperature column to Celsius."""
    counters = {
        "celsius_explicit": 0,
        "fahrenheit_explicit": 0,
        "inferred_fahrenheit": 0,
        "unitless_kept_as_celsius": 0,
        "unparseable": 0,
    }
    output: list[float] = []

    for raw in series:
        magnitude, unit = parse_measurement(raw)
        if magnitude is None:
            counters["unparseable"] += int(pd.notna(raw))
            output.append(np.nan)
            continue
        if unit.startswith("f"):
            counters["fahrenheit_explicit"] += 1
            output.append(fahrenheit_to_celsius(magnitude))
        elif unit.startswith("c"):
            counters["celsius_explicit"] += 1
            output.append(magnitude)
        elif _FAHRENHEIT_RANGE[0] <= magnitude <= _FAHRENHEIT_RANGE[1]:
            counters["inferred_fahrenheit"] += 1
            output.append(fahrenheit_to_celsius(magnitude))
        else:
            counters["unitless_kept_as_celsius"] += 1
            output.append(magnitude)

    return pd.Series(output, index=series.index, dtype="float64").round(2), counters


def _convert_glucose(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Normalise a glucose column to mg/dL."""
    counters = {"mg_per_dl_explicit": 0, "mmol_per_l_explicit": 0, "inferred_mmol_per_l": 0, "bare_mg_per_dl": 0, "unparseable": 0}
    output: list[float] = []

    for raw in series:
        magnitude, unit = parse_measurement(raw)
        if magnitude is None:
            counters["unparseable"] += int(pd.notna(raw))
            output.append(np.nan)
            continue
        if "mmol" in unit:
            counters["mmol_per_l_explicit"] += 1
            output.append(magnitude * MMOL_PER_L_TO_MG_PER_DL)
        elif "mg" in unit:
            counters["mg_per_dl_explicit"] += 1
            output.append(magnitude)
        elif _GLUCOSE_MMOL_RANGE[0] <= magnitude <= _GLUCOSE_MMOL_RANGE[1]:
            counters["inferred_mmol_per_l"] += 1
            output.append(magnitude * MMOL_PER_L_TO_MG_PER_DL)
        else:
            counters["bare_mg_per_dl"] += 1
            output.append(magnitude)

    return pd.Series(output, index=series.index, dtype="float64").round(2), counters


def _strip_suffix(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Strip a redundant unit token, keeping the magnitude unchanged."""
    counters = {"suffix_stripped": 0, "bare_number": 0, "unparseable": 0}
    output: list[float] = []

    for raw in series:
        magnitude, unit = parse_measurement(raw)
        if magnitude is None:
            counters["unparseable"] += int(pd.notna(raw))
            output.append(np.nan)
            continue
        if unit:
            counters["suffix_stripped"] += 1
        else:
            counters["bare_number"] += 1
        output.append(magnitude)

    return pd.Series(output, index=series.index, dtype="float64").round(2), counters


#: Column -> (handler, canonical unit label).
_HANDLERS: dict[str, tuple[Any, str]] = {
    "temperature": (_convert_temperature, "degrees Celsius"),
    "glucose": (_convert_glucose, "mg/dL"),
    "sbp": (_strip_suffix, "mmHg"),
    "dbp": (_strip_suffix, "mmHg"),
    "mbp": (_strip_suffix, "mmHg"),
    "heart_rate": (_strip_suffix, "bpm"),
    "spo2": (_strip_suffix, "%"),
    "resp_rate": (_strip_suffix, "breaths/min"),
    "creatinine": (_strip_suffix, "mg/dL"),
    "wbc_count": (_strip_suffix, "10^9/L"),
    "platelet_count": (_strip_suffix, "10^9/L"),
    "age": (_strip_suffix, "years"),
    "sofa_score": (_strip_suffix, "points"),
}


def convert_units(frame: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalise every mixed-unit column to its canonical unit.

    Parameters
    ----------
    frame:
        Table whose measurement columns may hold unit-suffixed strings.
    columns:
        Subset of columns to convert. Defaults to every column this module knows about
        and that is present in ``frame``.

    Returns
    -------
    (pandas.DataFrame, dict)
        Table with float measurement columns, and a per-column audit recording how
        many cells took each conversion path plus the before/after distribution
        summary.

    Example
    -------
    >>> converted, audit = convert_units(frame)
    >>> audit["temperature"]["converted_from_fahrenheit"]
    404
    >>> audit["temperature"]["after"]["max"] < 43.0
    True
    """
    working = frame.copy()
    columns = columns or [column for column in _HANDLERS if column in working.columns]
    audit: dict[str, Any] = {}

    for column in columns:
        if column not in working.columns or column not in _HANDLERS:
            continue
        handler, canonical_unit = _HANDLERS[column]

        raw_numeric = pd.to_numeric(
            working[column].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False), errors="coerce"
        )
        converted, counters = handler(working[column])
        working[column] = converted

        column_audit: dict[str, Any] = {
            "canonical_unit": canonical_unit,
            "paths": counters,
            "before": _distribution(raw_numeric),
            "after": _distribution(converted),
        }
        if column == "temperature":
            column_audit["converted_from_fahrenheit"] = (
                counters["fahrenheit_explicit"] + counters["inferred_fahrenheit"]
            )
        if column == "glucose":
            column_audit["converted_from_mmol_per_l"] = (
                counters["mmol_per_l_explicit"] + counters["inferred_mmol_per_l"]
            )
        audit[column] = column_audit

    return working, audit


def _distribution(series: pd.Series) -> dict[str, float | int]:
    """Summarise a numeric series for the before/after comparison table."""
    valid = series.dropna()
    if valid.empty:
        return {"count": 0, "min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "count": int(len(valid)),
        "min": float(np.round(valid.min(), 2)),
        "mean": float(np.round(valid.mean(), 2)),
        "max": float(np.round(valid.max(), 2)),
    }
