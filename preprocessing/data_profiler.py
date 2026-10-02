"""
Module purpose:

Profile a raw clinical table *before* any transformation is attempted.

An agent cannot plan a  cleaning strategy it cannot describe. This module is the
agent's perception step: it converts an opaque CSV into a compact, machine-readable
dictionary that fits comfortably inside an LLM context window - typically under two
kilobytes for a thousand-row table - which is precisely the property that makes
agent-guided preprocessing viable where pasting the data into a chat window is not.

The profile is deliberately *diagnostic rather than prescriptive*. It reports what
is present (four datetime spellings, 40% Fahrenheit, 100 duplicated rows); deciding
what to do about it belongs to :mod:`agents.preprocessing_agent`.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from config import PLAUSIBLE_RANGES, VITAL_COLUMNS

__all__ = ["profile_data", "detect_date_formats", "detect_unit_inconsistencies", "format_profile"]


#: Regular expressions identifying each datetime spelling the generator emits, plus a
#: catch-all. Detection is done on the *text*, never by trial-parsing, because
#: ``10/02/2026`` parses successfully under two different interpretations and only the
#: surface form tells us which one the hospital system meant.
_DATE_PATTERNS: dict[str, re.Pattern[str]] = {
    "ISO (YYYY-MM-DD HH:MM)": re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"),
    "Slash ISO (YYYY/MM/DD HH:MM)": re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}"),
    "Day-first (DD/MM/YYYY HH:MM)": re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}"),
    "Month name (Mon DD YYYY HH:MM)": re.compile(r"^[A-Za-z]{3} \d{2} \d{4} \d{2}:\d{2}"),
}

_UNIT_SUFFIX_PATTERN = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*([A-Za-z/%°]+)\s*$")


def detect_date_formats(series: pd.Series) -> dict[str, int]:
    """Count how many values in a datetime-like column use each spelling.

    Parameters
    ----------
    series:
        A column of datetime *strings* (not parsed timestamps).

    Returns
    -------
    dict[str, int]
        Spelling name -> occurrence count, including an ``"Unrecognised"`` bucket.
        Buckets with a zero count are omitted.

    Example
    -------
    >>> import pandas as pd
    >>> detect_date_formats(pd.Series(["2026-01-01 08:00", "01/02/2026 09:00"]))
    {'ISO (YYYY-MM-DD HH:MM)': 1, 'Day-first (DD/MM/YYYY HH:MM)': 1}
    """
    counts = {name: 0 for name in _DATE_PATTERNS}
    counts["Unrecognised"] = 0
    for value in series.dropna().astype(str):
        for name, pattern in _DATE_PATTERNS.items():
            if pattern.match(value):
                counts[name] += 1
                break
        else:
            counts["Unrecognised"] += 1
    return {name: count for name, count in counts.items() if count > 0}


def detect_unit_inconsistencies(frame: pd.DataFrame, columns: list[str] | None = None) -> dict[str, dict[str, int]]:
    """Find numeric columns whose values carry mixed textual unit suffixes.

    Parameters
    ----------
    frame:
        The raw table.
    columns:
        Columns to inspect. Defaults to the bedside vital-sign channels.

    Returns
    -------
    dict[str, dict[str, int]]
        Column -> {unit token or ``"<bare number>"`` -> count}. Only columns showing
        more than one distinct token are returned, because a column where every value
        says ``mmHg`` is consistent, merely verbose.
    """
    columns = columns or VITAL_COLUMNS
    findings: dict[str, dict[str, int]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        tokens: dict[str, int] = {}
        for value in frame[column].dropna():
            text = str(value).strip()
            match = _UNIT_SUFFIX_PATTERN.match(text)
            key = match.group(1) if match else "<bare number>"
            tokens[key] = tokens.get(key, 0) + 1
        if len(tokens) > 1:
            findings[column] = dict(sorted(tokens.items(), key=lambda item: -item[1]))
    return findings


def _numeric_view(series: pd.Series) -> pd.Series:
    """Coerce a possibly unit-suffixed column to floats, ignoring the suffix."""
    text = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(text, errors="coerce")


def profile_data(frame: pd.DataFrame, datetime_columns: tuple[str, ...] = ("intime", "outtime", "charttime")) -> dict[str, Any]:
    """Build a complete data-quality profile of a raw clinical table.

    Parameters
    ----------
    frame:
        The raw table, loaded with ``dtype=str`` or with mixed types - both work.
    datetime_columns:
        Columns to run datetime-spelling detection over.

    Returns
    -------
    dict
        A JSON-serialisable profile with keys ``n_rows``, ``n_columns``, ``columns``,
        ``dtypes``, ``missing``, ``missing_pct``, ``duplicates``, ``date_formats``,
        ``unit_inconsistencies``, ``implausible_values``, ``categorical_variants`` and
        ``numeric_summary``.

    Example
    -------
    >>> profile = profile_data(raw_frame)
    >>> profile["duplicates"]["exact_duplicate_rows"]
    100
    """
    profile: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }

    missing = frame.isna().sum()
    profile["missing"] = {column: int(count) for column, count in missing.items()}
    profile["missing_pct"] = {
        column: round(100.0 * count / max(len(frame), 1), 2) for column, count in profile["missing"].items()
    }

    profile["duplicates"] = {
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "rows_involved_in_duplication": int(frame.duplicated(keep=False).sum()),
    }

    profile["date_formats"] = {
        column: detect_date_formats(frame[column]) for column in datetime_columns if column in frame.columns
    }

    profile["unit_inconsistencies"] = detect_unit_inconsistencies(frame)

    implausible: dict[str, dict[str, Any]] = {}
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in frame.columns:
            continue
        numeric = _numeric_view(frame[column])
        out_of_range = int(((numeric < low) | (numeric > high)).sum())
        non_numeric = int((frame[column].notna() & numeric.isna()).sum())
        if out_of_range or non_numeric:
            implausible[column] = {
                "out_of_range": out_of_range,
                "unparseable_text": non_numeric,
                "allowed_range": [low, high],
            }
    profile["implausible_values"] = implausible

    categorical_variants: dict[str, dict[str, int]] = {}
    for column in frame.columns:
        if column in PLAUSIBLE_RANGES or column in datetime_columns:
            continue
        values = frame[column].dropna().astype(str)
        if values.empty or values.nunique() > 25:
            continue
        canonical_groups = values.str.strip().str.lower().nunique()
        if values.nunique() > canonical_groups:
            categorical_variants[column] = {
                "distinct_raw_values": int(values.nunique()),
                "distinct_after_casefold": int(canonical_groups),
            }
    profile["categorical_variants"] = categorical_variants

    numeric_summary: dict[str, dict[str, float]] = {}
    for column in VITAL_COLUMNS:
        if column not in frame.columns:
            continue
        numeric = _numeric_view(frame[column]).dropna()
        if numeric.empty:
            continue
        numeric_summary[column] = {
            "min": float(np.round(numeric.min(), 2)),
            "median": float(np.round(numeric.median(), 2)),
            "max": float(np.round(numeric.max(), 2)),
            "observed": int(len(numeric)),
        }
    profile["numeric_summary"] = numeric_summary

    return profile


def format_profile(profile: dict[str, Any]) -> str:
    """Render a profile as the compact text block handed to the planning LLM.

    Keeping this representation small is the whole point of the profiling step: the
    agent reasons over roughly 1.5 KB of summary instead of a 160 KB CSV.

    Parameters
    ----------
    profile:
        Output of :func:`profile_data`.

    Returns
    -------
    str
        Human- and LLM-readable multi-line summary.
    """
    lines = [
        "DATASET PROFILE",
        f"  Rows: {profile['n_rows']:,}    Columns: {profile['n_columns']}",
        f"  Exact duplicate rows: {profile['duplicates']['exact_duplicate_rows']:,}",
        "",
        "  Missing values (top 10):",
    ]
    ranked = sorted(profile["missing_pct"].items(), key=lambda item: -item[1])[:10]
    for column, pct in ranked:
        if pct > 0:
            lines.append(f"    {column:<18} {profile['missing'][column]:>5,} ({pct:.1f}%)")

    lines.append("")
    lines.append("  Datetime spellings detected:")
    for column, spellings in profile["date_formats"].items():
        rendered = ", ".join(f"{name.split(' (')[0]}={count}" for name, count in spellings.items())
        lines.append(f"    {column:<12} {len(spellings)} format(s): {rendered}")

    if profile["unit_inconsistencies"]:
        lines.append("")
        lines.append("  Mixed unit tokens:")
        for column, tokens in profile["unit_inconsistencies"].items():
            rendered = ", ".join(f"{token}={count}" for token, count in tokens.items())
            lines.append(f"    {column:<18} {rendered}")

    if profile["implausible_values"]:
        lines.append("")
        lines.append("  Implausible / unparseable cells:")
        for column, detail in profile["implausible_values"].items():
            lines.append(
                f"    {column:<18} out_of_range={detail['out_of_range']:>4}  "
                f"unparseable={detail['unparseable_text']:>4}  allowed={detail['allowed_range']}"
            )

    if profile["categorical_variants"]:
        lines.append("")
        lines.append("  Categorical spelling variation:")
        for column, detail in profile["categorical_variants"].items():
            lines.append(
                f"    {column:<18} {detail['distinct_raw_values']} raw -> "
                f"{detail['distinct_after_casefold']} after case-folding"
            )

    return "\n".join(lines)
