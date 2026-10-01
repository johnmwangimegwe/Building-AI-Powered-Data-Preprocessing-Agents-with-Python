"""
Module purpose:

Standardise the three datetime columns of the admission table - ``intime``,
``outtime`` and ``charttime`` - into a single unambiguous representation.

Why this module is format-aware
-------------------------------
The naive approach is ``pd.to_datetime(series, errors="coerce")``. It runs without
raising, produces a tidy datetime column, and is **wrong**. Given the value
``10/02/2026``, pandas applies a month-first reading and returns the 2nd of October.
The hospital system that wrote it meant the 10th of February. Nothing in the output
signals the error: the column parses cleanly, the dtype is correct, and the corrupted
timestamps propagate silently into every downstream length-of-stay calculation.

In the reference dataset this failure mode shifts affected admissions by up to eight
months and produces negative lengths of stay. Because a negative stay is impossible,
it is detectable - which is exactly why this module reports it rather than quietly
repairing it.

The fix is to classify each value by its **surface form** with a regular expression,
then parse it with the one format that matches. Ambiguity is resolved by the
spelling, not by a guess.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["standardise_datetimes", "parse_datetime_series", "reconcile_admission_windows"]


#: Ordered (regex, strptime format, human label) triples. Order matters only in that
#: the first match wins; the patterns are mutually exclusive by construction.
_FORMAT_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"), "%Y-%m-%d %H:%M", "ISO (YYYY-MM-DD HH:MM)"),
    (re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}"), "%Y/%m/%d %H:%M", "Slash ISO (YYYY/MM/DD HH:MM)"),
    (re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}"), "%d/%m/%Y %H:%M", "Day-first (DD/MM/YYYY HH:MM)"),
    (re.compile(r"^[A-Za-z]{3} \d{2} \d{4} \d{2}:\d{2}"), "%b %d %Y %H:%M", "Month name (Mon DD YYYY HH:MM)"),
)


def parse_datetime_series(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """Parse a column of mixed-format datetime strings correctly.

    Each value is matched against the known surface forms and parsed with the
    corresponding ``strptime`` format. Values matching nothing are parsed with a
    tolerant fallback and counted separately so that they can be reviewed.

    Parameters
    ----------
    series:
        Column of datetime strings.

    Returns
    -------
    (pandas.Series, dict)
        Parsed ``datetime64[ns]`` series aligned to the input index, and an audit
        dictionary with ``formats_detected`` (label -> count), ``n_formats``,
        ``fallback_parsed``, ``unparseable`` and ``naive_parse_disagreements``.

    Notes
    -----
    ``naive_parse_disagreements`` is the number of cells where this function and the
    naive ``pd.to_datetime`` default disagree. It is the number that makes the risk
    concrete on a conference slide: it is not zero.

    Example
    -------
    >>> parsed, audit = parse_datetime_series(pd.Series(["10/02/2026 08:00"]))
    >>> parsed.iloc[0].strftime("%Y-%m-%d")
    '2026-02-10'
    >>> audit["naive_parse_disagreements"]
    1
    """
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    counts: dict[str, int] = {}
    matched_any = pd.Series(False, index=series.index)

    for pattern, strptime_format, label in _FORMAT_RULES:
        candidate = text.notna() & text.str.match(pattern.pattern, na=False) & ~matched_any
        n_matched = int(candidate.sum())
        if n_matched == 0:
            continue
        counts[label] = n_matched
        parsed.loc[candidate] = pd.to_datetime(
            text.loc[candidate], format=strptime_format, errors="coerce"
        )
        matched_any |= candidate

    leftover = text.notna() & ~matched_any
    fallback_parsed = 0
    if leftover.any():
        fallback = pd.to_datetime(text.loc[leftover], errors="coerce", format="mixed", dayfirst=True)
        parsed.loc[leftover] = fallback
        fallback_parsed = int(fallback.notna().sum())
        counts["Unrecognised (tolerant fallback)"] = int(leftover.sum())

    # Quantify what the naive approach would have cost.
    naive = pd.to_datetime(text, errors="coerce", format="mixed")
    comparable = parsed.notna() & naive.notna()
    disagreements = int((parsed[comparable] != naive[comparable]).sum())

    audit: dict[str, Any] = {
        "formats_detected": counts,
        "n_formats": len(counts),
        "fallback_parsed": fallback_parsed,
        "unparseable": int(text.notna().sum() - parsed.notna().sum()),
        "naive_parse_disagreements": disagreements,
    }
    return parsed, audit


def reconcile_admission_windows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Force one admission window per ``admission_id`` and clamp chart times into it.

    Every row of an admission repeats that admission's ``intime`` and ``outtime``. After
    correct parsing these should agree exactly; any residual disagreement indicates a
    genuinely corrupt cell, which is resolved by majority vote across the admission's
    rows. Observations charted outside the reconciled window are recorded and clamped
    to the boundary rather than deleted, because an off-by-minutes chart time is a
    clocking artefact, not a reason to discard a clinical measurement.

    Parameters
    ----------
    frame:
        Table with parsed ``intime``, ``outtime``, ``charttime`` and ``admission_id``.

    Returns
    -------
    (pandas.DataFrame, dict)
        Table with reconciled windows plus a ``length_of_stay_hours`` column, and an
        audit dictionary.
    """
    working = frame.copy()

    def _mode(values: pd.Series) -> Any:
        non_null = values.dropna()
        if non_null.empty:
            return pd.NaT
        return non_null.mode().iloc[0]

    windows = working.groupby("admission_id").agg(
        _intime=("intime", _mode), _outtime=("outtime", _mode)
    )
    disagreeing = int(
        (working.groupby("admission_id")["intime"].nunique(dropna=True) > 1).sum()
        + (working.groupby("admission_id")["outtime"].nunique(dropna=True) > 1).sum()
    )

    working = working.merge(windows, left_on="admission_id", right_index=True, how="left")
    working["intime"] = working["_intime"]
    working["outtime"] = working["_outtime"]
    working = working.drop(columns=["_intime", "_outtime"])

    inverted = int((working["outtime"] <= working["intime"]).sum())

    before_window = working["charttime"] < working["intime"]
    after_window = working["charttime"] > working["outtime"]
    n_before, n_after = int(before_window.sum()), int(after_window.sum())
    working.loc[before_window, "charttime"] = working.loc[before_window, "intime"]
    working.loc[after_window, "charttime"] = working.loc[after_window, "outtime"]

    length_of_stay = (working["outtime"] - working["intime"]).dt.total_seconds() / 3600.0
    working["length_of_stay_hours"] = length_of_stay.round(2)

    audit: dict[str, Any] = {
        "admissions": int(working["admission_id"].nunique()),
        "admissions_with_conflicting_windows": disagreeing,
        "inverted_windows": inverted,
        "charttime_clamped_to_intime": n_before,
        "charttime_clamped_to_outtime": n_after,
        "length_of_stay_hours": {
            "min": float(np.round(np.nanmin(length_of_stay), 2)),
            "median": float(np.round(np.nanmedian(length_of_stay), 2)),
            "max": float(np.round(np.nanmax(length_of_stay), 2)),
        },
    }
    return working, audit


def standardise_datetimes(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = ("intime", "outtime", "charttime"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse every datetime column, then reconcile the admission windows.

    Parameters
    ----------
    frame:
        Raw table with datetime columns stored as strings.
    columns:
        Datetime columns to standardise.

    Returns
    -------
    (pandas.DataFrame, dict)
        Table with true ``datetime64[ns]`` columns and a ``length_of_stay_hours``
        column, plus a per-column audit and the window-reconciliation audit.

    Example
    -------
    >>> cleaned, audit = standardise_datetimes(raw_frame)
    >>> audit["charttime"]["n_formats"]
    4
    """
    working = frame.copy()
    audit: dict[str, Any] = {}

    for column in columns:
        if column not in working.columns:
            continue
        parsed, column_audit = parse_datetime_series(working[column])
        working[column] = parsed
        audit[column] = column_audit

    working, window_audit = reconcile_admission_windows(working)
    audit["window_reconciliation"] = window_audit
    audit["formats_before"] = max(
        (audit[column].get("n_formats", 0) for column in columns if column in audit), default=0
    )
    audit["formats_after"] = 1
    audit["total_naive_parse_disagreements"] = sum(
        audit[column].get("naive_parse_disagreements", 0) for column in columns if column in audit
    )
    return working, audit
