"""
Module purpose:

Reject clinically impossible measurements and record why each one was rejected.

A heart rate of 480 bpm and an oxygen saturation of 140% are not outliers to be
winsorised; they are sensor faults or transcription errors, and the honest
representation of a faulty sensor reading is *absent data*, not a plausible-looking
number.

The critical design decision is that rejection **converts the cell to missing rather
than dropping the row**. In an irregularly sampled clinical record, a row carries
several channels; discarding the row because one channel is corrupt throws away the
valid measurements recorded at the same instant. Every rejected cell is logged with
its admission, chart time, channel and offending value, so the decision is fully
reversible from the audit trail.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import PLAUSIBLE_RANGES

__all__ = ["validate_ranges"]


def validate_ranges(
    frame: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
    max_logged_examples: int = 200,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Blank out-of-range measurements and produce a rejection log.

    Parameters
    ----------
    frame:
        Table with numeric measurement columns (run unit conversion first, so that a
        Fahrenheit temperature is not rejected for being "too hot").
    ranges:
        ``{column: (minimum, maximum)}``, both bounds inclusive. Defaults to
        ``config.PLAUSIBLE_RANGES``.
    max_logged_examples:
        Cap on the number of individual rejections written to the log, to keep the
        JSON report a readable size. The per-column *counts* are always complete.

    Returns
    -------
    (pandas.DataFrame, dict)
        Table with implausible cells set to ``NaN``, plus an audit containing
        ``total_cells_rejected``, ``by_column`` and a ``rejections`` sample log.

    Example
    -------
    >>> validated, audit = validate_ranges(frame)
    >>> audit["by_column"]["spo2"]["rejected"]
    48
    """
    working = frame.copy()
    ranges = ranges or PLAUSIBLE_RANGES

    by_column: dict[str, Any] = {}
    rejections: list[dict[str, Any]] = []
    total = 0

    for column, (low, high) in ranges.items():
        if column not in working.columns:
            continue
        numeric = pd.to_numeric(working[column], errors="coerce")

        unparseable = working[column].notna() & numeric.isna()
        out_of_range = numeric.notna() & ((numeric < low) | (numeric > high))
        reject_mask = unparseable | out_of_range

        n_rejected = int(reject_mask.sum())
        total += n_rejected

        by_column[column] = {
            "allowed_range": [low, high],
            "rejected": n_rejected,
            "out_of_range": int(out_of_range.sum()),
            "unparseable_text": int(unparseable.sum()),
            "observed_before": int(working[column].notna().sum()),
            "observed_after": int((numeric.notna() & ~reject_mask).sum()),
        }

        for position in working.index[reject_mask][: max(0, max_logged_examples - len(rejections))]:
            rejections.append(
                {
                    "row_index": int(position),
                    "admission_id": str(working.at[position, "admission_id"])
                    if "admission_id" in working.columns
                    else None,
                    "charttime": str(working.at[position, "charttime"])
                    if "charttime" in working.columns
                    else None,
                    "column": column,
                    "value": str(working.at[position, column]),
                    "reason": "unparseable_text" if bool(unparseable.at[position]) else "out_of_physiological_range",
                }
            )

        working[column] = numeric.where(~reject_mask)

    audit: dict[str, Any] = {
        "total_cells_rejected": total,
        "by_column": by_column,
        "rejections_logged": len(rejections),
        "rejections_truncated": total > len(rejections),
        "rejections": rejections,
        "policy": "cell_set_to_missing__row_retained",
    }
    return working, audit
