"""
Module purpose:

Detect and remove duplicated clinical observations.

Duplication in hospital exports is rarely malicious; it arises when a monitor feed and
a nurse's manual entry both land in the warehouse, or when an extract is re-run and
appended. The effect on modelling is nonetheless severe: a duplicated observation
doubles that patient-hour's weight in any loss function and biases every
population-level statistic the dataset is used to compute.

Two notions of duplication are supported:

``exact``
    Every field identical. Safe to drop unconditionally.
``clinical``
    Same admission and same charted minute. This catches the monitor-plus-nurse case
    where a free-text field differs but the observation is the same event. It is
    offered but **not** applied by default, because collapsing it discards a real
    disagreement between two sources that a clinician may want to see.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["count_duplicates", "remove_duplicates"]


def count_duplicates(frame: pd.DataFrame, subset: list[str] | None = None) -> dict[str, int]:
    """Count duplicated rows without modifying the table.

    Parameters
    ----------
    frame:
        Table to inspect.
    subset:
        Optional column subset defining the duplicate key. ``None`` means "all columns".

    Returns
    -------
    dict[str, int]
        ``redundant_rows`` (copies beyond the first) and ``rows_involved``
        (every row that participates in a duplicate group, originals included).
    """
    return {
        "redundant_rows": int(frame.duplicated(subset=subset).sum()),
        "rows_involved": int(frame.duplicated(subset=subset, keep=False).sum()),
    }


def remove_duplicates(
    frame: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop duplicated rows and return a full audit record.

    Parameters
    ----------
    frame:
        Table to deduplicate.
    subset:
        Columns defining the duplicate key; ``None`` compares entire rows.
    keep:
        Which member of each duplicate group survives - ``"first"`` or ``"last"``.

    Returns
    -------
    (pandas.DataFrame, dict)
        The deduplicated table (index reset) and an audit dictionary containing
        ``rows_before``, ``rows_after``, ``duplicates_removed``,
        ``duplicates_remaining`` and ``dropped_row_indices``.

    Notes
    -----
    The original positional indices of every dropped row are recorded. This is the
    single most useful field in the whole audit trail: it lets a reviewer open the raw
    CSV, jump to those line numbers, and confirm by eye that the agent removed what it
    claimed to remove. The reference workflow this project follows treats that
    cross-check as the definition of a trustworthy cleaning run.

    Example
    -------
    >>> deduplicated, audit = remove_duplicates(raw_frame)
    >>> audit["duplicates_removed"]
    100
    >>> audit["duplicates_remaining"]
    0
    """
    before = count_duplicates(frame, subset=subset)
    duplicate_mask = frame.duplicated(subset=subset, keep=keep)
    dropped_indices = [int(position) for position in frame.index[duplicate_mask]]

    cleaned = frame.loc[~duplicate_mask].reset_index(drop=True)
    after = count_duplicates(cleaned, subset=subset)

    audit: dict[str, Any] = {
        "strategy": "exact_row_match" if subset is None else f"key_match::{'+'.join(subset)}",
        "keep": keep,
        "rows_before": int(len(frame)),
        "rows_after": int(len(cleaned)),
        "duplicates_removed": int(before["redundant_rows"]),
        "duplicates_remaining": int(after["redundant_rows"]),
        "rows_involved_before": int(before["rows_involved"]),
        "dropped_row_indices": dropped_indices,
    }
    return cleaned, audit
