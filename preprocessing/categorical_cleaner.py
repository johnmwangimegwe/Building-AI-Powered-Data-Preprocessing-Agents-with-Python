"""
Module purpose:

Collapse free-text variation in categorical clinical fields onto a controlled
vocabulary.

``Emergency``, ``emergency``, ``ER`` and ``Emergency Admission`` are one admission
type recorded by four people. Left alone they become four one-hot columns, splitting
the statistical power of the cohort across spellings and making any model's
coefficients uninterpretable.

The mapping is **explicit**, not fuzzy. A controlled vocabulary lives in
``config.CANONICAL_CATEGORIES`` and anything not in it is reported as an unmapped
value rather than silently guessed. Fuzzy string matching is attractive here and
wrong: in a clinical setting, quietly mapping an unrecognised diagnosis onto the
nearest known one invents a fact about a patient.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import CANONICAL_CATEGORIES

__all__ = ["standardise_categories"]


def standardise_categories(
    frame: pd.DataFrame,
    vocabularies: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map categorical columns onto their controlled vocabularies.

    Parameters
    ----------
    frame:
        Table containing categorical columns.
    vocabularies:
        ``{column: {lowercase_variant: canonical_value}}``. Defaults to
        ``config.CANONICAL_CATEGORIES``.

    Returns
    -------
    (pandas.DataFrame, dict)
        Table with standardised categories, and a per-column audit giving
        ``distinct_before``, ``distinct_after``, ``values_before``, ``values_after``,
        ``cells_changed`` and ``unmapped_values``.

    Notes
    -----
    Unmapped values are left **exactly as they were** and listed in the audit. A
    reviewer extends the vocabulary and re-runs; nothing is lost in the meantime.

    Example
    -------
    >>> standardised, audit = standardise_categories(frame)
    >>> audit["admission_type"]["distinct_before"], audit["admission_type"]["distinct_after"]
    (9, 4)
    """
    working = frame.copy()
    vocabularies = vocabularies or CANONICAL_CATEGORIES
    audit: dict[str, Any] = {}

    for column, mapping in vocabularies.items():
        if column not in working.columns:
            continue

        original = working[column].astype("string")
        normalised_key = original.str.strip().str.lower()
        mapped = normalised_key.map(mapping)

        unmapped_mask = mapped.isna() & original.notna()
        unmapped_values = sorted(original[unmapped_mask].dropna().unique().tolist())

        result = mapped.where(~unmapped_mask, original)
        cells_changed = int((result.fillna("<NA>") != original.fillna("<NA>")).sum())

        working[column] = result

        audit[column] = {
            "distinct_before": int(original.dropna().nunique()),
            "distinct_after": int(result.dropna().nunique()),
            "values_before": sorted(original.dropna().unique().tolist()),
            "values_after": sorted(result.dropna().unique().tolist()),
            "cells_changed": cells_changed,
            "unmapped_values": unmapped_values,
        }

    return working, audit
