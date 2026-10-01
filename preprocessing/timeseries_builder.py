"""
Module purpose:

Transform the event-based admission table into a regular hourly clinical time series.

The raw table records *when somebody happened to measure something*. A model needs
*what the patient's state was at each hour*. Those are different objects, and the
conversion between them is where clinical missingness acquires its meaning.

Concretely, an admission spanning

$$ [t_{\\text{in}},\\, t_{\\text{out}}] $$

is expanded to the hourly grid

$$ \\mathcal{H} = \\{\\,\\lfloor t_{\\text{in}} \\rfloor_{1\\text{h}} + k\\,\\text{hours}
   \\;:\\; k = 0, 1, \\ldots, K\\,\\} $$

and each observation is assigned to the hour bin containing its chart time. Where
two observations fall in one bin they are averaged; where no observation falls in a
bin the cell is genuinely missing - not missing because somebody forgot to record it,
but missing because nothing was measured during that hour.

This is the step that makes the missing-data problem *visible*. The raw table looks
around 15% incomplete; after alignment the same data is roughly 85% incomplete,
because most patient-hours were never measured at all. Showing that jump is the most
persuasive moment in the pipeline: the missingness was always there, hidden by the
event-based layout.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import RESAMPLE_FREQ, TIMESERIES_CHANNELS

__all__ = ["build_hourly_timeseries"]

#: Admission-level attributes carried forward onto every hourly row.
_STATIC_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "age",
    "gender",
    "ethnicity",
    "admission_type",
    "diagnosis",
    "medication",
    "sofa_score",
    "length_of_stay_hours",
)


def build_hourly_timeseries(
    frame: pd.DataFrame,
    channels: list[str] | None = None,
    freq: str = RESAMPLE_FREQ,
    max_hours: int = 336,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand event rows into one row per admission-hour.

    Parameters
    ----------
    frame:
        Cleaned event table with parsed ``intime`` / ``outtime`` / ``charttime`` and
        numeric measurement columns.
    channels:
        Measurement channels to align. Defaults to ``config.TIMESERIES_CHANNELS``.
    freq:
        Pandas offset alias for the grid spacing. ``"1h"`` throughout this project.
    max_hours:
        Safety cap on the number of hours generated for a single admission (14 days).
        Guards against a corrupt discharge time inflating the output by millions of
        rows - a real failure mode when window reconciliation has not been run.

    Returns
    -------
    (pandas.DataFrame, dict)
        The hourly table - sorted by ``admission_id`` then ``hour``, with an
        ``hours_since_admission`` column - and an audit describing the expansion and
        the missingness before and after alignment.

    Example
    -------
    >>> hourly, audit = build_hourly_timeseries(cleaned)
    >>> audit["rows_before"], audit["rows_after"]
    (900, 8214)
    >>> audit["missing_pct_after"]["heart_rate"] > audit["missing_pct_before"]["heart_rate"]
    True
    """
    channels = channels or TIMESERIES_CHANNELS
    working = frame.dropna(subset=["admission_id", "charttime", "intime", "outtime"]).copy()

    missing_before = {
        channel: round(100.0 * working[channel].isna().mean(), 2)
        for channel in channels
        if channel in working.columns
    }

    working["hour"] = working["charttime"].dt.floor(freq)

    aggregation: dict[str, Any] = {channel: "mean" for channel in channels if channel in working.columns}
    for column in _STATIC_COLUMNS:
        if column in working.columns:
            aggregation[column] = "first"
    aggregation["intime"] = "first"
    aggregation["outtime"] = "first"

    binned = working.groupby(["admission_id", "hour"], as_index=False).agg(aggregation)

    collisions = int(len(working) - len(binned))

    frames: list[pd.DataFrame] = []
    truncated_admissions = 0

    for admission_id, group in binned.groupby("admission_id", sort=True):
        start = group["intime"].iloc[0].floor(freq)
        end = group["outtime"].iloc[0].ceil(freq)
        n_hours = int((end - start).total_seconds() // 3600) + 1
        if n_hours > max_hours:
            end = start + pd.Timedelta(hours=max_hours - 1)
            n_hours = max_hours
            truncated_admissions += 1
        if n_hours < 1:
            continue

        grid = pd.date_range(start=start, end=end, freq=freq)
        aligned = pd.DataFrame({"hour": grid})
        aligned["admission_id"] = admission_id
        aligned = aligned.merge(group.drop(columns=["intime", "outtime"]), on=["admission_id", "hour"], how="left")

        for column in _STATIC_COLUMNS:
            if column in aligned.columns:
                aligned[column] = aligned[column].ffill().bfill()

        aligned["hours_since_admission"] = np.arange(len(aligned), dtype=int)
        frames.append(aligned)

    hourly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["admission_id", "hour"])
    hourly = hourly.sort_values(["admission_id", "hour"]).reset_index(drop=True)

    missing_after = {
        channel: round(100.0 * hourly[channel].isna().mean(), 2)
        for channel in channels
        if channel in hourly.columns
    }

    hours_per_admission = hourly.groupby("admission_id").size()

    audit: dict[str, Any] = {
        "freq": freq,
        "rows_before": int(len(working)),
        "rows_after": int(len(hourly)),
        "expansion_factor": round(len(hourly) / max(len(working), 1), 2),
        "admissions": int(hourly["admission_id"].nunique()) if len(hourly) else 0,
        "observations_merged_into_shared_hour": collisions,
        "admissions_truncated_at_max_hours": truncated_admissions,
        "hours_per_admission": {
            "min": int(hours_per_admission.min()) if len(hourly) else 0,
            "median": float(hours_per_admission.median()) if len(hourly) else 0.0,
            "max": int(hours_per_admission.max()) if len(hourly) else 0,
        },
        "missing_pct_before": missing_before,
        "missing_pct_after": missing_after,
    }
    return hourly, audit
