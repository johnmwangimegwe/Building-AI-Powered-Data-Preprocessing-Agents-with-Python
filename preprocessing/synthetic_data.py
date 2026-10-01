"""
Module purpose:

Generate the synthetic *hospital admission* dataset that the preprocessing agent
operates on.

Why synthetic data? Real intensive-care records cannot be redistributed, and a
conference demonstration needs a dataset whose defects are *known in advance* so
that every claim the agent makes can be checked against ground truth. This module
therefore produces two artefacts simultaneously:

1. ``raw_hospital_admission_data.csv`` - the messy, analyst-facing file; and
2. a ``ground_truth`` manifest describing exactly how many defects of each kind were
   injected, which the validation agent later uses to score the pipeline.

Design of the data model
------------------------
Each row is **one clinical observation event**, not a patient and not an hour. A row
carries the admission window (``intime``, ``outtime``) plus the instant the
measurement was charted (``charttime``). Because nurses and monitors record
irregularly, ``charttime`` values are unevenly spaced inside the admission window.
This is the structural reason missing data exists at all, and it is what makes the
later hourly reconstruction meaningful rather than cosmetic.

Injected defects
----------------
The generator reproduces the defect classes named in the project brief:

* duplicated observation rows,
* four mutually inconsistent datetime spellings (including day-first ``DD/MM/YYYY``,
  which silently mis-parses under pandas defaults),
* temperatures recorded in a mixture of Celsius and Fahrenheit,
* numeric cells carrying a unit suffix (``"85.4 mmHg"``, ``"6.6 mmol/L"``),
* clinically impossible values,
* free-text variation in categorical fields, and
* missing values at channel-specific rates.

Critically, every injected defect *preserves the underlying true value where a true
value exists*. A Fahrenheit temperature is the real Celsius reading converted, not a
constant; a ``"mmHg"`` string is the real pressure with a suffix. That property is
what allows unit conversion to be scored for correctness rather than merely observed
to run.

Example usage
-------------
>>> from preprocessing.synthetic_data import generate_raw_dataset
>>> frame, truth = generate_raw_dataset(seed=2026)
>>> frame.shape
(1000, 24)
>>> truth["duplicate_rows_injected"] > 0
True

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CATEGORY_NOISE_RATE,
    DUPLICATE_RATE,
    FAHRENHEIT_RATE,
    INVALID_VALUE_RATE,
    MISSING_RATES,
    N_ADMISSIONS,
    N_RAW_RECORDS,
    OBSERVATIONS_PER_ADMISSION,
    RANDOM_SEED,
    RAW_COLUMNS,
    RAW_CSV,
    UNIT_SUFFIX_RATE,
)

__all__ = ["generate_raw_dataset", "write_raw_dataset", "DATE_FORMAT_SPELLINGS"]


#: The four datetime spellings mixed into ``intime`` / ``outtime`` / ``charttime``.
#: ``"%d/%m/%Y %H:%M"`` is the dangerous one: ``10/02/2026`` is the tenth of February,
#: but pandas' default month-first assumption silently reads it as the second of
#: October. The date cleaner must therefore be *format-aware*, not merely tolerant.
DATE_FORMAT_SPELLINGS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M",
    "%d/%m/%Y %H:%M",
    "%b %d %Y %H:%M",
)

_DIAGNOSES = ("Sepsis", "Pneumonia", "Trauma", "Stroke", "Cardiac Failure")
_ADMISSION_TYPES = ("Emergency", "ICU", "Ward", "Outpatient")
_MEDICATIONS = ("No Medication", "Antibiotic", "Insulin", "Vasopressor", "Sedative")
_DATA_SOURCES = ("Monitor", "Nurse Entry", "Laboratory")
_ETHNICITIES = ("African", "African-American", "Other")

#: Non-canonical spellings injected into categorical columns.
_CATEGORY_NOISE: dict[str, dict[str, tuple[str, ...]]] = {
    "gender": {"Male": ("male", "M", "MALE"), "Female": ("female", "F", "FEMALE")},
    "ethnicity": {"African-American": ("AA", "african-american"), "African": ("african",), "Other": ("other",)},
    "admission_type": {
        "Emergency": ("ER", "emergency", "Emergency Admission"),
        "ICU": ("icu", "ICU transfer"),
        "Ward": ("ward",),
        "Outpatient": ("outpatient",),
    },
    "diagnosis": {
        "Sepsis": ("sepsis", "SEPSIS"),
        "Pneumonia": ("pneumonia", "PNEUMONIA"),
        "Trauma": ("trauma",),
        "Stroke": ("stroke",),
        "Cardiac Failure": ("cardiac failure", "Cardiac"),
    },
    "medication": {
        "Antibiotic": ("antibiotic", "ANTIBIOTIC"),
        "Insulin": ("insulin", "INSULIN"),
        "Vasopressor": ("vasopressor",),
        "Sedative": ("sedative",),
        "No Medication": ("no medication", "NO MEDICATION"),
    },
    "data_source": {
        "Monitor": ("monitor", "MONITOR"),
        "Nurse Entry": ("nurse entry", "Nurse entry"),
        "Laboratory": ("laboratory", "Lab"),
    },
}

#: Clinically impossible values used to exercise the range validator.
_IMPOSSIBLE_VALUES: dict[str, tuple[float, ...]] = {
    "heart_rate": (0.0, 5.0, 320.0, 480.0),
    "sbp": (0.0, 15.0, 340.0),
    "dbp": (0.0, 5.0, 210.0),
    "mbp": (0.0, 8.0, 260.0),
    "spo2": (12.0, 118.0, 140.0),
    "resp_rate": (0.0, 1.0, 95.0),
    "glucose": (2.0, 1200.0),
    "temperature": (12.0, 61.0),
}


def _patient_baseline(rng: np.random.Generator, diagnosis: str) -> dict[str, float]:
    """Draw a physiologically coherent baseline for one admission.

    Parameters
    ----------
    rng:
        Seeded NumPy generator, so the dataset is byte-for-byte reproducible.
    diagnosis:
        Primary diagnosis. Septic and cardiac patients are shifted towards
        tachycardia and hypotension so that the channels are mutually correlated -
        without correlation, the cross-channel DeepTSE refinement in
        ``models.deeptse_imputer`` would have nothing to learn.

    Returns
    -------
    dict[str, float]
        Mean value per channel for this admission.
    """
    severity = {"Sepsis": 1.0, "Cardiac Failure": 0.7, "Pneumonia": 0.5, "Stroke": 0.3, "Trauma": 0.6}[diagnosis]
    sbp = float(rng.normal(125 - 18 * severity, 10))
    return {
        "heart_rate": float(rng.normal(78 + 26 * severity, 7)),
        "sbp": sbp,
        "dbp": sbp * 0.62 + float(rng.normal(0, 4)),
        "mbp": sbp * 0.75 + float(rng.normal(0, 3)),
        "temperature": float(rng.normal(36.9 + 1.0 * severity, 0.35)),
        "spo2": float(rng.normal(97.5 - 4.5 * severity, 1.2)),
        "resp_rate": float(rng.normal(15 + 8 * severity, 2.0)),
        "glucose": float(rng.normal(108 + 38 * severity, 15)),
        "creatinine": float(rng.normal(0.9 + 1.4 * severity, 0.3)),
        "wbc_count": float(rng.normal(7.5 + 6.0 * severity, 2.0)),
        "platelet_count": float(rng.normal(255 - 70 * severity, 45)),
        "severity": severity,
    }


def _format_datetime(moment: datetime, spelling: str) -> str:
    """Render a timestamp using one of the four inconsistent spellings."""
    return moment.strftime(spelling)


def generate_raw_dataset(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate the messy raw hospital-admission event table.

    Parameters
    ----------
    seed:
        Random seed controlling every stochastic choice, including which cells are
        corrupted. Re-running with the same seed reproduces the file exactly.

    Returns
    -------
    (pandas.DataFrame, dict)
        The raw frame with ``config.RAW_COLUMNS`` columns and exactly
        ``config.N_RAW_RECORDS`` rows, plus a ground-truth manifest counting the
        defects that were injected.

    Notes
    -----
    All corrupted cells are written as **strings**, because that is how a messy CSV
    actually arrives: pandas infers ``object`` dtype for any column holding a single
    ``"85.4 mmHg"``. Reproducing that faithfully is what makes the downstream
    profiling step realistic.
    """
    rng = np.random.default_rng(seed)
    truth: dict[str, Any] = {
        "seed": seed,
        "fahrenheit_cells": 0,
        "unit_suffix_cells": 0,
        "impossible_cells": 0,
        "category_noise_cells": 0,
        "missing_cells": {},
        "date_spellings": {spelling: 0 for spelling in DATE_FORMAT_SPELLINGS},
    }

    records: list[dict[str, Any]] = []
    window_start = datetime(2026, 1, 1, 0, 0)

    for admission_index in range(N_ADMISSIONS):
        admission_id = f"ADM{admission_index + 1:04d}"
        patient_id = f"P{admission_index + 1:03d}"
        diagnosis = str(rng.choice(_DIAGNOSES))
        baseline = _patient_baseline(rng, diagnosis)

        # Admission window: 2 to 6 days, starting somewhere in the first 45 days of 2026.
        intime = window_start + timedelta(
            days=int(rng.integers(0, 45)), hours=int(rng.integers(0, 24))
        )
        stay_hours = int(rng.integers(24, 72))
        outtime = intime + timedelta(hours=stay_hours)

        # One datetime spelling is chosen per admission for intime/outtime (a hospital
        # system exports consistently), but charttime spellings vary per row because
        # observations are keyed in by different staff and devices.
        admission_spelling = str(rng.choice(DATE_FORMAT_SPELLINGS))
        truth["date_spellings"][admission_spelling] += 2

        age = float(rng.integers(19, 92))
        gender = str(rng.choice(("Male", "Female")))
        ethnicity = str(rng.choice(_ETHNICITIES))
        admission_type = str(rng.choice(_ADMISSION_TYPES))
        sofa_score = float(np.clip(round(rng.normal(4 + 12 * baseline["severity"], 3)), 0, 24))

        # Irregular observation times: sorted, unique-to-the-minute offsets inside the stay.
        offsets = np.sort(rng.uniform(0.5, stay_hours - 0.5, size=OBSERVATIONS_PER_ADMISSION))

        for observation_index, offset_hours in enumerate(offsets):
            charttime = intime + timedelta(hours=float(offset_hours))
            charttime = charttime.replace(second=0, microsecond=0)
            drift = np.sin(offset_hours / 9.0) * 0.6  # slow physiological drift

            row: dict[str, Any] = {
                "admission_id": admission_id,
                "patient_id": patient_id,
                "intime": _format_datetime(intime, admission_spelling),
                "outtime": _format_datetime(outtime, admission_spelling),
                "charttime": "",  # filled below with its own spelling
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
                "admission_type": admission_type,
                "diagnosis": diagnosis,
                "medication": str(rng.choice(_MEDICATIONS)),
                "sofa_score": sofa_score,
                "data_source": str(rng.choice(_DATA_SOURCES)),
            }

            chart_spelling = str(rng.choice(DATE_FORMAT_SPELLINGS))
            truth["date_spellings"][chart_spelling] += 1
            row["charttime"] = _format_datetime(charttime, chart_spelling)

            for channel in ("heart_rate", "sbp", "dbp", "mbp", "temperature", "spo2", "resp_rate", "glucose"):
                scale = {
                    "heart_rate": 4.0,
                    "sbp": 6.0,
                    "dbp": 4.0,
                    "mbp": 4.0,
                    "temperature": 0.22,
                    "spo2": 0.9,
                    "resp_rate": 1.4,
                    "glucose": 11.0,
                }[channel]
                row[channel] = round(baseline[channel] + drift * scale + float(rng.normal(0, scale * 0.5)), 1)

            # Laboratory values move slowly; they are charted less often, which the
            # missingness rates below reflect.
            for channel in ("creatinine", "wbc_count", "platelet_count"):
                row[channel] = round(baseline[channel] + float(rng.normal(0, abs(baseline[channel]) * 0.08)), 2)

            row["_observation_index"] = observation_index
            records.append(row)

    frame = pd.DataFrame.from_records(records)

    # ---------------------------------------------------------------------------------
    # Defect injection. Order matters: values are corrupted *before* they are blanked,
    # so the missing-rate targets remain the published figures.
    # ---------------------------------------------------------------------------------

    # (a) Temperature unit mixture -- the true Celsius reading converted, never a constant.
    fahrenheit_mask = rng.random(len(frame)) < FAHRENHEIT_RATE
    celsius = pd.to_numeric(frame["temperature"], errors="coerce")
    temperature_text = celsius.round(1).astype(str) + " C"
    temperature_text[fahrenheit_mask] = (celsius[fahrenheit_mask] * 9.0 / 5.0 + 32.0).round(1).astype(str) + " F"
    frame["temperature"] = temperature_text
    truth["fahrenheit_cells"] = int(fahrenheit_mask.sum())

    # (b) Unit suffixes on numeric columns -- again carrying the real value.
    for channel, suffix, converter in (
        ("mbp", " mmHg", lambda value: value),
        ("sbp", " mmHg", lambda value: value),
        ("glucose", " mmol/L", lambda value: value / 18.0182),
    ):
        suffix_mask = rng.random(len(frame)) < UNIT_SUFFIX_RATE
        numeric = pd.to_numeric(frame[channel], errors="coerce")
        converted = numeric[suffix_mask].apply(converter).round(2).astype(str) + suffix
        frame[channel] = frame[channel].astype(object)
        frame.loc[suffix_mask, channel] = converted
        truth["unit_suffix_cells"] += int(suffix_mask.sum())

    # (c) Clinically impossible values.
    for channel, impossible in _IMPOSSIBLE_VALUES.items():
        invalid_mask = rng.random(len(frame)) < INVALID_VALUE_RATE
        if channel == "temperature":
            replacement = [f"{rng.choice(impossible)} C" for _ in range(int(invalid_mask.sum()))]
        else:
            replacement = list(rng.choice(impossible, size=int(invalid_mask.sum())))
        frame[channel] = frame[channel].astype(object)
        frame.loc[invalid_mask, channel] = replacement
        truth["impossible_cells"] += int(invalid_mask.sum())

    # Age: a handful of implausible entries plus the literal string "unknown".
    age_invalid = rng.random(len(frame)) < INVALID_VALUE_RATE
    frame["age"] = frame["age"].astype(object)
    frame.loc[age_invalid, "age"] = list(
        rng.choice([150.0, -5.0, 0.0, "unknown"], size=int(age_invalid.sum()))
    )
    truth["impossible_cells"] += int(age_invalid.sum())

    # (d) Categorical spelling variation.
    for column, mapping in _CATEGORY_NOISE.items():
        noise_mask = rng.random(len(frame)) < CATEGORY_NOISE_RATE
        values = frame[column].astype(object).copy()
        for position in np.flatnonzero(noise_mask):
            canonical = values.iloc[position]
            variants = mapping.get(str(canonical))
            if variants:
                values.iloc[position] = str(rng.choice(variants))
                truth["category_noise_cells"] += 1
        frame[column] = values

    # (e) Missing values, at the published per-channel rates.
    for channel, rate in MISSING_RATES.items():
        if channel not in frame.columns:
            continue
        missing_mask = rng.random(len(frame)) < rate
        # Never blank the very first observation of an admission for *every* channel at
        # once; leading gaps are wanted, but an admission with no data at all is noise.
        frame.loc[missing_mask, channel] = np.nan
        truth["missing_cells"][channel] = int(missing_mask.sum())

    frame = frame.drop(columns=["_observation_index"])
    frame = frame[RAW_COLUMNS]

    # (f) Duplicated observation rows. Injected last so that duplicates are exact copies
    # including their defects, then the frame is trimmed back to exactly 1000 rows so the
    # shipped file matches the published record count.
    n_duplicates = int(round(N_RAW_RECORDS * DUPLICATE_RATE))
    keep_positions = np.sort(rng.choice(len(frame), size=len(frame) - n_duplicates, replace=False))
    retained = frame.iloc[keep_positions]
    # Duplicate sources are drawn from the *retained* rows, so each injected copy is
    # guaranteed to have its original still present. ``df.duplicated().sum()`` on the
    # shipped file is then exactly ``n_duplicates``.
    duplicate_sources = rng.choice(len(retained), size=n_duplicates, replace=False)
    duplicated_rows = retained.iloc[duplicate_sources].copy()
    frame = pd.concat([retained, duplicated_rows], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    truth["duplicate_rows_injected"] = n_duplicates
    truth["n_rows"] = int(len(frame))
    truth["n_columns"] = int(frame.shape[1])

    return frame, truth


def write_raw_dataset(path: Path | str = RAW_CSV, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate the raw dataset and persist it to CSV.

    Parameters
    ----------
    path:
        Destination CSV path. Defaults to ``config.RAW_CSV``.
    seed:
        Random seed forwarded to :func:`generate_raw_dataset`.

    Returns
    -------
    (pandas.DataFrame, dict)
        The frame that was written, and the ground-truth manifest.

    Example
    -------
    >>> frame, truth = write_raw_dataset()
    >>> truth["duplicate_rows_injected"]
    100
    """
    frame, truth = generate_raw_dataset(seed=seed)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    truth["path"] = str(destination)
    return frame, truth
