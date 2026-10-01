"""
Module purpose:

Single source of truth for the *Medical AI Preprocessing Agent* project.

Every other module imports its paths, clinical schema, physiological plausibility
ranges and plotting palette from here. Centralising this configuration means that
the notebook, the agent layer, the deterministic preprocessing tools, the
validation agent and the Streamlit dashboard can never disagree about where a file
lives or what a "plausible heart rate" is.

The module deliberately contains **no logic beyond path resolution**. It is safe to
import from anywhere (notebook, script, Streamlit process) and has no side effects
other than creating the output directories.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# 1. Project paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

for _directory in (DATA_DIR, REPORTS_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# Canonical dataset produced by ``preprocessing.synthetic_data``.
RAW_CSV: Path = DATA_DIR / "raw_hospital_admission_data.csv"

# The dataset supplied by the project author before the pipeline existed. It is kept
# for provenance and can be run through the identical pipeline by flipping a flag in
# the notebook. See README.md ("Two raw datasets") for why both exist.
RAW_CSV_ORIGINAL: Path = DATA_DIR / "raw_hospital_admission_medical_data_original.csv"

CLEANED_CSV: Path = DATA_DIR / "cleaned_medical_data.csv"
HOURLY_CSV: Path = DATA_DIR / "processed_hourly_timeseries.csv"

VALIDATION_REPORT_JSON: Path = REPORTS_DIR / "validation_report.json"
CLEANING_SUMMARY_HTML: Path = REPORTS_DIR / "cleaning_summary.html"
AGENT_TRACE_JSON: Path = REPORTS_DIR / "agent_trace.json"
DASHBOARD_PAYLOAD_JSON: Path = REPORTS_DIR / "dashboard_payload.json"

# --------------------------------------------------------------------------------------
# 2. Dataset generation parameters
# --------------------------------------------------------------------------------------

RANDOM_SEED: int = 2026
N_ADMISSIONS: int = 100
OBSERVATIONS_PER_ADMISSION: int = 10
N_RAW_RECORDS: int = N_ADMISSIONS * OBSERVATIONS_PER_ADMISSION  # 1000 event rows

#: Proportion of *rows* that are exact duplicates of another row (injected, then the
#: frame is truncated back to ``N_RAW_RECORDS`` so the shipped file is exactly 1000 rows).
DUPLICATE_RATE: float = 0.10

#: Proportion of temperature readings recorded in Fahrenheit rather than Celsius.
FAHRENHEIT_RATE: float = 0.40

#: Proportion of *numeric* cells that carry a unit suffix (e.g. ``"85.4 mmHg"``).
UNIT_SUFFIX_RATE: float = 0.12

#: Proportion of physiological cells replaced by a clinically impossible value.
INVALID_VALUE_RATE: float = 0.05

#: Proportion of categorical cells written with a non-canonical spelling.
CATEGORY_NOISE_RATE: float = 0.30

#: Per-channel missing-completely-at-random rates in the raw event table.
MISSING_RATES: dict[str, float] = {
    "heart_rate": 0.15,
    "sbp": 0.20,
    "dbp": 0.20,
    "mbp": 0.15,
    "temperature": 0.15,
    "spo2": 0.10,
    "resp_rate": 0.15,
    "glucose": 0.20,
    "creatinine": 0.20,
    "wbc_count": 0.25,
    "platelet_count": 0.15,
    "sofa_score": 0.10,
    "age": 0.05,
}

# --------------------------------------------------------------------------------------
# 3. Clinical schema
# --------------------------------------------------------------------------------------

ADMISSION_COLUMNS: list[str] = ["admission_id", "patient_id", "intime", "outtime", "charttime"]
DEMOGRAPHIC_COLUMNS: list[str] = ["age", "gender", "ethnicity"]

#: Continuous bedside channels that become the hourly time series. Order matters: it is
#: the column order of every derived matrix and of the dashboard.
VITAL_COLUMNS: list[str] = [
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "temperature",
    "spo2",
    "resp_rate",
    "glucose",
]

LAB_COLUMNS: list[str] = ["creatinine", "wbc_count", "platelet_count"]
CONTEXT_COLUMNS: list[str] = [
    "admission_type",
    "diagnosis",
    "medication",
    "sofa_score",
    "data_source",
]

RAW_COLUMNS: list[str] = (
    ADMISSION_COLUMNS + DEMOGRAPHIC_COLUMNS + VITAL_COLUMNS + LAB_COLUMNS + CONTEXT_COLUMNS
)

#: Channels carried into the Value-Mask-Decay representation and DeepTSE refinement.
TIMESERIES_CHANNELS: list[str] = VITAL_COLUMNS

#: Physiologically plausible ranges. Values outside these bounds are *not* deleted:
#: the cell is set to missing and the reason is recorded in the audit trail, so a
#: reviewer can always reconstruct what the agent rejected and why.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "age": (0.0, 120.0),
    "heart_rate": (20.0, 250.0),
    "sbp": (50.0, 250.0),
    "dbp": (20.0, 150.0),
    "mbp": (30.0, 180.0),
    "temperature": (30.0, 43.0),
    "spo2": (50.0, 100.0),
    "resp_rate": (4.0, 60.0),
    "glucose": (20.0, 600.0),
    "creatinine": (0.1, 15.0),
    "wbc_count": (0.5, 60.0),
    "platelet_count": (5.0, 800.0),
    "sofa_score": (0.0, 24.0),
}

#: Canonical category vocabularies used by ``preprocessing.categorical_cleaner``.
CANONICAL_CATEGORIES: dict[str, dict[str, str]] = {
    "gender": {
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female",
    },
    "ethnicity": {
        "african": "African",
        "aa": "African-American",
        "african-american": "African-American",
        "african american": "African-American",
        "other": "Other",
    },
    "admission_type": {
        "er": "Emergency",
        "emergency": "Emergency",
        "emergency admission": "Emergency",
        "icu": "ICU",
        "icu transfer": "ICU",
        "ward": "Ward",
        "outpatient": "Outpatient",
    },
    "diagnosis": {
        "sepsis": "Sepsis",
        "pneumonia": "Pneumonia",
        "trauma": "Trauma",
        "stroke": "Stroke",
        "cardiac failure": "Cardiac Failure",
        "cardiac": "Cardiac Failure",
    },
    "medication": {
        "no medication": "No Medication",
        "none": "No Medication",
        "antibiotic": "Antibiotic",
        "insulin": "Insulin",
        "vasopressor": "Vasopressor",
        "sedative": "Sedative",
    },
    "data_source": {
        "monitor": "Monitor",
        "nurse entry": "Nurse Entry",
        "laboratory": "Laboratory",
        "lab": "Laboratory",
    },
}

# --------------------------------------------------------------------------------------
# 4. Missing-value handling parameters
# --------------------------------------------------------------------------------------

#: Base of the exponential confidence decay applied to carried-forward observations.
DECAY_BASE: float = 0.75

#: Resampling frequency for time-series reconstruction ("1h" == hourly).
RESAMPLE_FREQ: str = "1h"

#: Fraction of *genuinely observed* hourly cells held out to benchmark imputers.
HOLDOUT_FRACTION: float = 0.15

# --------------------------------------------------------------------------------------
# 5. Visual identity
# --------------------------------------------------------------------------------------
# Categorical slots are assigned in fixed order and never cycled. The first three slots
# are validated for all-pairs colour-vision separation, which is why no figure in this
# project plots more than three categorical series at once.

PALETTE: dict[str, str] = {
    "series_1": "#2a78d6",  # blue    - "after cleaning" / clean state
    "series_2": "#eb6834",  # orange  - "before cleaning" / problem state
    "series_3": "#1baf7a",  # aqua    - third series (always directly labelled)
    "series_4": "#eda100",  # yellow  - reserved, small multiples only
    "surface": "#fcfcfb",
    "grid": "#e4e3df",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#7a7873",
    "good": "#008300",
    "warning": "#eda100",
    "critical": "#e34948",
}

#: Single-hue sequential ramp (light -> dark) for magnitude encodings.
SEQUENTIAL_BLUE: list[str] = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
]

__all__ = [name for name in dir() if not name.startswith("_")]
