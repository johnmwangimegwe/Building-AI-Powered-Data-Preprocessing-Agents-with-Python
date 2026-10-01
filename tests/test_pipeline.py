"""
Module purpose:

Regression tests for the preprocessing pipeline.

These tests guard the claims the talk makes on stage. Each one corresponds to a
statement a slide asserts, so a failing test means a slide has become untrue:

* the generator injects exactly the defects it advertises;
* day-first dates parse to the intended day, not the naive month-first reading;
* Fahrenheit converts correctly and mmol/L scales correctly;
* the decay sequence is exactly $1, 0.75, 0.5625, \\ldots$;
* a neutral-filled cell standardises to exactly zero;
* refinement never modifies an observed value;
* the full agent run passes every validation assertion.

Run with::

    python -m pytest tests/ -v

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.preprocessing_agent import PreprocessingAgent
from agents.tool_registry import CANONICAL_ORDER, TOOL_REGISTRY, describe_tools
from agents.validation_agent import ValidationAgent
from config import DECAY_BASE, N_RAW_RECORDS, TIMESERIES_CHANNELS
from models.deeptse_imputer import benchmark_against_locf
from preprocessing.date_cleaner import parse_datetime_series
from preprocessing.missing_value_handler import locf_with_mask_and_decay, neutral_fill_leading_gaps
from preprocessing.synthetic_data import generate_raw_dataset
from preprocessing.unit_converter import convert_units, fahrenheit_to_celsius, parse_measurement


@pytest.fixture(scope="module")
def raw_data() -> tuple[pd.DataFrame, dict]:
    """Generate the synthetic dataset once for the whole module."""
    return generate_raw_dataset(seed=2026)


@pytest.fixture(scope="module")
def agent_result(raw_data):
    """Run the agent once for the whole module."""
    frame, _ = raw_data
    return PreprocessingAgent(verbose=False).run(frame, trace_path=None)


# ------------------------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------------------------


def test_dataset_shape(raw_data):
    frame, truth = raw_data
    assert len(frame) == N_RAW_RECORDS
    assert frame.shape[1] == 24
    assert truth["n_rows"] == N_RAW_RECORDS


def test_duplicates_are_exactly_as_injected(raw_data):
    frame, truth = raw_data
    assert int(frame.duplicated().sum()) == truth["duplicate_rows_injected"]


def test_generation_is_reproducible():
    first, _ = generate_raw_dataset(seed=7)
    second, _ = generate_raw_dataset(seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_mixed_units_present(raw_data):
    frame, _ = raw_data
    temperatures = frame["temperature"].dropna().astype(str)
    assert temperatures.str.contains("F").any()
    assert temperatures.str.contains("C").any()
    assert frame["glucose"].dropna().astype(str).str.contains("mmol/L").any()


# ------------------------------------------------------------------------------------
# Date parsing -- the silent-corruption case
# ------------------------------------------------------------------------------------


def test_day_first_dates_parse_to_the_intended_day():
    """``10/02/2026`` is the tenth of February, not the second of October."""
    parsed, audit = parse_datetime_series(pd.Series(["10/02/2026 08:00"]))
    assert parsed.iloc[0].month == 2
    assert parsed.iloc[0].day == 10
    assert audit["naive_parse_disagreements"] == 1


def test_all_four_spellings_resolve_to_the_same_instant():
    spellings = pd.Series(
        ["2026-02-10 08:00", "2026/02/10 08:00", "10/02/2026 08:00", "Feb 10 2026 08:00"]
    )
    parsed, audit = parse_datetime_series(spellings)
    assert parsed.nunique() == 1
    assert audit["n_formats"] == 4
    assert audit["unparseable"] == 0


# ------------------------------------------------------------------------------------
# Unit conversion
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fahrenheit", "celsius"),
    [(98.6, 37.0), (32.0, 0.0), (212.0, 100.0), (100.4, 38.0)],
)
def test_fahrenheit_conversion(fahrenheit, celsius):
    assert fahrenheit_to_celsius(fahrenheit) == pytest.approx(celsius, abs=1e-9)


def test_parse_measurement_splits_magnitude_and_unit():
    assert parse_measurement("6.6 mmol/L") == (6.6, "mmol/l")
    assert parse_measurement("85 mmHg") == (85.0, "mmhg")
    assert parse_measurement(120.0) == (120.0, "")
    assert parse_measurement(np.nan) == (None, "")


def test_unit_conversion_preserves_the_underlying_value():
    frame = pd.DataFrame({"temperature": ["37.0 C", "98.6 F"], "glucose": ["108.0", "6.0 mmol/L"]})
    converted, audit = convert_units(frame, columns=["temperature", "glucose"])
    assert converted["temperature"].tolist() == pytest.approx([37.0, 37.0], abs=0.01)
    assert converted["glucose"].iloc[1] == pytest.approx(108.11, abs=0.01)
    assert audit["temperature"]["converted_from_fahrenheit"] == 1


def test_no_temperature_survives_in_the_fahrenheit_band(agent_result):
    values = agent_result.cleaned_events["temperature"].dropna()
    assert not values.between(90.0, 115.0).any()


# ------------------------------------------------------------------------------------
# Value, Mask and Decay
# ------------------------------------------------------------------------------------


def test_locf_mask_and_decay_match_the_published_example():
    """The worked example from the talk: 80, gap, gap, 85."""
    output = locf_with_mask_and_decay(pd.Series([80.0, np.nan, np.nan, 85.0]), population_mean=75.0)
    assert output["value"].tolist() == [80.0, 80.0, 80.0, 85.0]
    assert output["mask"].tolist() == [1, 0, 0, 1]
    assert output["decay"].tolist() == pytest.approx([1.0, 0.75, 0.5625, 1.0], abs=1e-9)


def test_decay_follows_the_formula():
    output = locf_with_mask_and_decay(pd.Series([80.0] + [np.nan] * 6), population_mean=75.0)
    expected = [1.0] + [DECAY_BASE**step for step in range(1, 7)]
    assert output["decay"].tolist() == pytest.approx(expected, abs=1e-9)


def test_neutral_fill_touches_only_the_leading_gap():
    filled, n_filled = neutral_fill_leading_gaps(pd.Series([np.nan, np.nan, 80.0, np.nan]), 75.0)
    assert n_filled == 2
    assert filled.tolist()[:3] == [75.0, 75.0, 80.0]
    assert np.isnan(filled.iloc[3])


def test_neutral_filled_cells_standardise_to_zero(agent_result):
    audit = agent_result.audits["build_value_mask_decay"]
    for channel, worst in audit["neutral_fill_zero_check"].items():
        assert worst < 0.01, f"{channel} neutral fill does not standardise to zero"


def test_masks_and_decays_are_well_formed(agent_result):
    processed = agent_result.processed
    for channel in TIMESERIES_CHANNELS:
        assert processed[f"{channel}_mask"].isin([0, 1]).all()
        assert processed[f"{channel}_decay"].between(0.0, 1.0).all()
        assert processed[f"{channel}_value"].notna().all()


# ------------------------------------------------------------------------------------
# Time-series reconstruction
# ------------------------------------------------------------------------------------


def test_hourly_grid_is_contiguous(agent_result):
    processed = agent_result.processed
    gaps = processed.groupby("admission_id")["hour"].apply(
        lambda series: series.sort_values().diff().dropna().dt.total_seconds().div(3600).eq(1.0).all()
    )
    assert gaps.all()


def test_alignment_expands_the_table(agent_result):
    audit = agent_result.audits["build_hourly_timeseries"]
    assert audit["rows_after"] > audit["rows_before"]
    assert audit["expansion_factor"] > 1.0


def test_alignment_reveals_more_missingness_than_the_event_table_showed(agent_result):
    audit = agent_result.audits["build_hourly_timeseries"]
    for channel in TIMESERIES_CHANNELS:
        assert audit["missing_pct_after"][channel] > audit["missing_pct_before"][channel]


# ------------------------------------------------------------------------------------
# Refinement -- the inviolable rule
# ------------------------------------------------------------------------------------


def test_refinement_never_modifies_an_observed_value(agent_result):
    processed = agent_result.processed
    for channel in TIMESERIES_CHANNELS:
        observed = processed[f"{channel}_mask"] == 1
        difference = (processed.loc[observed, f"{channel}_refined"] - processed.loc[observed, f"{channel}_value"]).abs()
        assert difference.max() == 0.0, f"{channel} observed values were altered"


def test_refiner_reports_zero_observed_modifications(agent_result):
    assert agent_result.audits["refine_missing_values"]["observed_cells_modified"] == 0


def test_benchmark_runs_and_reports_both_estimators(agent_result):
    benchmark = benchmark_against_locf(agent_result.hourly, holdout_mode="block")
    assert benchmark["overall"]["held_out_cells"] > 0
    for entry in benchmark["by_channel"].values():
        assert entry["locf_mae"] > 0
        assert entry["deeptse_mae"] > 0


# ------------------------------------------------------------------------------------
# Agent behaviour
# ------------------------------------------------------------------------------------


def test_tool_registry_matches_canonical_order():
    assert set(CANONICAL_ORDER) == set(TOOL_REGISTRY)
    assert "remove_duplicates" in describe_tools()


def test_agent_executes_and_every_check_passes(agent_result):
    assert agent_result.all_checks_passed
    assert len(agent_result.steps) == len(CANONICAL_ORDER)


def test_agent_rejects_an_unknown_tool():
    from agents.tool_registry import get_tool

    with pytest.raises(KeyError):
        get_tool("drop_the_whole_table")


def test_validation_report_passes_every_assertion(raw_data, agent_result):
    frame, truth = raw_data
    report = ValidationAgent().validate(frame, agent_result, ground_truth=truth)
    failed = [assertion["name"] for assertion in report["assertions"] if not assertion["passed"]]
    assert not failed, f"failed assertions: {failed}"
    assert report["verdict"]["passed"]


def test_no_admission_is_lost(raw_data, agent_result):
    frame, _ = raw_data
    assert frame["admission_id"].nunique() == agent_result.processed["admission_id"].nunique()
