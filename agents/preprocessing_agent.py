"""
Module purpose:

The preprocessing agent - the plan / act / observe / validate / refine loop.

This module is the project's argument in executable form. A chatbot can *suggest*
cleaning steps; an agent runs a cycle:

    profile  ->  plan  ->  execute a tool  ->  observe its audit
                   ^                                  |
                   |                                  v
                   +------ refine, if a check fails --+

Each iteration the agent executes one tool, reads the audit the tool returns, and
applies a **postcondition check** that expresses what that tool was supposed to
achieve. Duplicate removal must leave zero duplicates. Unit conversion must leave
every temperature inside the human range. Refinement must modify zero observed cells.
If a check fails, the agent records the failure, attempts one documented repair, and
re-checks - and if the repair does not work it stops and says so instead of writing
out a file that looks finished.

The loop is bounded, the action space is closed, and the entire trajectory is
serialised to ``reports/agent_trace.json``. Together those three properties are what
makes an autonomous data transformation reviewable after the fact, which is the only
basis on which anyone should trust one.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from agents.llm_planner import PlanStep, RuleBasedPlanner, build_planner
from agents.tool_registry import TOOL_REGISTRY, get_tool
from config import AGENT_TRACE_JSON, PLAUSIBLE_RANGES, TIMESERIES_CHANNELS
from preprocessing.data_profiler import profile_data

__all__ = ["PreprocessingAgent", "AgentStepRecord", "AgentResult"]


@dataclass
class AgentStepRecord:
    """One completed iteration of the agent loop.

    Attributes
    ----------
    step:
        1-based position in the executed plan.
    tool:
        Tool name.
    reason:
        The planner's justification.
    audit:
        The audit dictionary the tool returned.
    checks:
        Postcondition results, each ``{"name", "passed", "detail"}``.
    rows_before / rows_after:
        Table height either side of the call.
    duration_seconds:
        Wall-clock execution time.
    repaired:
        Whether a failed check triggered a repair attempt.
    """

    step: int
    tool: str
    reason: str
    audit: dict[str, Any]
    checks: list[dict[str, Any]]
    rows_before: int
    rows_after: int
    duration_seconds: float
    repaired: bool = False

    @property
    def passed(self) -> bool:
        """``True`` when every postcondition check passed."""
        return all(check["passed"] for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, used for the agent trace."""
        return {
            "step": self.step,
            "tool": self.tool,
            "reason": self.reason,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "duration_seconds": round(self.duration_seconds, 3),
            "checks": self.checks,
            "passed": self.passed,
            "repaired": self.repaired,
            "audit": _json_safe(self.audit),
        }


@dataclass
class AgentResult:
    """Everything the agent produced in one run.

    Attributes
    ----------
    cleaned_events:
        The cleaned event-level table.
    hourly:
        The hourly time series before Value-Mask-Decay expansion.
    processed:
        The final model-ready table.
    profile_before / profile_after:
        Data profiles either side of the run.
    steps:
        Executed step records.
    concerns:
        Planner concerns plus anything raised during execution.
    audits:
        Tool name -> audit dictionary, for the validation agent.
    """

    cleaned_events: pd.DataFrame
    hourly: pd.DataFrame
    processed: pd.DataFrame
    profile_before: dict[str, Any]
    profile_after: dict[str, Any]
    steps: list[AgentStepRecord] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    audits: dict[str, Any] = field(default_factory=dict)

    @property
    def all_checks_passed(self) -> bool:
        """``True`` when every step's postconditions held."""
        return all(step.passed for step in self.steps)

    def to_trace(self, planner_name: str) -> dict[str, Any]:
        """Assemble the serialisable trajectory written to ``agent_trace.json``."""
        return {
            "planner": planner_name,
            "steps_executed": len(self.steps),
            "all_checks_passed": self.all_checks_passed,
            "concerns": self.concerns,
            "plan": [{"step": record.step, "tool": record.tool, "reason": record.reason} for record in self.steps],
            "trajectory": [record.to_dict() for record in self.steps],
        }


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy and pandas scalars into plain Python types.

    ``json.dump`` refuses ``numpy.int64``. Every audit in this project passes through
    here before serialisation, which is why the reports never fail to write at the end
    of a long run.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


# ------------------------------------------------------------------------------------
# Postcondition checks. Each returns (name, passed, detail).
# ------------------------------------------------------------------------------------


def _check_no_duplicates(frame: pd.DataFrame, audit: dict[str, Any]) -> list[tuple[str, bool, str]]:
    remaining = int(frame.duplicated().sum())
    return [("no_exact_duplicates_remain", remaining == 0, f"{remaining} duplicate rows remain")]


def _check_datetimes(frame: pd.DataFrame, audit: dict[str, Any]) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    for column in ("intime", "outtime", "charttime"):
        if column not in frame.columns:
            continue
        is_datetime = pd.api.types.is_datetime64_any_dtype(frame[column])
        results.append((f"{column}_is_datetime", is_datetime, str(frame[column].dtype)))
    window = audit.get("window_reconciliation", {})
    results.append(
        (
            "no_inverted_admission_windows",
            window.get("inverted_windows", 0) == 0,
            f"{window.get('inverted_windows', 0)} admissions discharge before admission",
        )
    )
    length_of_stay = window.get("length_of_stay_hours", {})
    results.append(
        (
            "length_of_stay_positive",
            float(length_of_stay.get("min", 0)) > 0,
            f"minimum stay {length_of_stay.get('min')} hours",
        )
    )
    return results


def _check_units(frame: pd.DataFrame, audit: dict[str, Any]) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    for column in ("temperature", "glucose", "sbp", "mbp"):
        if column not in frame.columns:
            continue
        results.append(
            (
                f"{column}_is_numeric",
                pd.api.types.is_numeric_dtype(frame[column]),
                str(frame[column].dtype),
            )
        )
    if "temperature" in frame.columns:
        values = pd.to_numeric(frame["temperature"], errors="coerce").dropna()
        # The failure this check exists to catch is a *missed conversion*: a real body
        # temperature left in Fahrenheit reads between 90 and 115. Values far outside
        # both scales (12, 61) are impossible in either unit, so rejecting them belongs
        # to range validation, not here - flagging them at this step would invite a
        # "repair" that converts nonsense into different nonsense.
        stray = int(values.between(90.0, 115.0).sum())
        results.append(
            ("no_temperature_left_in_fahrenheit_band", stray == 0, f"{stray} values in the 90-115 body-temperature-Fahrenheit band")
        )
    return results


def _check_categories(frame: pd.DataFrame, audit: dict[str, Any]) -> list[tuple[str, bool, str]]:
    unmapped = {column: detail["unmapped_values"] for column, detail in audit.items() if detail.get("unmapped_values")}
    return [
        (
            "all_categories_mapped",
            not unmapped,
            "unmapped: " + json.dumps(unmapped) if unmapped else "every value matched the vocabulary",
        )
    ]


#: A range-validation pass may legitimately reject a few per cent of a channel. Losing
#: most of it means something upstream is wrong - almost always that the step ran before
#: units were canonical, so an entire measurement scale was read as unparseable text.
MAX_ACCEPTABLE_CHANNEL_LOSS = 0.50


def _check_ranges(frame: pd.DataFrame, audit: dict[str, Any]) -> list[tuple[str, bool, str]]:
    offenders: list[str] = []
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        n_bad = int(((values < low) | (values > high)).sum())
        if n_bad:
            offenders.append(f"{column}={n_bad}")

    # A range check alone is not enough, and this is the subtle part. "Are all values in
    # range?" is trivially satisfied by a column that has been emptied: an empty set has
    # no offending members. Run range validation before unit conversion and every
    # "98.6 F" is unparseable text, so the whole channel is rejected - and the check
    # above still reports a clean pass. Data loss has to be checked for on its own terms.
    catastrophic: list[str] = []
    for column, detail in audit.get("by_column", {}).items():
        observed_before = detail.get("observed_before", 0)
        observed_after = detail.get("observed_after", 0)
        if observed_before >= 20:
            lost_fraction = 1.0 - observed_after / observed_before
            if lost_fraction > MAX_ACCEPTABLE_CHANNEL_LOSS:
                catastrophic.append(f"{column} lost {lost_fraction:.0%} ({observed_before:,} -> {observed_after:,})")

    return [
        ("all_values_within_physiological_range", not offenders, ", ".join(offenders) or "clean"),
        (
            "no_channel_lost_the_majority_of_its_observations",
            not catastrophic,
            "; ".join(catastrophic) or f"every channel retained over {1 - MAX_ACCEPTABLE_CHANNEL_LOSS:.0%} of its observations",
        ),
    ]


def _check_hourly(frame: pd.DataFrame, audit: dict[str, Any]) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    if "hour" in frame.columns and "admission_id" in frame.columns:
        gaps = frame.groupby("admission_id")["hour"].apply(
            lambda series: series.sort_values().diff().dropna().dt.total_seconds().div(3600).eq(1.0).all()
        )
        contiguous = bool(gaps.all()) if len(gaps) else False
        results.append(("hourly_grid_is_contiguous", contiguous, f"{int((~gaps).sum())} admissions with a broken grid"))
        duplicated_keys = int(frame.duplicated(subset=["admission_id", "hour"]).sum())
        results.append(("one_row_per_admission_hour", duplicated_keys == 0, f"{duplicated_keys} duplicated keys"))
    results.append(
        (
            "expansion_increased_row_count",
            audit.get("rows_after", 0) > audit.get("rows_before", 0),
            f"{audit.get('rows_before')} -> {audit.get('rows_after')} rows",
        )
    )
    return results


def _check_value_mask_decay(frame: pd.DataFrame, audit: dict[str, Any]) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    missing_after = [
        channel
        for channel in TIMESERIES_CHANNELS
        if f"{channel}_value" in frame.columns and frame[f"{channel}_value"].isna().any()
    ]
    results.append(("no_missing_values_remain", not missing_after, ", ".join(missing_after) or "all channels complete"))

    bad_mask = [
        channel
        for channel in TIMESERIES_CHANNELS
        if f"{channel}_mask" in frame.columns and not frame[f"{channel}_mask"].isin([0, 1]).all()
    ]
    results.append(("mask_is_binary", not bad_mask, ", ".join(bad_mask) or "all masks in {0, 1}"))

    bad_decay = [
        channel
        for channel in TIMESERIES_CHANNELS
        if f"{channel}_decay" in frame.columns
        and not frame[f"{channel}_decay"].between(0.0, 1.0).all()
    ]
    results.append(("decay_within_unit_interval", not bad_decay, ", ".join(bad_decay) or "all decays in [0, 1]"))

    zero_check = audit.get("neutral_fill_zero_check", {})
    worst = max(zero_check.values()) if zero_check else 0.0
    results.append(
        (
            "neutral_fill_standardises_to_zero",
            worst < 0.01,
            f"largest absolute z among neutral-filled cells: {worst}",
        )
    )
    return results


def _check_refinement(frame: pd.DataFrame, audit: dict[str, Any]) -> list[tuple[str, bool, str]]:
    modified = audit.get("observed_cells_modified", 0)
    return [
        (
            "observed_cells_never_modified",
            modified == 0,
            f"{modified} observed cells were altered (must be 0)",
        ),
        (
            "refinement_columns_created",
            all(f"{channel}_refined" in frame.columns for channel in TIMESERIES_CHANNELS if f"{channel}_value" in frame.columns),
            "refined columns present",
        ),
    ]


CHECKS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], list[tuple[str, bool, str]]]] = {
    "remove_duplicates": _check_no_duplicates,
    "standardise_datetimes": _check_datetimes,
    "convert_units": _check_units,
    "standardise_categories": _check_categories,
    "validate_ranges": _check_ranges,
    "build_hourly_timeseries": _check_hourly,
    "build_value_mask_decay": _check_value_mask_decay,
    "refine_missing_values": _check_refinement,
}


class PreprocessingAgent:
    """Plan, execute, observe, validate and refine a clinical cleaning run.

    Parameters
    ----------
    planner:
        Anything exposing ``plan(profile) -> (steps, concerns)``. Defaults to
        :class:`agents.llm_planner.RuleBasedPlanner`.
    verbose:
        Stream progress to stdout. Recommended on stage - the audience should watch
        the loop turn.
    max_repair_attempts:
        How many times a single failing step may be retried with a documented repair.

    Example
    -------
    >>> agent = PreprocessingAgent()
    >>> result = agent.run(raw_frame)
    >>> result.all_checks_passed
    True
    """

    def __init__(
        self,
        planner: Any | None = None,
        verbose: bool = True,
        max_repair_attempts: int = 1,
    ) -> None:
        self.planner = planner or RuleBasedPlanner()
        self.verbose = verbose
        self.max_repair_attempts = max_repair_attempts

    # -- reporting ---------------------------------------------------------------

    def _say(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # -- repair strategies -------------------------------------------------------

    def _repair(self, tool_name: str, frame: pd.DataFrame, failed: list[dict[str, Any]]) -> tuple[pd.DataFrame, str] | None:
        """Attempt one documented repair for a failed postcondition.

        Parameters
        ----------
        tool_name:
            Tool whose check failed.
        frame:
            Current table state.
        failed:
            The failing check records.

        Returns
        -------
        (pandas.DataFrame, str) | None
            Repaired table and a description, or ``None`` when no repair is defined -
            in which case the agent stops rather than guessing.

        Notes
        -----
        Repairs are enumerated, not generated. An agent that invents a fix for an
        unanticipated failure in clinical data is more dangerous than one that halts,
        because the failure it invents a fix for is the one nobody reviewed.
        """
        failed_names = {check["name"] for check in failed}

        if tool_name == "remove_duplicates" and "no_exact_duplicates_remain" in failed_names:
            repaired = frame.drop_duplicates().reset_index(drop=True)
            return repaired, "Re-applied exact-match deduplication over the full column set."

        if tool_name == "validate_ranges" and "all_values_within_physiological_range" in failed_names:
            from preprocessing.range_validator import validate_ranges

            repaired, _ = validate_ranges(frame)
            return repaired, "Re-ran range validation; the first pass ran before units were canonical."

        if tool_name == "validate_ranges" and "no_channel_lost_the_majority_of_its_observations" in failed_names:
            # There is deliberately no repair for this. The measurements are already gone,
            # and no later step can recover them. Returning None halts the run, which is
            # the only honest response: the alternative is writing out a file that claims
            # to be clean and is empty.
            return None

        if tool_name == "convert_units" and "no_temperature_left_in_fahrenheit_band" in failed_names:
            from preprocessing.unit_converter import fahrenheit_to_celsius

            repaired = frame.copy()
            values = pd.to_numeric(repaired["temperature"], errors="coerce")
            stray = values.between(90.0, 115.0)
            repaired.loc[stray, "temperature"] = values[stray].apply(fahrenheit_to_celsius).round(2)
            return repaired, f"Converted {int(stray.sum())} unlabelled Fahrenheit temperatures inferred from range."

        return None

    # -- the loop ----------------------------------------------------------------

    def run(self, raw: pd.DataFrame, trace_path: Path | None = AGENT_TRACE_JSON) -> AgentResult:
        """Execute a full cleaning run over the raw event table.

        Parameters
        ----------
        raw:
            Raw hospital admission event table as loaded from CSV.
        trace_path:
            Where to write the JSON trajectory. ``None`` skips writing.

        Returns
        -------
        AgentResult
            Cleaned event table, hourly table, processed table, profiles either side,
            per-step records, concerns and the collected audits.

        Raises
        ------
        RuntimeError
            When a postcondition fails and no repair restores it. Halting is the
            correct behaviour: the alternative is writing a file that claims to be
            clean and is not.
        """
        self._say("=" * 78)
        self._say("AI MEDICAL PREPROCESSING AGENT")
        self._say("=" * 78)

        self._say("\n[PERCEIVE] Profiling the raw dataset ...")
        profile_before = profile_data(raw)
        self._say(
            f"           {profile_before['n_rows']:,} rows x {profile_before['n_columns']} columns | "
            f"{profile_before['duplicates']['exact_duplicate_rows']:,} duplicate rows | "
            f"{len(profile_before['unit_inconsistencies'])} columns with mixed units"
        )

        self._say(f"\n[PLAN] Planner: {self.planner.name}")
        steps, concerns = self.planner.plan(profile_before)
        for index, step in enumerate(steps, start=1):
            self._say(f"       {index}. {step.tool}")
            self._say(f"          why: {step.reason}")
        for concern in concerns:
            self._say(f"       ! concern: {concern}")

        frame = raw.copy()
        hourly = pd.DataFrame()
        cleaned_events = pd.DataFrame()
        records: list[AgentStepRecord] = []
        audits: dict[str, Any] = {}

        for index, step in enumerate(steps, start=1):
            tool = get_tool(step.tool)
            self._say(f"\n[ACT {index}/{len(steps)}] {tool.name}")

            rows_before = len(frame)
            started = time.perf_counter()

            kwargs = dict(step.kwargs or {})
            # The refiner needs the pre-expansion hourly table for self-supervised fitting.
            if tool.name == "refine_missing_values" and not hourly.empty:
                kwargs.setdefault("hourly_raw", hourly)

            frame, audit = tool.run(frame, **kwargs)
            duration = time.perf_counter() - started
            audits[tool.name] = audit

            if tool.name == "validate_ranges":
                cleaned_events = frame.copy()
            if tool.name == "build_hourly_timeseries":
                hourly = frame.copy()

            self._say(f"[OBSERVE] {_summarise_audit(tool.name, audit)}")

            check_function = CHECKS.get(tool.name)
            raw_checks = check_function(frame, audit) if check_function else []
            checks = [{"name": name, "passed": bool(passed), "detail": detail} for name, passed, detail in raw_checks]
            failed = [check for check in checks if not check["passed"]]
            repaired = False

            attempts = 0
            while failed and attempts < self.max_repair_attempts:
                attempts += 1
                self._say(f"[VALIDATE] FAILED: {', '.join(check['name'] for check in failed)}")
                self._say(f"[REFINE {attempts}] Attempting a documented repair ...")
                repair = self._repair(tool.name, frame, failed)
                if repair is None:
                    self._say("[REFINE] No repair defined for this failure; the agent will stop.")
                    break
                frame, description = repair
                repaired = True
                self._say(f"[REFINE] {description}")
                raw_checks = check_function(frame, audit) if check_function else []
                checks = [{"name": name, "passed": bool(passed), "detail": detail} for name, passed, detail in raw_checks]
                failed = [check for check in checks if not check["passed"]]

            if failed:
                record = AgentStepRecord(index, tool.name, step.reason, audit, checks, rows_before, len(frame), duration, repaired)
                records.append(record)
                if trace_path is not None:
                    _write_trace(trace_path, AgentResult(cleaned_events, hourly, frame, profile_before, {}, records, concerns, audits), self.planner.name)
                raise RuntimeError(
                    f"Postcondition failure in {tool.name} that no repair resolved: "
                    f"{[check['name'] for check in failed]}. The run was halted and the partial "
                    f"trajectory written to {trace_path}."
                )

            self._say(f"[VALIDATE] {len(checks)} check(s) passed" if checks else "[VALIDATE] no checks defined")
            records.append(AgentStepRecord(index, tool.name, step.reason, audit, checks, rows_before, len(frame), duration, repaired))

        if cleaned_events.empty:
            cleaned_events = frame.copy()

        profile_after = profile_data(
            frame, datetime_columns=("hour",) if "hour" in frame.columns else ()
        )

        result = AgentResult(
            cleaned_events=cleaned_events,
            hourly=hourly,
            processed=frame,
            profile_before=profile_before,
            profile_after=profile_after,
            steps=records,
            concerns=concerns,
            audits=audits,
        )

        self._say("\n" + "=" * 78)
        self._say(
            f"RUN COMPLETE  |  {len(records)} tools executed  |  "
            f"{sum(len(record.checks) for record in records)} checks  |  "
            f"all passed: {result.all_checks_passed}"
        )
        self._say("=" * 78)

        if trace_path is not None:
            _write_trace(trace_path, result, self.planner.name)
            self._say(f"Agent trajectory written to {trace_path}")

        return result


def _write_trace(path: Path, result: AgentResult, planner_name: str) -> None:
    """Serialise the agent trajectory to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result.to_trace(planner_name)), indent=2), encoding="utf-8")


def _summarise_audit(tool_name: str, audit: dict[str, Any]) -> str:
    """One-line human summary of a tool's audit, printed during the run."""
    if tool_name == "remove_duplicates":
        return f"removed {audit['duplicates_removed']:,} duplicate rows ({audit['rows_before']:,} -> {audit['rows_after']:,})"
    if tool_name == "standardise_datetimes":
        window = audit.get("window_reconciliation", {})
        return (
            f"{audit.get('formats_before')} spellings -> 1 | "
            f"{audit.get('total_naive_parse_disagreements', 0)} cells would have been mis-parsed by naive parsing | "
            f"median stay {window.get('length_of_stay_hours', {}).get('median')} h"
        )
    if tool_name == "convert_units":
        temperature = audit.get("temperature", {})
        return (
            f"temperature: {temperature.get('converted_from_fahrenheit', 0)} values converted from Fahrenheit, "
            f"mean {temperature.get('before', {}).get('mean')} -> {temperature.get('after', {}).get('mean')} C"
        )
    if tool_name == "standardise_categories":
        total_before = sum(detail["distinct_before"] for detail in audit.values())
        total_after = sum(detail["distinct_after"] for detail in audit.values())
        return f"{total_before} distinct category values collapsed to {total_after}"
    if tool_name == "validate_ranges":
        return f"{audit['total_cells_rejected']:,} implausible cells set to missing (rows retained)"
    if tool_name == "build_hourly_timeseries":
        return (
            f"{audit['rows_before']:,} event rows -> {audit['rows_after']:,} hourly rows "
            f"({audit['expansion_factor']}x) across {audit['admissions']} admissions"
        )
    if tool_name == "build_value_mask_decay":
        channel = next(iter(audit.get("channels", {}).values()), {})
        return (
            f"Value-Mask-Decay built for {len(audit.get('channels', {}))} channels | "
            f"example observed rate {channel.get('observed_pct')}%"
        )
    if tool_name == "refine_missing_values":
        return (
            f"{audit['cells_refined']:,} imputed cells refined | "
            f"{audit['cells_protected_observed']:,} observed cells protected | "
            f"observed cells modified: {audit['observed_cells_modified']}"
        )
    return "completed"
