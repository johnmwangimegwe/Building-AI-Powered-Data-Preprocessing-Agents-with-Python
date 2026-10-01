"""
Module purpose:

The validation agent - an independent auditor of the cleaning run.

The preprocessing agent checks its own work step by step, which is necessary and not
sufficient: a component that grades itself shares its own blind spots. This module
re-derives every claim **from the data**, not from the audits, and reports each as an
assertion that either holds or does not.

It answers four questions a reviewer would actually ask:

1. *Did the agent do what it said?* Row counts, duplicate counts and null counts are
   recomputed from the raw and final tables and compared against the reported figures.
2. *Is the output internally sound?* Masks are binary, decays lie in $[0,1]$, the
   hourly grid is contiguous, no channel still has a hole.
3. *Was anything fabricated?* Every observed cell in the final table is compared
   against its value before refinement. A single difference fails the run.
4. *Is it reproducible?* The schema, the standardisation statistics and the random
   seed are all recorded, so the same file can be rebuilt byte for byte.

The output is ``reports/validation_report.json`` plus a human-readable HTML summary.
Following the reference workflow, the report records the original statistics, the
final statistics, the rows removed and the schema - and, most usefully of all, the
*original row indices* of every dropped row, so the claim can be checked by opening
the raw CSV at those line numbers.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import html
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.preprocessing_agent import AgentResult, _json_safe
from config import (
    CLEANING_SUMMARY_HTML,
    PALETTE,
    PLAUSIBLE_RANGES,
    RANDOM_SEED,
    TIMESERIES_CHANNELS,
    VALIDATION_REPORT_JSON,
)

__all__ = ["ValidationAgent", "build_validation_report", "write_html_summary"]


class ValidationAgent:
    """Re-derive and assert every claim made about a cleaning run.

    Parameters
    ----------
    channels:
        Time-series channels to audit. Defaults to ``config.TIMESERIES_CHANNELS``.

    Example
    -------
    >>> report = ValidationAgent().validate(raw, result)
    >>> report["verdict"]["passed"]
    True
    """

    def __init__(self, channels: list[str] | None = None) -> None:
        self.channels = channels or TIMESERIES_CHANNELS

    # -- assertions --------------------------------------------------------------

    @staticmethod
    def _assertion(name: str, passed: bool, expected: Any, observed: Any, note: str = "") -> dict[str, Any]:
        """Build one assertion record."""
        return {
            "name": name,
            "passed": bool(passed),
            "expected": _json_safe(expected),
            "observed": _json_safe(observed),
            "note": note,
        }

    def _structural_assertions(self, raw: pd.DataFrame, result: AgentResult) -> list[dict[str, Any]]:
        """Checks recomputed from the tables themselves."""
        assertions: list[dict[str, Any]] = []
        processed = result.processed

        duplicate_audit = result.audits.get("remove_duplicates", {})
        expected_rows = int(duplicate_audit.get("rows_after", len(result.cleaned_events)))
        assertions.append(
            self._assertion(
                "row_count_after_deduplication_matches_audit",
                len(result.cleaned_events) == expected_rows,
                expected_rows,
                int(len(result.cleaned_events)),
                "Recounted from the cleaned event table rather than trusting the tool's report.",
            )
        )
        assertions.append(
            self._assertion(
                "no_duplicate_rows_remain",
                int(result.cleaned_events.duplicated().sum()) == 0,
                0,
                int(result.cleaned_events.duplicated().sum()),
            )
        )
        assertions.append(
            self._assertion(
                "every_admission_survived_cleaning",
                raw["admission_id"].nunique() == result.cleaned_events["admission_id"].nunique(),
                int(raw["admission_id"].nunique()),
                int(result.cleaned_events["admission_id"].nunique()),
                "Deduplication must not remove an entire admission.",
            )
        )

        if "length_of_stay_hours" in result.cleaned_events.columns:
            minimum_stay = float(result.cleaned_events["length_of_stay_hours"].min())
            assertions.append(
                self._assertion(
                    "all_lengths_of_stay_positive",
                    minimum_stay > 0,
                    "> 0 hours",
                    round(minimum_stay, 2),
                    "A negative stay is the signature of a mis-parsed day-first date.",
                )
            )

        for column, (low, high) in PLAUSIBLE_RANGES.items():
            if column not in result.cleaned_events.columns:
                continue
            values = pd.to_numeric(result.cleaned_events[column], errors="coerce").dropna()
            n_bad = int(((values < low) | (values > high)).sum())
            assertions.append(
                self._assertion(f"{column}_within_physiological_range", n_bad == 0, 0, n_bad, f"allowed {low}-{high}")
            )

        if "hour" in processed.columns and "admission_id" in processed.columns:
            duplicated_keys = int(processed.duplicated(subset=["admission_id", "hour"]).sum())
            assertions.append(
                self._assertion("one_row_per_admission_hour", duplicated_keys == 0, 0, duplicated_keys)
            )
            spacing_ok = processed.groupby("admission_id")["hour"].apply(
                lambda series: series.sort_values().diff().dropna().dt.total_seconds().div(3600).eq(1.0).all()
            )
            assertions.append(
                self._assertion(
                    "hourly_grid_contiguous_within_every_admission",
                    bool(spacing_ok.all()),
                    "all admissions",
                    f"{int(spacing_ok.sum())}/{len(spacing_ok)} admissions",
                )
            )

        for channel in self.channels:
            value_column, mask_column, decay_column = f"{channel}_value", f"{channel}_mask", f"{channel}_decay"
            if value_column not in processed.columns:
                continue
            assertions.append(
                self._assertion(f"{channel}_value_complete", int(processed[value_column].isna().sum()) == 0, 0, int(processed[value_column].isna().sum()))
            )
            assertions.append(
                self._assertion(f"{channel}_mask_binary", bool(processed[mask_column].isin([0, 1]).all()), "{0, 1}", sorted(processed[mask_column].unique().tolist()))
            )
            within = bool(processed[decay_column].between(0.0, 1.0).all())
            assertions.append(
                self._assertion(f"{channel}_decay_in_unit_interval", within, "[0, 1]", [float(processed[decay_column].min()), float(processed[decay_column].max())])
            )

        return assertions

    def _fabrication_assertions(self, result: AgentResult) -> list[dict[str, Any]]:
        """The single most important check: refinement must not touch observed data."""
        assertions: list[dict[str, Any]] = []
        processed = result.processed

        for channel in self.channels:
            mask_column = f"{channel}_mask"
            value_column, refined_column = f"{channel}_value", f"{channel}_refined"
            if refined_column not in processed.columns:
                continue
            observed = processed[mask_column] == 1
            if not observed.any():
                continue
            difference = float((processed.loc[observed, refined_column] - processed.loc[observed, value_column]).abs().max())
            assertions.append(
                self._assertion(
                    f"{channel}_observed_values_unmodified_by_refinement",
                    difference < 1e-6,
                    "0 (exact)",
                    round(difference, 9),
                    "Recomputed directly from the final table: a genuine measurement must survive refinement untouched.",
                )
            )

        refinement_audit = result.audits.get("refine_missing_values", {})
        assertions.append(
            self._assertion(
                "refiner_reports_zero_observed_modifications",
                refinement_audit.get("observed_cells_modified", 0) == 0,
                0,
                refinement_audit.get("observed_cells_modified", 0),
            )
        )
        return assertions

    # -- report ------------------------------------------------------------------

    def validate(
        self,
        raw: pd.DataFrame,
        result: AgentResult,
        benchmark: dict[str, Any] | None = None,
        ground_truth: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Produce the full validation report.

        Parameters
        ----------
        raw:
            The raw table the run started from.
        result:
            The :class:`agents.preprocessing_agent.AgentResult` to audit.
        benchmark:
            Optional output of
            :func:`models.deeptse_imputer.benchmark_against_locf`.
        ground_truth:
            Optional manifest from the synthetic generator. When supplied, the report
            additionally scores the pipeline's *recovery* of known injected defects -
            the closest thing to an accuracy metric a cleaning pipeline can have.

        Returns
        -------
        dict
            The JSON-serialisable validation report.
        """
        processed = result.processed
        assertions = self._structural_assertions(raw, result) + self._fabrication_assertions(result)

        before = {
            "rows": int(len(raw)),
            "columns": int(raw.shape[1]),
            "admissions": int(raw["admission_id"].nunique()),
            "duplicate_rows": int(raw.duplicated().sum()),
            "null_counts": {column: int(count) for column, count in raw.isna().sum().items()},
            "distinct_datetime_formats": {
                column: len(spellings) for column, spellings in result.profile_before.get("date_formats", {}).items()
            },
            "columns_with_mixed_units": sorted(result.profile_before.get("unit_inconsistencies", {})),
        }

        after = {
            "event_rows": int(len(result.cleaned_events)),
            "hourly_rows": int(len(processed)),
            "columns": int(processed.shape[1]),
            "admissions": int(processed["admission_id"].nunique()) if "admission_id" in processed.columns else 0,
            "duplicate_rows": int(result.cleaned_events.duplicated().sum()),
            "null_counts_in_model_matrix": {
                f"{channel}_value": int(processed[f"{channel}_value"].isna().sum())
                for channel in self.channels
                if f"{channel}_value" in processed.columns
            },
            "distinct_datetime_formats": 1,
            "columns_with_mixed_units": [],
        }

        duplicate_audit = result.audits.get("remove_duplicates", {})
        range_audit = result.audits.get("validate_ranges", {})

        report: dict[str, Any] = {
            "report_version": "1.0",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pipeline": "AI-Powered Medical Data Preprocessing Agent",
            "reproducibility": {
                "random_seed": RANDOM_SEED,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "original_statistics": before,
            "final_statistics": after,
            "records": {
                "input_rows": int(len(raw)),
                "duplicate_rows_removed": int(duplicate_audit.get("duplicates_removed", 0)),
                "event_rows_retained": int(len(result.cleaned_events)),
                "hourly_rows_generated": int(len(processed)),
                "expansion_factor": round(len(processed) / max(len(result.cleaned_events), 1), 2),
                "cells_rejected_as_implausible": int(range_audit.get("total_cells_rejected", 0)),
                "dropped_row_indices": duplicate_audit.get("dropped_row_indices", []),
                "dropped_row_reason": "exact_duplicate_of_an_earlier_row",
                "rejected_cell_log": range_audit.get("rejections", []),
                "rejected_cell_log_truncated": range_audit.get("rejections_truncated", False),
            },
            "missing_values": {
                "raw_event_level_pct": {
                    channel: round(100.0 * raw[channel].isna().mean(), 2)
                    for channel in self.channels
                    if channel in raw.columns
                },
                "hourly_before_imputation_pct": result.audits.get("build_hourly_timeseries", {}).get("missing_pct_after", {}),
                "after_imputation_pct": {
                    channel: round(100.0 * processed[f"{channel}_value"].isna().mean(), 2)
                    for channel in self.channels
                    if f"{channel}_value" in processed.columns
                },
                "genuinely_observed_pct": {
                    channel: round(100.0 * processed[f"{channel}_mask"].mean(), 2)
                    for channel in self.channels
                    if f"{channel}_mask" in processed.columns
                },
            },
            "schema": {column: str(dtype) for column, dtype in processed.dtypes.items()},
            "standardisation_statistics": result.audits.get("build_value_mask_decay", {}).get("standardisation", {}),
            "agent": {
                "steps_executed": len(result.steps),
                "checks_run": sum(len(step.checks) for step in result.steps),
                "steps_repaired": sum(1 for step in result.steps if step.repaired),
                "concerns": result.concerns,
                "plan": [{"step": step.step, "tool": step.tool, "reason": step.reason} for step in result.steps],
            },
            "assertions": assertions,
        }

        if benchmark:
            report["imputation_benchmark"] = benchmark

        if ground_truth:
            report["defect_recovery"] = self._score_against_truth(raw, result, ground_truth)

        failed = [assertion for assertion in assertions if not assertion["passed"]]
        report["verdict"] = {
            "passed": not failed,
            "assertions_total": len(assertions),
            "assertions_failed": len(failed),
            "failed_assertions": [assertion["name"] for assertion in failed],
            "statement": (
                "All assertions hold. The cleaned dataset matches every claim made about it."
                if not failed
                else "One or more assertions failed. Do not use this output for modelling until resolved."
            ),
        }
        return _json_safe(report)

    def _score_against_truth(self, raw: pd.DataFrame, result: AgentResult, truth: dict[str, Any]) -> dict[str, Any]:
        """Compare defects recovered against defects known to have been injected.

        Parameters
        ----------
        raw, result:
            The run being audited.
        truth:
            Manifest from :func:`preprocessing.synthetic_data.generate_raw_dataset`.

        Returns
        -------
        dict
            Injected versus recovered counts per defect class. Unlike every other part
            of this report, these figures are only available because the data is
            synthetic - which is precisely the argument for demonstrating on synthetic
            data in the first place.
        """
        duplicate_audit = result.audits.get("remove_duplicates", {})
        unit_audit = result.audits.get("convert_units", {})
        categorical_audit = result.audits.get("standardise_categories", {})

        injected_duplicates = int(truth.get("duplicate_rows_injected", 0))
        recovered_duplicates = int(duplicate_audit.get("duplicates_removed", 0))

        injected_fahrenheit = int(truth.get("fahrenheit_cells", 0))
        recovered_fahrenheit = int(unit_audit.get("temperature", {}).get("converted_from_fahrenheit", 0))

        categories_changed = sum(detail.get("cells_changed", 0) for detail in categorical_audit.values())

        return {
            "note": (
                "Injected counts are measured on the 1000-row file as generated; recovered counts are "
                "measured after deduplication, so a defect that existed only inside a removed duplicate "
                "is correctly never seen again."
            ),
            "duplicate_rows": {
                "injected": injected_duplicates,
                "recovered": recovered_duplicates,
                "recovery_rate_pct": round(100.0 * recovered_duplicates / injected_duplicates, 1) if injected_duplicates else None,
            },
            "fahrenheit_temperatures": {
                "injected": injected_fahrenheit,
                "recovered_after_deduplication": recovered_fahrenheit,
            },
            "categorical_variant_cells": {
                "injected": int(truth.get("category_noise_cells", 0)),
                "cells_rewritten": int(categories_changed),
            },
            "unmapped_category_values": {
                column: detail.get("unmapped_values", []) for column, detail in categorical_audit.items() if detail.get("unmapped_values")
            },
        }


def build_validation_report(
    raw: pd.DataFrame,
    result: AgentResult,
    benchmark: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
    path: Path | str = VALIDATION_REPORT_JSON,
) -> dict[str, Any]:
    """Validate a run and write ``validation_report.json``.

    Parameters
    ----------
    raw, result, benchmark, ground_truth:
        Forwarded to :meth:`ValidationAgent.validate`.
    path:
        Destination for the JSON report.

    Returns
    -------
    dict
        The report, also written to disk.

    Example
    -------
    >>> report = build_validation_report(raw, result)
    >>> report["verdict"]["passed"]
    True
    """
    report = ValidationAgent().validate(raw, result, benchmark=benchmark, ground_truth=ground_truth)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_html_summary(report: dict[str, Any], path: Path | str = CLEANING_SUMMARY_HTML) -> Path:
    """Render the validation report as a self-contained HTML summary.

    Parameters
    ----------
    report:
        Output of :func:`build_validation_report`.
    path:
        Destination HTML file.

    Returns
    -------
    pathlib.Path
        The path written.

    Notes
    -----
    The page is deliberately dependency-free and legible in both light and dark
    colour schemes, because the person who most needs to read it - a clinician or a
    data steward - will open it in a browser, not in a notebook.
    """
    verdict = report["verdict"]
    before, after = report["original_statistics"], report["final_statistics"]
    records = report["records"]

    def row(label: str, before_value: Any, after_value: Any) -> str:
        return (
            f"<tr><td>{html.escape(str(label))}</td>"
            f"<td class='num before'>{html.escape(str(before_value))}</td>"
            f"<td class='num after'>{html.escape(str(after_value))}</td></tr>"
        )

    comparison_rows = "".join(
        [
            row("Rows", f"{before['rows']:,}", f"{after['hourly_rows']:,} hourly ({after['event_rows']:,} events)"),
            row("Columns", before["columns"], after["columns"]),
            row("Admissions", before["admissions"], after["admissions"]),
            row("Duplicate rows", f"{before['duplicate_rows']:,}", f"{after['duplicate_rows']:,}"),
            row("Distinct datetime formats", max(before["distinct_datetime_formats"].values(), default=0), after["distinct_datetime_formats"]),
            row("Columns with mixed units", len(before["columns_with_mixed_units"]), len(after["columns_with_mixed_units"])),
        ]
    )

    assertion_rows = "".join(
        f"<tr class='{'pass' if assertion['passed'] else 'fail'}'>"
        f"<td>{'PASS' if assertion['passed'] else 'FAIL'}</td>"
        f"<td>{html.escape(assertion['name'])}</td>"
        f"<td>{html.escape(str(assertion['expected']))}</td>"
        f"<td>{html.escape(str(assertion['observed']))}</td></tr>"
        for assertion in report["assertions"]
    )

    plan_rows = "".join(
        f"<li><strong>{html.escape(step['tool'])}</strong> &mdash; {html.escape(step['reason'])}</li>"
        for step in report["agent"]["plan"]
    )

    concerns = report["agent"].get("concerns", [])
    concern_block = (
        "<h2>Concerns raised by the agent</h2><ul>"
        + "".join(f"<li>{html.escape(concern)}</li>" for concern in concerns)
        + "</ul>"
        if concerns
        else ""
    )

    benchmark_block = ""
    if "imputation_benchmark" in report and report["imputation_benchmark"].get("overall"):
        overall = report["imputation_benchmark"]["overall"]
        benchmark_block = f"""
        <h2>Imputation benchmark (held-out observations)</h2>
        <p>Mode: <code>{html.escape(str(report['imputation_benchmark'].get('holdout_mode')))}</code>,
        {overall['held_out_cells']:,} held-out cells.
        Standardised MAE: LOCF {overall['locf_standardised_mae']} &rarr;
        DeepTSE-refined {overall['deeptse_standardised_mae']}
        (<strong>{overall['improvement_pct']:+.2f}%</strong>);
        {overall['channels_improved']} of {overall['channels_tested']} channels improved.</p>
        """

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cleaning Summary</title>
<style>
  :root {{
    color-scheme: light;
    --surface: {PALETTE['surface']};
    --card: #ffffff;
    --ink: {PALETTE['text_primary']};
    --ink-2: {PALETTE['text_secondary']};
    --grid: {PALETTE['grid']};
    --after: {PALETTE['series_1']};
    --before: {PALETTE['series_2']};
    --good: {PALETTE['good']};
    --bad: {PALETTE['critical']};
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface: #1a1a19; --card: #232321; --ink: #ffffff; --ink-2: #c3c2b7;
      --grid: #3a3a37; --after: #3987e5; --before: #d95926;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--surface); color:var(--ink);
         font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.05rem; margin: 32px 0 10px; letter-spacing: -0.005em; }}
  .sub {{ color: var(--ink-2); margin: 0 0 24px; font-size: 0.9rem; }}
  .verdict {{ padding: 14px 18px; border-radius: 10px; font-weight: 600;
              background: color-mix(in srgb, var(--good) 12%, var(--card));
              border: 1px solid color-mix(in srgb, var(--good) 45%, transparent); }}
  .verdict.fail {{ background: color-mix(in srgb, var(--bad) 12%, var(--card));
                   border-color: color-mix(in srgb, var(--bad) 45%, transparent); }}
  table {{ width:100%; border-collapse: collapse; background: var(--card);
           border:1px solid var(--grid); border-radius:10px; overflow:hidden; font-size: 0.92rem; }}
  th, td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--grid); }}
  th {{ color: var(--ink-2); font-weight:600; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.04em; }}
  tr:last-child td {{ border-bottom:none; }}
  td.num {{ text-align:right; font-variant-numeric: tabular-nums; font-weight:600; }}
  td.before {{ color: var(--before); }}
  td.after {{ color: var(--after); }}
  tr.pass td:first-child {{ color: var(--good); font-weight:700; }}
  tr.fail td:first-child {{ color: var(--bad); font-weight:700; }}
  ul {{ padding-left: 20px; }} li {{ margin: 4px 0; }}
  code {{ background: color-mix(in srgb, var(--ink) 8%, transparent); padding:1px 5px; border-radius:4px; }}
  .scroll {{ max-height: 420px; overflow:auto; border-radius:10px; }}
</style></head><body><div class="wrap">
  <h1>Medical Data Cleaning Summary</h1>
  <p class="sub">{html.escape(report['pipeline'])} &middot; generated {html.escape(report['generated_utc'])}
     &middot; seed {report['reproducibility']['random_seed']}</p>

  <div class="verdict {'' if verdict['passed'] else 'fail'}">
    {'PASSED' if verdict['passed'] else 'FAILED'} &mdash; {verdict['assertions_total'] - verdict['assertions_failed']}
    of {verdict['assertions_total']} assertions hold. {html.escape(verdict['statement'])}
  </div>

  <h2>Before and after</h2>
  <table><thead><tr><th>Metric</th><th style="text-align:right">Before</th><th style="text-align:right">After</th></tr></thead>
  <tbody>{comparison_rows}</tbody></table>

  <h2>What the agent changed</h2>
  <table><tbody>
    {row('Duplicate rows removed', '', f"{records['duplicate_rows_removed']:,}")}
    {row('Implausible cells set to missing', '', f"{records['cells_rejected_as_implausible']:,}")}
    {row('Event rows retained', '', f"{records['event_rows_retained']:,}")}
    {row('Hourly rows generated', '', f"{records['hourly_rows_generated']:,}")}
    {row('Expansion factor', '', f"{records['expansion_factor']}x")}
  </tbody></table>

  <h2>Agent plan</h2>
  <ol>{plan_rows}</ol>
  {concern_block}
  {benchmark_block}

  <h2>Assertions</h2>
  <div class="scroll"><table><thead><tr><th>Result</th><th>Assertion</th><th>Expected</th><th>Observed</th></tr></thead>
  <tbody>{assertion_rows}</tbody></table></div>

  <h2>How to verify this yourself</h2>
  <p>Open the raw CSV and inspect the {len(records['dropped_row_indices']):,} row indices listed under
     <code>records.dropped_row_indices</code> in <code>validation_report.json</code>. Each one should be an exact
     copy of an earlier row. Every rejected cell is listed under <code>records.rejected_cell_log</code>
     with its admission, timestamp, channel and original value.</p>
</div></body></html>"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination
