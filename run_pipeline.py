"""
Module purpose:

End-to-end headless runner for the whole project.

One command regenerates the dataset, runs the agent, benchmarks the imputer, writes
every report and renders every figure::

    python run_pipeline.py

This exists for three reasons. It is the continuous-integration entry point, so a
broken commit is caught without opening a notebook. It is the fallback for the
conference talk if the notebook kernel misbehaves on stage. And it is the shortest
honest answer to "does this actually run?" - the project's own claim to
reproducibility is only worth what a single reproducible command can demonstrate.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from agents.preprocessing_agent import PreprocessingAgent  # noqa: E402
from agents.validation_agent import build_validation_report, write_html_summary  # noqa: E402
from agents.llm_planner import build_planner  # noqa: E402
from config import (  # noqa: E402
    AGENT_TRACE_JSON,
    CLEANED_CSV,
    CLEANING_SUMMARY_HTML,
    FIGURES_DIR,
    HOURLY_CSV,
    RANDOM_SEED,
    RAW_CSV,
    RAW_CSV_ORIGINAL,
    VALIDATION_REPORT_JSON,
)
from models.deeptse_imputer import benchmark_against_locf  # noqa: E402
from preprocessing.categorical_cleaner import standardise_categories  # noqa: E402
from preprocessing.synthetic_data import write_raw_dataset  # noqa: E402
from visualization import plots  # noqa: E402


def main() -> int:
    """Run the complete pipeline and write every artefact.

    Returns
    -------
    int
        ``0`` when every validation assertion holds, ``1`` otherwise - so the command
        can gate a CI job directly.
    """
    parser = argparse.ArgumentParser(description="Run the medical preprocessing agent end to end.")
    parser.add_argument(
        "--use-original-csv",
        action="store_true",
        help="Run against the author's original raw file instead of regenerating the calibrated one.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for data generation.")
    parser.add_argument("--llm", action="store_true", help="Use the LLM planner when an API key is available.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering.")
    arguments = parser.parse_args()

    started = time.perf_counter()

    if arguments.use_original_csv:
        source = Path(RAW_CSV_ORIGINAL)
        if not source.exists():
            print(f"Original CSV not found at {source}", file=sys.stderr)
            return 2
        print(f"[1/6] Using the author's original raw file: {source.name}")
        ground_truth = None
    else:
        print("[1/6] Generating the calibrated synthetic dataset ...")
        _, ground_truth = write_raw_dataset(seed=arguments.seed)
        source = Path(RAW_CSV)

    raw = pd.read_csv(source, dtype=str)
    print(f"      {len(raw):,} rows x {raw.shape[1]} columns loaded from {source.name}")

    print("\n[2/6] Running the preprocessing agent ...")
    planner = build_planner(prefer_llm=arguments.llm)
    try:
        result = PreprocessingAgent(planner=planner).run(raw)
    except RuntimeError as error:
        # A halt is a designed outcome, not a crash. It means a postcondition failed and
        # no enumerated repair restored it, so the agent refused to write out a dataset
        # it could not vouch for. The author's original raw file triggers exactly this:
        # roughly 30% of heart_rate, spo2 and age are a single sentinel constant, so
        # range validation removes more than half of those channels.
        print("\n" + "=" * 78)
        print("AGENT HALTED — no dataset was written.")
        print("=" * 78)
        print(f"\n{error}\n")
        print("This is the intended behaviour when a quality gate fails. Inspect the")
        print(f"partial trajectory in {AGENT_TRACE_JSON.name}, fix the input or the plan,")
        print("and re-run. The agent does not emit a file it cannot vouch for.")
        return 1

    print("\n[3/6] Writing cleaned datasets ...")
    result.cleaned_events.to_csv(CLEANED_CSV, index=False)
    result.processed.to_csv(HOURLY_CSV, index=False)
    print(f"      {CLEANED_CSV.name}: {len(result.cleaned_events):,} rows")
    print(f"      {HOURLY_CSV.name}: {len(result.processed):,} rows x {result.processed.shape[1]} columns")

    print("\n[4/6] Benchmarking imputation against LOCF ...")
    benchmark = benchmark_against_locf(result.hourly, holdout_mode="block")
    overall = benchmark.get("overall", {})
    print(
        f"      pooled standardised MAE {overall.get('locf_standardised_mae')} -> "
        f"{overall.get('deeptse_standardised_mae')} ({overall.get('improvement_pct', 0):+.2f}%), "
        f"{overall.get('channels_improved')}/{overall.get('channels_tested')} channels improved"
    )

    print("\n[5/6] Validating the run ...")
    report = build_validation_report(raw, result, benchmark=benchmark, ground_truth=ground_truth)
    write_html_summary(report)
    verdict = report["verdict"]
    print(
        f"      {verdict['assertions_total'] - verdict['assertions_failed']}/"
        f"{verdict['assertions_total']} assertions passed"
    )
    if not verdict["passed"]:
        print(f"      FAILED: {verdict['failed_assertions']}")
    print(f"      {VALIDATION_REPORT_JSON.name} and {CLEANING_SUMMARY_HTML.name} written")

    if not arguments.no_figures:
        print("\n[6/6] Rendering figures ...")
        plots.apply_house_style()
        missing = report["missing_values"]
        duplicate_audit = result.audits["remove_duplicates"]
        hourly_audit = result.audits["build_hourly_timeseries"]
        _, categorical_audit = standardise_categories(raw)

        plots.plot_missingness_before_after(
            missing["raw_event_level_pct"], missing["hourly_before_imputation_pct"], missing["after_imputation_pct"]
        )
        plots.plot_duplicate_comparison(
            duplicate_audit["duplicates_removed"], 0, duplicate_audit["rows_before"], duplicate_audit["rows_after"]
        )
        plots.plot_temperature_unit_problem(
            raw["temperature"].astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False),
            result.cleaned_events["temperature"],
        )
        plots.plot_category_consolidation(categorical_audit)
        plots.plot_timeseries_expansion(
            hourly_audit["rows_before"], hourly_audit["rows_after"], hourly_audit["hours_per_admission"]
        )
        plots.plot_decay_curve()
        plots.plot_value_mask_decay_example(result.processed)
        plots.plot_patient_trajectory(result.processed)
        plots.plot_imputation_benchmark(benchmark)
        plots.plot_missingness_heatmap(result.processed)
        print(f"      10 figures written to {FIGURES_DIR}")
    else:
        print("\n[6/6] Figures skipped.")

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed:.1f}s. Launch the dashboard with:  streamlit run visualization/dashboard.py")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
