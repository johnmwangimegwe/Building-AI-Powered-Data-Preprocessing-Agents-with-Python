"""
Module purpose:

Streamlit dashboard summarising a completed preprocessing run.

The dashboard is the *reviewer's* view of the pipeline. Where the notebook explains
how each step works, this shows only what happened and whether it can be trusted: the
headline numbers, the before/after comparisons, the assertion table, and a download of
the model-ready dataset.

It reads exclusively from artefacts already on disk - ``reports/validation_report.json``
and the CSV files in ``data/`` - and never re-runs the pipeline. That separation is
deliberate: a dashboard that recomputes its own numbers can disagree with the report,
and then nobody knows which to believe.

Run it with::

    streamlit run visualization/dashboard.py

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    AGENT_TRACE_JSON,
    CLEANED_CSV,
    HOURLY_CSV,
    PALETTE,
    RAW_CSV,
    TIMESERIES_CHANNELS,
    VALIDATION_REPORT_JSON,
)

BEFORE = PALETTE["series_2"]
AFTER = PALETTE["series_1"]
THIRD = PALETTE["series_3"]

st.set_page_config(page_title="Medical AI Cleaning Dashboard", page_icon="🩺", layout="wide")


@st.cache_data(show_spinner=False)
def load_report(path: str) -> dict | None:
    """Load the validation report, or ``None`` when the pipeline has not been run."""
    file = Path(path)
    if not file.exists():
        return None
    return json.loads(file.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    """Load a pipeline CSV artefact, or ``None`` when it is missing."""
    file = Path(path)
    if not file.exists():
        return None
    return pd.read_csv(file)


def styled_bar(
    categories: list[str],
    series: dict[str, list[float]],
    title: str,
    y_title: str,
    colours: list[str] | None = None,
) -> go.Figure:
    """Grouped bar chart following the project's visual contract.

    Parameters
    ----------
    categories:
        X-axis category labels.
    series:
        ``{series name: values}``, assigned colours in the order given.
    title, y_title:
        Chart title and y-axis label.
    colours:
        Explicit colour order. Defaults to before-orange, after-blue, third-aqua, which
        keeps the dashboard consistent with the notebook: orange is always the state
        before a transformation and blue the state after. Charts whose series are not in
        that order pass their own list rather than relying on the default.

    Returns
    -------
    plotly.graph_objects.Figure
        A hover-enabled figure. Interactivity is the default here rather than an
        extra: a reader who wants the exact value should be able to get it without
        the chart being cluttered by a label on every bar.
    """
    colours = colours or [BEFORE, AFTER, THIRD]
    figure = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        figure.add_bar(
            name=name,
            x=categories,
            y=values,
            marker_color=colours[index % len(colours)],
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}}<extra></extra>",
        )
    figure.update_layout(
        title=title,
        yaxis_title=y_title,
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        template="simple_white",
        height=380,
        margin=dict(l=10, r=10, t=56, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return figure


st.title("Medical AI Cleaning Dashboard")
st.caption("From messy hospital admission records to validated hourly clinical time series")

report = load_report(str(VALIDATION_REPORT_JSON))

if report is None:
    st.error(
        "No validation report found. Run the pipeline first:\n\n"
        "```bash\npython run_pipeline.py\n```\n\n"
        "or execute `Medical_AI_Preprocessing_Agent.ipynb` end to end."
    )
    st.stop()

before = report["original_statistics"]
after = report["final_statistics"]
records = report["records"]
verdict = report["verdict"]

# ------------------------------------------------------------------------------------
# Verdict banner
# ------------------------------------------------------------------------------------
if verdict["passed"]:
    st.success(
        f"**VALIDATION PASSED** — {verdict['assertions_total']} assertions checked, all hold. "
        f"{verdict['statement']}"
    )
else:
    st.error(
        f"**VALIDATION FAILED** — {verdict['assertions_failed']} of {verdict['assertions_total']} "
        f"assertions failed: {', '.join(verdict['failed_assertions'])}"
    )

# ------------------------------------------------------------------------------------
# Headline metrics
# ------------------------------------------------------------------------------------
st.subheader("Dataset overview")
columns = st.columns(5)
columns[0].metric("Admissions", f"{before['admissions']:,}")
columns[1].metric("Raw event rows", f"{before['rows']:,}", f"-{records['duplicate_rows_removed']:,} duplicates")
columns[2].metric("Hourly rows", f"{after['hourly_rows']:,}", f"{records['expansion_factor']}x expansion")
columns[3].metric("Implausible cells rejected", f"{records['cells_rejected_as_implausible']:,}")
columns[4].metric("Assertions passed", f"{verdict['assertions_total'] - verdict['assertions_failed']}/{verdict['assertions_total']}")

st.divider()

# ------------------------------------------------------------------------------------
# Cleaning performance
# ------------------------------------------------------------------------------------
st.subheader("Cleaning performance")
left, right = st.columns(2)

with left:
    st.plotly_chart(
        styled_bar(
            ["Duplicate rows", "Distinct datetime formats", "Columns with mixed units"],
            {
                "Before": [
                    before["duplicate_rows"],
                    max(before["distinct_datetime_formats"].values(), default=0),
                    len(before["columns_with_mixed_units"]),
                ],
                "After": [
                    after["duplicate_rows"],
                    after["distinct_datetime_formats"],
                    len(after["columns_with_mixed_units"]),
                ],
            },
            "Structural defects removed",
            "Count",
        ),
        use_container_width=True,
    )

with right:
    missing = report["missing_values"]
    channels = [channel for channel in TIMESERIES_CHANNELS if channel in missing["raw_event_level_pct"]]
    st.plotly_chart(
        styled_bar(
            [channel.replace("_", " ") for channel in channels],
            {
                "Raw events": [missing["raw_event_level_pct"].get(channel, 0) for channel in channels],
                "After hourly alignment": [missing["hourly_before_imputation_pct"].get(channel, 0) for channel in channels],
                "Final matrix": [missing["after_imputation_pct"].get(channel, 0) for channel in channels],
            },
            "Missing values by pipeline stage (%)",
            "Cells missing (%)",
            colours=[BEFORE, THIRD, AFTER],
        ),
        use_container_width=True,
    )

st.caption(
    "Hourly alignment **raises** measured missingness: the event-based layout was concealing every "
    "hour in which nothing was recorded. The pipeline solves the larger problem, not the smaller one."
)

# ------------------------------------------------------------------------------------
# Observed versus imputed
# ------------------------------------------------------------------------------------
st.subheader("How much of the final matrix is a real measurement?")
observed = report["missing_values"]["genuinely_observed_pct"]
channels = list(observed)
figure = go.Figure()
figure.add_bar(
    name="Genuinely observed (mask = 1)",
    x=[channel.replace("_", " ") for channel in channels],
    y=[observed[channel] for channel in channels],
    marker_color=AFTER,
    hovertemplate="<b>%{x}</b><br>observed: %{y:.1f}%<extra></extra>",
)
figure.add_bar(
    name="Imputed (mask = 0)",
    x=[channel.replace("_", " ") for channel in channels],
    y=[100 - observed[channel] for channel in channels],
    marker_color=BEFORE,
    hovertemplate="<b>%{x}</b><br>imputed: %{y:.1f}%<extra></extra>",
)
figure.update_layout(
    barmode="stack",
    template="simple_white",
    height=360,
    yaxis_title="Share of patient-hours (%)",
    bargap=0.3,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
)
st.plotly_chart(figure, use_container_width=True)
st.caption(
    "The mask column keeps this distinction available to the model. Without it, an imputed value "
    "and a measured one would be indistinguishable in training."
)

# ------------------------------------------------------------------------------------
# Imputation benchmark
# ------------------------------------------------------------------------------------
if "imputation_benchmark" in report and report["imputation_benchmark"].get("by_channel"):
    st.subheader("Imputation benchmark")
    benchmark = report["imputation_benchmark"]
    overall = benchmark.get("overall", {})
    benchmark_table = (
        pd.DataFrame(benchmark["by_channel"]).T.reset_index().rename(columns={"index": "channel"})
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Holdout mode", str(benchmark.get("holdout_mode")))
    metric_columns[1].metric("Hidden observations", f"{overall.get('held_out_cells', 0):,}")
    metric_columns[2].metric(
        "Pooled error reduction",
        f"{overall.get('improvement_pct', 0):+.2f}%",
        f"{overall.get('channels_improved', 0)}/{overall.get('channels_tested', 0)} channels improved",
    )

    improvements = benchmark_table.sort_values("mae_improvement_pct")
    figure = go.Figure()
    figure.add_bar(
        x=improvements["mae_improvement_pct"],
        y=improvements["channel"].str.replace("_", " "),
        orientation="h",
        marker_color=[PALETTE["good"] if value > 0 else PALETTE["critical"] for value in improvements["mae_improvement_pct"]],
        hovertemplate="<b>%{y}</b><br>error reduction: %{x:.1f}%<extra></extra>",
    )
    figure.update_layout(
        template="simple_white",
        height=380,
        xaxis_title="Reduction in mean absolute error versus LOCF (%)",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    figure.add_vline(x=0, line_color=PALETTE["text_secondary"], line_width=1)
    st.plotly_chart(figure, use_container_width=True)
    with st.expander("Raw errors per channel"):
        st.dataframe(benchmark_table, use_container_width=True, hide_index=True)
    st.caption(
        "Channels where refinement loses are shown, not hidden. Where the carried-forward value already "
        "wins, ship LOCF for that channel."
    )

# ------------------------------------------------------------------------------------
# Agent plan and assertions
# ------------------------------------------------------------------------------------
st.subheader("What the agent did, and how it was checked")
plan_tab, assertion_tab, trace_tab = st.tabs(["Plan", "Assertions", "Agent trace"])

with plan_tab:
    for step in report["agent"]["plan"]:
        st.markdown(f"**{step['step']}. `{step['tool']}`** — {step['reason']}")
    for concern in report["agent"].get("concerns", []):
        st.warning(concern)

with assertion_tab:
    assertions = pd.DataFrame(report["assertions"])
    assertions["result"] = assertions["passed"].map({True: "PASS", False: "FAIL"})
    st.dataframe(
        assertions[["result", "name", "expected", "observed", "note"]],
        use_container_width=True,
        hide_index=True,
        height=420,
    )

with trace_tab:
    trace_file = Path(AGENT_TRACE_JSON)
    if trace_file.exists():
        trace = json.loads(trace_file.read_text(encoding="utf-8"))
        st.caption(f"Planner: `{trace['planner']}` · {trace['steps_executed']} steps · all checks passed: {trace['all_checks_passed']}")
        for step in trace["trajectory"]:
            status = "PASS" if step["passed"] else "FAIL"
            repaired = " · repaired" if step.get("repaired") else ""
            with st.expander(f"{status} · step {step['step']} · {step['tool']} · {step['duration_seconds']}s{repaired}"):
                st.write(f"**Rows:** {step['rows_before']:,} → {step['rows_after']:,}")
                st.json(step["checks"])
    else:
        st.info("No agent trace on disk yet.")

# ------------------------------------------------------------------------------------
# Downloads
# ------------------------------------------------------------------------------------
st.divider()
st.subheader("Model-ready dataset")

hourly = load_csv(str(HOURLY_CSV))
cleaned = load_csv(str(CLEANED_CSV))

if hourly is not None:
    preview_channel = st.selectbox("Preview channel", TIMESERIES_CHANNELS, index=0)
    preview_columns = ["admission_id", "hour", "hours_since_admission"] + [
        f"{preview_channel}{suffix}" for suffix in ("_value", "_mask", "_decay", "_refined")
    ]
    available = [column for column in preview_columns if column in hourly.columns]
    st.dataframe(hourly[available].head(60), use_container_width=True, hide_index=True)

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download hourly time series (CSV)",
        hourly.to_csv(index=False).encode("utf-8"),
        file_name="processed_hourly_timeseries.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if cleaned is not None:
        download_columns[1].download_button(
            "Download cleaned events (CSV)",
            cleaned.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_medical_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    download_columns[2].download_button(
        "Download validation report (JSON)",
        json.dumps(report, indent=2).encode("utf-8"),
        file_name="validation_report.json",
        mime="application/json",
        use_container_width=True,
    )
else:
    st.info(f"No hourly dataset found at `{HOURLY_CSV}`. Run `python run_pipeline.py` first.")

st.divider()
st.caption(
    f"Raw input: `{Path(RAW_CSV).name}` · report generated {report['generated_utc']} · "
    f"seed {report['reproducibility']['random_seed']} · pandas {report['reproducibility']['pandas']}"
)
