"""
Module purpose:

The figure suite for the notebook and the conference talk.

Every figure follows one visual contract, because a deck whose charts each look like
a different project is harder to follow than one with a single visual language:

* **Colour carries one meaning throughout.** Orange is always the *before* state and
  blue always the *after* state. A third series, when unavoidable, is aqua and is
  always directly labelled. Colours are assigned in a fixed order and never cycled,
  and the three-hue set is separable under the common colour-vision deficiencies.
* **Identity is never colour alone.** Two or more series always carry a legend, and
  values are direct-labelled wherever they fit, so the chart survives a projector
  with poor colour reproduction and a greyscale handout.
* **Ink is recessive.** Grids are faint, axis spines are dropped, and text wears text
  colours rather than series colours.

Each function returns a Matplotlib ``Figure`` and, when ``save_as`` is given, writes a
PNG into ``outputs/figures``.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from config import DECAY_BASE, FIGURES_DIR, PALETTE, SEQUENTIAL_BLUE, TIMESERIES_CHANNELS

__all__ = [
    "apply_house_style",
    "plot_missingness_before_after",
    "plot_duplicate_comparison",
    "plot_temperature_unit_problem",
    "plot_category_consolidation",
    "plot_timeseries_expansion",
    "plot_decay_curve",
    "plot_value_mask_decay_example",
    "plot_patient_trajectory",
    "plot_imputation_benchmark",
    "plot_missingness_heatmap",
]

BEFORE = PALETTE["series_2"]
AFTER = PALETTE["series_1"]
THIRD = PALETTE["series_3"]


def apply_house_style() -> None:
    """Install the project's Matplotlib defaults.

    Call once per notebook session, before any figure is drawn.
    """
    matplotlib.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text_secondary"],
            "axes.titlecolor": PALETTE["text_primary"],
            "axes.titlesize": 12.5,
            "axes.titleweight": "600",
            "axes.titlepad": 14,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": PALETTE["text_secondary"],
            "ytick.color": PALETTE["text_secondary"],
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "text.color": PALETTE["text_primary"],
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "font.family": "sans-serif",
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def _finish(figure: Figure, axes: Any, save_as: str | None, hide_spines: tuple[str, ...] = ("top", "right")) -> Figure:
    """Apply the shared final pass: recessive spines, tight layout, optional save."""
    for axis in np.atleast_1d(axes).ravel():
        for spine in hide_spines:
            axis.spines[spine].set_visible(False)
    figure.tight_layout()
    if save_as:
        destination = Path(FIGURES_DIR) / save_as
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination)
    return figure


def plot_missingness_before_after(
    raw_pct: dict[str, float],
    hourly_pct: dict[str, float],
    final_pct: dict[str, float],
    save_as: str | None = "01_missingness_before_after.png",
) -> Figure:
    """Grouped bars showing missingness at each of the three pipeline stages.

    Parameters
    ----------
    raw_pct:
        Per-channel missing percentage in the raw event table.
    hourly_pct:
        Per-channel missing percentage after hourly alignment, before imputation.
    final_pct:
        Per-channel missing percentage in the final model matrix.
    save_as:
        Filename under ``outputs/figures``, or ``None`` to skip saving.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The middle bar is the point of the figure and it goes the "wrong" way: alignment
    *raises* measured missingness, because the event layout was concealing every hour
    in which nothing was recorded. Reading the first and third bars alone would
    suggest the pipeline solved a 20% problem; it actually solved an 86% one.
    """
    channels = [channel for channel in TIMESERIES_CHANNELS if channel in raw_pct]
    positions = np.arange(len(channels))
    width = 0.27

    figure, axis = plt.subplots(figsize=(11.5, 5.0))
    series = [
        ("Raw event table", [raw_pct.get(channel, 0.0) for channel in channels], BEFORE),
        ("After hourly alignment", [hourly_pct.get(channel, 0.0) for channel in channels], THIRD),
        ("Final model matrix", [final_pct.get(channel, 0.0) for channel in channels], AFTER),
    ]

    for index, (label, values, colour) in enumerate(series):
        offset = (index - 1) * width
        bars = axis.bar(positions + offset, values, width * 0.92, label=label, color=colour, zorder=3)
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.0f}" if value >= 1 else ("0" if value == 0 else f"{value:.1f}"),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
                color=PALETTE["text_secondary"],
            )

    axis.set_xticks(positions)
    axis.set_xticklabels([channel.replace("_", " ") for channel in channels], rotation=20, ha="right")
    axis.set_ylabel("Cells missing")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis.set_ylim(0, 105)
    axis.set_title("Missingness is created by measurement, not by cleaning")
    axis.legend(loc="upper left", ncols=3, bbox_to_anchor=(0, 1.02))
    axis.grid(axis="x", visible=False)
    return _finish(figure, axis, save_as)


def plot_duplicate_comparison(
    duplicates_before: int,
    duplicates_after: int,
    rows_before: int,
    rows_after: int,
    save_as: str | None = "02_duplicates_before_after.png",
) -> Figure:
    """Paired bars for duplicate rows and total row count.

    Parameters
    ----------
    duplicates_before, duplicates_after:
        Redundant row counts either side of deduplication.
    rows_before, rows_after:
        Table heights either side of deduplication.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for axis, values, title, ylabel in (
        (left, (duplicates_before, duplicates_after), "Duplicate rows", "Rows"),
        (right, (rows_before, rows_after), "Total rows", "Rows"),
    ):
        bars = axis.bar(["Before", "After"], values, width=0.5, color=[BEFORE, AFTER], zorder=3)
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=11,
                fontweight="600",
                color=PALETTE["text_primary"],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(values) * 1.18 or 1)
        axis.grid(axis="x", visible=False)

    figure.suptitle("Exact-match deduplication", y=1.0, fontsize=12.5, fontweight="600")
    return _finish(figure, (left, right), save_as)


def plot_temperature_unit_problem(
    raw_values: pd.Series,
    converted_values: pd.Series,
    save_as: str | None = "03_temperature_units.png",
) -> Figure:
    """Histograms of temperature before and after unit normalisation.

    Parameters
    ----------
    raw_values:
        Numeric magnitudes stripped from the raw column, units ignored.
    converted_values:
        The same measurements after conversion to Celsius.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The left panel is bimodal and the two modes are one patient population measured on
    two scales. Any model trained on it would learn a "cohort" that is an artefact of
    thermometer firmware.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)

    raw_clean = pd.to_numeric(raw_values, errors="coerce").dropna()
    raw_clean = raw_clean[raw_clean.between(20, 120)]
    left.hist(raw_clean, bins=44, color=BEFORE, zorder=3)
    left.set_title("Before  ·  Celsius and Fahrenheit on one axis")
    left.set_xlabel("Recorded magnitude (unit ignored)")
    left.set_ylabel("Observations")
    left.annotate(
        "Celsius mode",
        xy=(37, left.get_ylim()[1] * 0.85),
        ha="center",
        fontsize=9,
        color=PALETTE["text_secondary"],
    )
    left.annotate(
        "Fahrenheit mode",
        xy=(99, left.get_ylim()[1] * 0.55),
        ha="center",
        fontsize=9,
        color=PALETTE["text_secondary"],
    )

    converted_clean = pd.to_numeric(converted_values, errors="coerce").dropna()
    converted_clean = converted_clean[converted_clean.between(30, 45)]
    right.hist(converted_clean, bins=44, color=AFTER, zorder=3)
    right.set_title("After  ·  all values in degrees Celsius")
    right.set_xlabel("Temperature (°C)")
    right.set_ylabel("Observations")

    for axis in (left, right):
        axis.grid(axis="x", visible=False)

    return _finish(figure, (left, right), save_as)


def plot_category_consolidation(
    categorical_audit: dict[str, Any],
    save_as: str | None = "04_category_consolidation.png",
) -> Figure:
    """Horizontal paired bars: distinct category values before and after mapping.

    Parameters
    ----------
    categorical_audit:
        Audit from :func:`preprocessing.categorical_cleaner.standardise_categories`.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    columns = list(categorical_audit)
    before = [categorical_audit[column]["distinct_before"] for column in columns]
    after = [categorical_audit[column]["distinct_after"] for column in columns]
    positions = np.arange(len(columns))
    height = 0.36

    figure, axis = plt.subplots(figsize=(9.5, 0.72 * len(columns) + 2.2))
    axis.barh(positions + height / 2, before, height * 0.92, label="Distinct raw values", color=BEFORE, zorder=3)
    axis.barh(positions - height / 2, after, height * 0.92, label="After controlled vocabulary", color=AFTER, zorder=3)

    for position, (before_value, after_value) in enumerate(zip(before, after)):
        axis.annotate(str(before_value), (before_value, position + height / 2), xytext=(5, 0),
                      textcoords="offset points", va="center", fontsize=9, color=PALETTE["text_secondary"])
        axis.annotate(str(after_value), (after_value, position - height / 2), xytext=(5, 0),
                      textcoords="offset points", va="center", fontsize=9, color=PALETTE["text_secondary"])

    axis.set_yticks(positions)
    axis.set_yticklabels([column.replace("_", " ") for column in columns])
    axis.set_xlabel("Distinct values")
    axis.set_xlim(0, max(before) * 1.18)
    axis.set_title("Free-text variants collapsed onto a controlled vocabulary")
    axis.legend(loc="lower right")
    axis.grid(axis="y", visible=False)
    return _finish(figure, axis, save_as)


def plot_timeseries_expansion(
    rows_event: int,
    rows_hourly: int,
    hours_per_admission: dict[str, Any],
    save_as: str | None = "05_timeseries_expansion.png",
) -> Figure:
    """Row-count expansion from the event table to the hourly grid.

    Parameters
    ----------
    rows_event, rows_hourly:
        Row counts either side of reconstruction.
    hours_per_admission:
        ``{"min", "median", "max"}`` summary from the builder audit.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    values = (rows_event, rows_hourly)
    bars = axis.bar(
        ["Event rows\n(irregular observations)", "Hourly rows\n(regular grid)"],
        values,
        width=0.5,
        color=[BEFORE, AFTER],
        zorder=3,
    )
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=12,
            fontweight="600",
            color=PALETTE["text_primary"],
        )
    axis.set_ylabel("Rows")
    axis.set_ylim(0, max(values) * 1.2)
    axis.set_title(
        f"Temporal alignment expands the table {rows_hourly / max(rows_event, 1):.1f}x\n"
        f"median stay {hours_per_admission.get('median')} hours "
        f"(range {hours_per_admission.get('min')}-{hours_per_admission.get('max')})"
    )
    axis.grid(axis="x", visible=False)
    return _finish(figure, axis, save_as)


def plot_decay_curve(
    decay_base: float = DECAY_BASE,
    max_gap: int = 12,
    save_as: str | None = "06_decay_curve.png",
) -> Figure:
    """The confidence decay $d_j = \\gamma^{\\,j}$ against gap length.

    Parameters
    ----------
    decay_base:
        $\\gamma$.
    max_gap:
        Largest gap plotted, in hours.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    gaps = np.arange(0, max_gap + 1)
    decay = decay_base ** gaps

    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.plot(gaps, decay, color=AFTER, marker="o", zorder=3)
    for gap in (0, 1, 2, 4, 8):
        if gap <= max_gap:
            axis.annotate(
                f"{decay_base ** gap:.2f}",
                (gap, decay_base ** gap),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color=PALETTE["text_secondary"],
            )
    axis.set_xlabel("Hours since the last genuine observation  ($j$)")
    axis.set_ylabel("Confidence weight  ($d_j$)")
    axis.set_ylim(0, 1.12)
    axis.set_xlim(-0.4, max_gap + 0.4)
    axis.set_title(f"Confidence in a carried-forward value:  $d_j = {decay_base}^{{\\,j}}$")
    axis.grid(axis="x", visible=False)
    return _finish(figure, axis, save_as)


def plot_value_mask_decay_example(
    frame: pd.DataFrame,
    channel: str = "heart_rate",
    admission_id: str | None = None,
    hours: int = 36,
    save_as: str | None = "07_value_mask_decay_example.png",
) -> Figure:
    """Three stacked panels showing Value, Mask and Decay for one admission.

    Parameters
    ----------
    frame:
        Processed table carrying the Value-Mask-Decay columns.
    channel:
        Channel to display.
    admission_id:
        Admission to display. Defaults to the one with the most observations of this
        channel, which makes the clearest illustration.
    hours:
        Number of leading hours to show.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mask_column = f"{channel}_mask"
    if admission_id is None:
        admission_id = frame.groupby("admission_id")[mask_column].sum().idxmax()

    subset = frame[frame["admission_id"] == admission_id].head(hours).reset_index(drop=True)
    time_index = subset["hours_since_admission"].to_numpy()
    values = subset[f"{channel}_value"].to_numpy()
    mask = subset[mask_column].to_numpy()
    decay = subset[f"{channel}_decay"].to_numpy()

    figure, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(11.0, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 0.8, 1.3]}
    )

    top.step(time_index, values, where="post", color=PALETTE["text_secondary"], linewidth=1.4, zorder=2, label="Carried-forward value")
    top.scatter(time_index[mask == 1], values[mask == 1], s=52, color=AFTER, zorder=4,
                edgecolor=PALETTE["surface"], linewidth=1.6, label="Genuinely observed")
    top.scatter(time_index[mask == 0], values[mask == 0], s=20, color=BEFORE, zorder=3, label="Imputed")
    top.set_ylabel(channel.replace("_", " "))
    top.set_title(f"Value, Mask and Decay  ·  {channel.replace('_', ' ')}  ·  admission {admission_id}")
    top.legend(loc="upper right", ncols=3)

    middle.bar(time_index, mask, width=0.82, color=np.where(mask == 1, AFTER, BEFORE), zorder=3)
    middle.set_ylabel("Mask")
    middle.set_yticks([0, 1])
    middle.set_ylim(0, 1.35)

    bottom.bar(time_index, decay, width=0.82, color=THIRD, zorder=3)
    bottom.set_ylabel("Decay")
    bottom.set_ylim(0, 1.15)
    bottom.set_xlabel("Hours since admission")

    for axis in (top, middle, bottom):
        axis.grid(axis="x", visible=False)

    return _finish(figure, (top, middle, bottom), save_as)


def plot_patient_trajectory(
    frame: pd.DataFrame,
    channel: str = "heart_rate",
    admission_id: str | None = None,
    hours: int = 48,
    save_as: str | None = "08_refinement_trajectory.png",
) -> Figure:
    """Compare the carried-forward and refined trajectories for one admission.

    Parameters
    ----------
    frame:
        Processed table carrying ``<channel>_value`` and ``<channel>_refined``.
    channel:
        Channel to display.
    admission_id:
        Admission to display.
    hours:
        Leading hours to show.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The two lines coincide exactly at every observed point - which is the guarantee
    being illustrated - and separate only across gaps, by an amount governed by the
    decay weight.
    """
    mask_column = f"{channel}_mask"
    if admission_id is None:
        admission_id = frame.groupby("admission_id")[mask_column].sum().idxmax()

    subset = frame[frame["admission_id"] == admission_id].head(hours).reset_index(drop=True)
    time_index = subset["hours_since_admission"].to_numpy()
    mask = subset[mask_column].to_numpy()

    figure, axis = plt.subplots(figsize=(11.0, 4.6))
    axis.step(time_index, subset[f"{channel}_value"], where="post", color=BEFORE, linewidth=2.0,
              label="LOCF (carried forward)", zorder=3)
    axis.plot(time_index, subset[f"{channel}_refined"], color=AFTER, linewidth=2.0,
              label="DeepTSE-refined", zorder=4)
    axis.scatter(time_index[mask == 1], subset[f"{channel}_value"].to_numpy()[mask == 1], s=58,
                 color=PALETTE["text_primary"], zorder=5, edgecolor=PALETTE["surface"], linewidth=1.8,
                 label="Genuine observation (never modified)")

    axis.set_xlabel("Hours since admission")
    axis.set_ylabel(channel.replace("_", " "))
    axis.set_title(f"Refinement moves imputed cells only  ·  admission {admission_id}")
    axis.legend(loc="best")
    axis.grid(axis="x", visible=False)
    return _finish(figure, axis, save_as)


def plot_imputation_benchmark(
    benchmark: dict[str, Any],
    save_as: str | None = "09_imputation_benchmark.png",
) -> Figure:
    """Per-channel mean absolute error, LOCF against refinement, on held-out data.

    Parameters
    ----------
    benchmark:
        Output of :func:`models.deeptse_imputer.benchmark_against_locf`.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The quantity plotted is the *percentage reduction in mean absolute error*, not the
    errors themselves. Plotting raw MAE would mean putting temperature (around 0.2 °C)
    and glucose (around 11 mg/dL) on one axis, which forces a logarithmic scale - and
    bar length on a log axis no longer encodes magnitude, so the chart would mislead
    precisely where it looks most authoritative. The raw errors belong in the table
    beside this figure; the chart carries the comparable, unit-free quantity.

    Channels where refinement *loses* are drawn in the critical colour with a negative
    label rather than omitted. Hiding them would make the method look better and the
    talk worse.
    """
    channels = list(benchmark["by_channel"])
    improvement = [benchmark["by_channel"][channel]["mae_improvement_pct"] for channel in channels]
    order = np.argsort(improvement)
    channels = [channels[index] for index in order]
    improvement = [improvement[index] for index in order]
    positions = np.arange(len(channels))

    figure, axis = plt.subplots(figsize=(10.5, 0.55 * len(channels) + 2.6))
    colours = [PALETTE["good"] if value > 0 else PALETTE["critical"] for value in improvement]
    axis.barh(positions, improvement, height=0.62, color=colours, zorder=3)
    axis.axvline(0, color=PALETTE["text_secondary"], linewidth=1.2, zorder=4)

    span = max(abs(min(improvement)), abs(max(improvement))) or 1.0
    for position, value in zip(positions, improvement):
        offset = 6 if value >= 0 else -6
        axis.annotate(
            f"{value:+.1f}%",
            (value, position),
            textcoords="offset points",
            xytext=(offset, 0),
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9.5,
            fontweight="600",
            color=PALETTE["good"] if value > 0 else PALETTE["critical"],
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([channel.replace("_", " ") for channel in channels])
    axis.set_xlabel("Reduction in mean absolute error versus LOCF")
    axis.set_xlim(-span * 1.45, span * 1.45)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))

    mode = benchmark.get("holdout_mode", "random")
    overall = benchmark.get("overall", {})
    axis.set_title(
        f"Does refinement beat carrying the last value forward?\n"
        f"{mode} holdout · {overall.get('held_out_cells', 0):,} hidden observations · "
        f"{overall.get('channels_improved', 0)} of {overall.get('channels_tested', 0)} channels improved · "
        f"pooled {overall.get('improvement_pct', 0):+.1f}%"
    )
    axis.grid(axis="y", visible=False)
    return _finish(figure, axis, save_as)


def plot_missingness_heatmap(
    frame: pd.DataFrame,
    channel: str = "heart_rate",
    n_admissions: int = 30,
    hours: int = 48,
    save_as: str | None = "10_missingness_heatmap.png",
) -> Figure:
    """Observation pattern across admissions and hours for one channel.

    Parameters
    ----------
    frame:
        Processed table carrying ``<channel>_mask``.
    channel:
        Channel to display.
    n_admissions:
        Number of admissions shown on the vertical axis.
    hours:
        Number of hours shown on the horizontal axis.
    save_as:
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Rendered on a single-hue sequential ramp: this encodes one ordered quantity
    (measured or not), so two hues would imply a polarity the data does not have.
    """
    mask_column = f"{channel}_mask"
    admissions = sorted(frame["admission_id"].unique())[:n_admissions]
    grid = np.zeros((len(admissions), hours))

    for row, admission in enumerate(admissions):
        subset = frame.loc[frame["admission_id"] == admission, mask_column].to_numpy()[:hours]
        grid[row, : len(subset)] = subset

    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    colours = matplotlib.colors.LinearSegmentedColormap.from_list(
        "observed", [PALETTE["grid"], SEQUENTIAL_BLUE[4]]
    )
    axis.imshow(grid, aspect="auto", cmap=colours, interpolation="nearest", vmin=0, vmax=1)

    axis.set_xlabel("Hours since admission")
    axis.set_ylabel("Admission")
    axis.set_yticks(np.arange(0, len(admissions), max(1, len(admissions) // 10)))
    axis.set_yticklabels(
        [admissions[index] for index in range(0, len(admissions), max(1, len(admissions) // 10))], fontsize=8
    )
    observed_pct = 100.0 * grid.mean()
    axis.set_title(
        f"When was {channel.replace('_', ' ')} actually measured?  ·  "
        f"{observed_pct:.1f}% of patient-hours carry a real measurement\n"
        "dark = observed, light = never measured"
    )
    axis.grid(visible=False)
    return _finish(figure, axis, save_as, hide_spines=("top", "right", "left", "bottom"))
