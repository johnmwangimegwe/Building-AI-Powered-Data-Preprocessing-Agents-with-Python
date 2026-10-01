"""
Module purpose:

Build the Value-Mask-Decay representation that makes clinical missingness explicit
and learnable.

The problem with plain imputation
---------------------------------
Filling a gap with the last observed value produces a table a model can consume, and
destroys the information that the cell was ever empty. In intensive care that
information is clinically loaded: a patient measured every fifteen minutes is being
watched closely, and one measured twice a day is stable. Missingness is not noise to
be removed - it is a signal about the patient.

The representation
------------------
Each channel $x$ is therefore expanded into three aligned series.

**Value.** The carried-forward series

$$ v_t = \\begin{cases} x_t, & x_t \\text{ observed} \\\\ v_{t-1}, & \\text{otherwise} \\end{cases} $$

**Mask.** The provenance indicator

$$ m_t = \\begin{cases} 1, & x_t \\text{ genuinely observed} \\\\ 0, & x_t \\text{ imputed} \\end{cases} $$

**Decay.** The confidence weight

$$ d_t = \\gamma^{\\,j_t}, \\qquad \\gamma = 0.75 $$

where $j_t$ is the number of hours since the most recent genuine observation, so
$j_t = 0$ whenever $m_t = 1$. A value carried forward one hour keeps weight $0.75$;
after six hours it retains $0.75^{6} \\approx 0.18$. The model is thereby told not
only *what* the value is but *how much to believe it*, which is the same intuition
that GRU-D formalises for clinical time series.

**Leading gaps.** Before a channel's first observation there is nothing to carry
forward, so LOCF is undefined. Those cells receive the population mean $\\mu$ of the
channel, computed over genuinely observed values only. After standardisation

$$ z = \\frac{x - \\mu}{\\sigma} $$

a neutral-filled cell becomes exactly $0$ - the "no information" point of the
standardised scale - while its mask stays $0$ and its decay stays at the floor, so
the fill is never mistaken for evidence.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import DECAY_BASE, TIMESERIES_CHANNELS

__all__ = [
    "build_value_mask_decay",
    "neutral_fill_leading_gaps",
    "locf_with_mask_and_decay",
    "standardise_channels",
]


def neutral_fill_leading_gaps(series: pd.Series, population_mean: float) -> tuple[pd.Series, int]:
    """Fill the gap preceding a channel's first observation with the population mean.

    Parameters
    ----------
    series:
        One admission's hourly values for one channel, time-ordered.
    population_mean:
        Mean of the channel over genuinely observed values across the whole cohort.

    Returns
    -------
    (pandas.Series, int)
        The series with only its leading gap filled, and the number of cells filled.

    Notes
    -----
    Only the *leading* run is touched. Interior gaps belong to LOCF, and trailing gaps
    after the last observation are also carried forward, because the most recent
    measurement remains the best available estimate of a discharged patient's state.

    Example
    -------
    >>> filled, n = neutral_fill_leading_gaps(pd.Series([np.nan, np.nan, 80.0]), 75.0)
    >>> list(filled)
    [75.0, 75.0, 80.0]
    >>> n
    2
    """
    values = series.copy()
    observed_positions = np.flatnonzero(values.notna().to_numpy())
    if observed_positions.size == 0:
        n_filled = int(values.isna().sum())
        return pd.Series(population_mean, index=series.index, dtype="float64"), n_filled
    first_observed = int(observed_positions[0])
    if first_observed == 0:
        return values, 0
    values.iloc[:first_observed] = population_mean
    return values, first_observed


def locf_with_mask_and_decay(
    series: pd.Series,
    population_mean: float,
    decay_base: float = DECAY_BASE,
) -> dict[str, np.ndarray]:
    """Compute the Value, Mask and Decay triple for one channel of one admission.

    Parameters
    ----------
    series:
        Time-ordered hourly values for one channel of one admission, with ``NaN``
        marking hours in which nothing was measured.
    population_mean:
        Cohort mean of genuinely observed values, used for the leading gap.
    decay_base:
        $\\gamma$ in $d_t = \\gamma^{j_t}$. Defaults to ``config.DECAY_BASE`` (0.75).

    Returns
    -------
    dict[str, numpy.ndarray]
        ``value`` (float), ``mask`` (int 0/1), ``decay`` (float), ``gap_hours``
        (int $j_t$) and ``fill_kind`` (object: ``observed`` / ``locf`` /
        ``neutral_fill``), all of the same length as ``series``.

    Example
    -------
    >>> out = locf_with_mask_and_decay(pd.Series([80.0, np.nan, np.nan, 85.0]), 75.0)
    >>> list(out["value"])
    [80.0, 80.0, 80.0, 85.0]
    >>> list(out["mask"])
    [1, 0, 0, 1]
    >>> [round(float(d), 4) for d in out["decay"]]
    [1.0, 0.75, 0.5625, 1.0]
    """
    observed = series.notna().to_numpy()
    mask = observed.astype(int)

    filled, _ = neutral_fill_leading_gaps(series, population_mean)
    value = filled.ffill().to_numpy(dtype="float64")

    n = len(series)
    gap_hours = np.zeros(n, dtype=int)
    fill_kind = np.empty(n, dtype=object)

    first_observed = int(np.flatnonzero(observed)[0]) if observed.any() else n
    since_last = 0
    for position in range(n):
        if observed[position]:
            since_last = 0
            fill_kind[position] = "observed"
        else:
            since_last += 1
            fill_kind[position] = "neutral_fill" if position < first_observed else "locf"
        gap_hours[position] = since_last

    # A neutral-filled leading cell has no last observation at all. Pinning its gap to
    # the length of the leading run keeps the decay monotone and drives it towards the
    # floor, which is the correct amount of confidence to place in a population mean.
    if first_observed > 0:
        gap_hours[:first_observed] = np.arange(first_observed, 0, -1)

    decay = np.power(float(decay_base), gap_hours.astype(float))
    decay[observed] = 1.0

    # Backward context: the *next* genuine observation, and how far ahead it lies.
    # This is only defined when the whole trajectory is in hand, so it belongs to
    # retrospective dataset construction and must never be used for real-time
    # inference. See ``models.deeptse_imputer`` for where that distinction is enforced.
    next_value = filled.bfill().to_numpy(dtype="float64")
    if np.isnan(next_value).any():
        next_value = np.where(np.isnan(next_value), value, next_value)

    forward_gap = gap_hours.astype(float)
    backward_gap = np.zeros(n, dtype=float)
    ahead = 0.0
    for position in range(n - 1, -1, -1):
        if observed[position]:
            ahead = 0.0
        else:
            ahead += 1.0
        backward_gap[position] = ahead
    backward_decay = np.power(float(decay_base), backward_gap)
    backward_decay[observed] = 1.0

    return {
        "value": value,
        "mask": mask,
        "decay": decay,
        "gap_hours": gap_hours,
        "fill_kind": fill_kind,
        "next_value": next_value,
        "backward_gap_hours": backward_gap.astype(int),
        "backward_decay": backward_decay,
        "forward_gap_hours": forward_gap.astype(int),
    }


def standardise_channels(
    frame: pd.DataFrame,
    channels: list[str],
    statistics: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Z-standardise the ``*_value`` columns using observed-only statistics.

    Parameters
    ----------
    frame:
        Hourly table already carrying ``<channel>_value`` and ``<channel>_mask``.
    channels:
        Channels to standardise.
    statistics:
        Pre-computed ``{channel: {"mean": float, "std": float}}``. When ``None``, the
        statistics are computed from cells with ``mask == 1`` only.

    Returns
    -------
    (pandas.DataFrame, dict)
        Table with added ``<channel>_z`` columns, and the statistics used - which must
        be persisted and reused at inference time to avoid train/serve skew.

    Notes
    -----
    The statistics deliberately exclude imputed cells. Including them would let the
    fill value shift the mean towards itself, and the neutral-fill guarantee
    ("a filled cell standardises to zero") would quietly stop holding.
    """
    working = frame.copy()
    computed: dict[str, dict[str, float]] = {}

    for channel in channels:
        value_column, mask_column = f"{channel}_value", f"{channel}_mask"
        if value_column not in working.columns:
            continue
        if statistics and channel in statistics:
            mean, std = statistics[channel]["mean"], statistics[channel]["std"]
        else:
            observed = working.loc[working[mask_column] == 1, value_column]
            mean = float(observed.mean())
            std = float(observed.std(ddof=0)) or 1.0
        computed[channel] = {"mean": round(mean, 4), "std": round(std, 4)}
        working[f"{channel}_z"] = ((working[value_column] - mean) / std).round(4)
        next_column = f"{channel}_next_value"
        if next_column in working.columns:
            working[f"{channel}_next_z"] = ((working[next_column] - mean) / std).round(4)

    return working, computed


def build_value_mask_decay(
    hourly: pd.DataFrame,
    channels: list[str] | None = None,
    decay_base: float = DECAY_BASE,
    group_column: str = "admission_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply neutral fill, LOCF, masking and decay to every channel of every admission.

    Parameters
    ----------
    hourly:
        Output of :func:`preprocessing.timeseries_builder.build_hourly_timeseries`.
    channels:
        Channels to expand. Defaults to ``config.TIMESERIES_CHANNELS``.
    decay_base:
        $\\gamma$ in the decay formula.
    group_column:
        Column identifying one patient trajectory. Carrying a value across admissions
        would be a serious clinical error, so grouping is never optional.

    Returns
    -------
    (pandas.DataFrame, dict)
        Table gaining ``<channel>_value``, ``<channel>_mask``, ``<channel>_decay``,
        ``<channel>_gap_hours``, ``<channel>_fill_kind`` and ``<channel>_z`` columns,
        plus an audit with per-channel fill counts, the standardisation statistics and
        a verification that neutral-filled cells standardise to zero.

    Example
    -------
    >>> vmd, audit = build_value_mask_decay(hourly)
    >>> audit["channels"]["heart_rate"]["observed"]
    1043
    >>> audit["neutral_fill_zero_check"]["heart_rate"]
    0.0
    """
    channels = channels or TIMESERIES_CHANNELS
    working = hourly.sort_values([group_column, "hour"]).reset_index(drop=True).copy()

    population_means = {
        channel: float(pd.to_numeric(working[channel], errors="coerce").mean())
        for channel in channels
        if channel in working.columns
    }

    per_channel: dict[str, dict[str, Any]] = {}

    for channel in channels:
        if channel not in working.columns:
            continue
        mean = population_means[channel]
        values = np.empty(len(working), dtype="float64")
        masks = np.empty(len(working), dtype=int)
        decays = np.empty(len(working), dtype="float64")
        gaps = np.empty(len(working), dtype=int)
        kinds = np.empty(len(working), dtype=object)
        next_values = np.empty(len(working), dtype="float64")
        backward_gaps = np.empty(len(working), dtype=int)
        backward_decays = np.empty(len(working), dtype="float64")

        for _, positions in working.groupby(group_column, sort=False).indices.items():
            positions = np.sort(positions)
            triple = locf_with_mask_and_decay(
                working[channel].iloc[positions], population_mean=mean, decay_base=decay_base
            )
            values[positions] = triple["value"]
            masks[positions] = triple["mask"]
            decays[positions] = triple["decay"]
            gaps[positions] = triple["gap_hours"]
            kinds[positions] = triple["fill_kind"]
            next_values[positions] = triple["next_value"]
            backward_gaps[positions] = triple["backward_gap_hours"]
            backward_decays[positions] = triple["backward_decay"]

        working[f"{channel}_value"] = np.round(values, 3)
        working[f"{channel}_mask"] = masks
        working[f"{channel}_decay"] = np.round(decays, 6)
        working[f"{channel}_gap_hours"] = gaps
        working[f"{channel}_fill_kind"] = kinds
        working[f"{channel}_next_value"] = np.round(next_values, 3)
        working[f"{channel}_backward_gap_hours"] = backward_gaps
        working[f"{channel}_backward_decay"] = np.round(backward_decays, 6)

        kind_counts = pd.Series(kinds).value_counts()
        per_channel[channel] = {
            "population_mean_used": round(mean, 3),
            "observed": int(kind_counts.get("observed", 0)),
            "locf_filled": int(kind_counts.get("locf", 0)),
            "neutral_filled": int(kind_counts.get("neutral_fill", 0)),
            "observed_pct": round(100.0 * masks.mean(), 2),
            "max_gap_hours": int(gaps.max()),
            "mean_decay_on_imputed": round(float(decays[masks == 0].mean()), 4) if (masks == 0).any() else None,
        }

    working, statistics = standardise_channels(working, channels)

    # Verify the neutral-fill guarantee empirically rather than asserting it in prose.
    zero_check: dict[str, float] = {}
    for channel in channels:
        kind_column, z_column = f"{channel}_fill_kind", f"{channel}_z"
        if kind_column not in working.columns or z_column not in working.columns:
            continue
        neutral = working.loc[working[kind_column] == "neutral_fill", z_column]
        zero_check[channel] = round(float(neutral.abs().max()), 4) if len(neutral) else 0.0

    audit: dict[str, Any] = {
        "decay_base": decay_base,
        "rows": int(len(working)),
        "channels": per_channel,
        "standardisation": statistics,
        "neutral_fill_zero_check": zero_check,
        "neutral_fill_zero_check_note": (
            "Largest absolute z-score among neutral-filled cells. A value of 0.0 confirms "
            "that a population-mean fill standardises exactly to the no-information point."
        ),
    }
    return working, audit
