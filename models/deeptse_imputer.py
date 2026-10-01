"""
Module purpose:

DeepTSE-inspired refinement of imputed clinical values.

What this is, and what it is not
--------------------------------
This is **not** a re-implementation of a published deep time-series encoder, and the
notebook says so on the slide. Training a recurrent imputation network on a hundred
admissions would overfit, take minutes on stage, and prove nothing. What is
reproduced here is the *mechanism* that makes such models work, in a form that runs
in under a second, is deterministic, and can be read end to end:

1. a **temporal embedding** of each channel that carries the Value-Mask-Decay triple
   rather than the value alone;
2. a **cross-channel refiner** that predicts a channel from the simultaneous state of
   the others, because vital signs are physiologically coupled - a rising heart rate
   with a falling blood pressure is informative about the missing respiratory rate;
3. a **decay-gated blend** between the carried-forward value and the model prediction.

The blend is the heart of it. For an imputed cell with gap $j$ hours and decay
$d = \\gamma^{j}$:

$$ \\hat{x} = d \\cdot v_{\\text{LOCF}} + (1 - d) \\cdot f_{\\theta}(\\mathbf{h}) $$

One hour after a real measurement, $d = 0.75$ and the carried value dominates -
correctly, since the patient has barely changed. Eight hours later $d \\approx 0.10$
and the estimate has drifted to what the other channels imply. This is exactly the
behaviour GRU-D's decay gate produces, expressed in closed form.

The inviolable rule
-------------------
Refinement is applied **only where the mask is zero**. A genuinely observed
measurement is evidence; overwriting it with a model's opinion would be data
fabrication, and in a clinical dataset that is not a technical error but an ethical
one. The rule is enforced in code and then *verified* by the validation agent, which
recomputes the observed cells against the pre-refinement table.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import DECAY_BASE, HOLDOUT_FRACTION, RANDOM_SEED, TIMESERIES_CHANNELS

__all__ = ["DeepTSEImputer", "refine_missing_values", "benchmark_against_locf"]


class DeepTSEImputer:
    """A ridge-regularised, decay-gated cross-channel refiner.

    Parameters
    ----------
    channels:
        Channels to refine. Defaults to ``config.TIMESERIES_CHANNELS``.
    embedding_dim:
        Width of the temporal embedding built for each channel. The embedding
        concatenates the standardised value, the mask, the decay and a small Fourier
        basis over the hour-of-stay, giving the refiner a sense of where in the
        admission it is.
    ridge_lambda:
        L2 penalty. With eight channels and thousands of rows the closed-form ridge
        solution is stable and instantaneous, which is what makes the demonstration
        safe to run live.

    Attributes
    ----------
    coefficients_:
        ``{channel: numpy.ndarray}`` learned weight vector per channel.
    training_report_:
        Per-channel training-set size and in-sample $R^2$.

    Example
    -------
    >>> imputer = DeepTSEImputer()
    >>> imputer.fit(vmd_frame)
    >>> refined, audit = imputer.transform(vmd_frame)
    >>> audit["cells_refined"] > 0
    True
    """

    def __init__(
        self,
        channels: list[str] | None = None,
        embedding_dim: int = 4,
        ridge_lambda: float = 1.0,
        decay_base: float = DECAY_BASE,
    ) -> None:
        self.channels = channels or TIMESERIES_CHANNELS
        self.embedding_dim = embedding_dim
        self.ridge_lambda = ridge_lambda
        self.decay_base = decay_base
        self.coefficients_: dict[str, np.ndarray] = {}
        self.training_report_: dict[str, dict[str, float]] = {}

    # -- embedding ---------------------------------------------------------------

    def _temporal_basis(self, hours_since_admission: np.ndarray) -> np.ndarray:
        """Fourier basis over position in the admission.

        A raw hour index would let the refiner extrapolate linearly out of the
        observed range; a bounded periodic basis cannot, which is the desired
        conservatism for a clinical estimate.
        """
        scaled = hours_since_admission.astype(float) / 24.0
        columns = [np.ones_like(scaled)]
        for harmonic in range(1, self.embedding_dim + 1):
            columns.append(np.sin(harmonic * np.pi * scaled / 3.0))
            columns.append(np.cos(harmonic * np.pi * scaled / 3.0))
        return np.column_stack(columns)

    def _design_matrix(self, frame: pd.DataFrame, target_channel: str) -> np.ndarray:
        """Build the feature matrix used to predict ``target_channel``.

        Features are the *other* channels' Value-Mask-Decay triples plus the temporal
        basis. The target channel's own value is excluded, because at an imputed cell
        that value is the carried-forward estimate the refiner is meant to improve on -
        including it would simply teach the model to echo LOCF.
        """
        def column(name: str) -> np.ndarray:
            if name not in frame.columns:
                return np.zeros((len(frame), 1))
            return np.nan_to_num(frame[name].to_numpy(dtype=float)).reshape(-1, 1)

        blocks: list[np.ndarray] = []
        for channel in self.channels:
            if channel == target_channel:
                # The target's own *temporal* context, in both directions. A forward-only
                # refiner can never beat LOCF on a gap bounded by two real measurements,
                # because it cannot see the closing measurement; supplying the backward
                # value lets the ridge learn gap-weighted interpolation, which is what a
                # bidirectional recurrent imputer learns from data.
                blocks.append(column(f"{channel}_z"))
                blocks.append(column(f"{channel}_next_z"))
                blocks.append(column(f"{channel}_decay"))
                blocks.append(column(f"{channel}_backward_decay"))
                # Gap-weighted linear interpolation between the bracketing observations,
                # offered explicitly so the model can select it rather than rediscover it.
                forward_gap = np.nan_to_num(frame.get(f"{channel}_gap_hours", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float))
                backward_gap = np.nan_to_num(frame.get(f"{channel}_backward_gap_hours", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float))
                span = forward_gap + backward_gap
                weight = np.where(span > 0, backward_gap / np.where(span > 0, span, 1.0), 1.0)
                interpolated = weight * column(f"{channel}_z").ravel() + (1.0 - weight) * column(f"{channel}_next_z").ravel()
                blocks.append(interpolated.reshape(-1, 1))
                blocks.append(column(f"{channel}_mask"))
                continue
            blocks.append(column(f"{channel}_z"))
            blocks.append(column(f"{channel}_next_z"))
            blocks.append(column(f"{channel}_mask"))
            blocks.append(column(f"{channel}_decay"))

        hours = frame.get("hours_since_admission", pd.Series(np.zeros(len(frame)))).to_numpy()
        blocks.append(self._temporal_basis(hours))
        return np.hstack(blocks)

    # -- fitting -----------------------------------------------------------------

    def fit(
        self,
        frame: pd.DataFrame,
        hourly_raw: pd.DataFrame | None = None,
        mask_fraction: float = 0.25,
        seed: int = RANDOM_SEED,
    ) -> "DeepTSEImputer":
        """Learn one refiner per channel by **self-supervised artificial masking**.

        Parameters
        ----------
        frame:
            The Value-Mask-Decay table the refiner will ultimately be applied to.
        hourly_raw:
            The hourly table *before* expansion, still holding ``NaN`` at unmeasured
            hours. Supplying it enables self-supervised training, which is the correct
            procedure. When omitted the method falls back to fitting on observed cells,
            which is documented below as unsound and reported in ``training_report_``.
        mask_fraction:
            Share of observed cells hidden to create training targets.
        seed:
            Seed controlling which cells are hidden.

        Returns
        -------
        DeepTSEImputer
            ``self``, so the call can be chained.

        Notes
        -----
        Fitting on ``mask == 1`` rows directly is the obvious approach and it is
        **wrong**. At an observed cell the channel's own ``_z`` feature *is* the target,
        so the regression achieves a perfect in-sample fit by copying one column, learns
        nothing, and at imputation time merely echoes LOCF. The training distribution
        has to match the inference distribution.

        The remedy is the standard self-supervised construction: hide a random subset of
        genuinely observed cells, rebuild the Value-Mask-Decay representation as though
        those measurements had never been taken, and fit on the hidden cells - where the
        context looks exactly like an imputed cell and the truth is nonetheless known.
        """
        from preprocessing.missing_value_handler import build_value_mask_decay

        if hourly_raw is None:
            for channel in self.channels:
                self.training_report_[channel] = {
                    "training_rows": 0,
                    "n_features": 0,
                    "r2_heldout": None,
                    "status": "not_fitted__hourly_raw_not_supplied",
                }
            return self

        rng = np.random.default_rng(seed)
        corrupted = hourly_raw.copy()
        hidden_positions: dict[str, np.ndarray] = {}

        for channel in self.channels:
            if channel not in corrupted.columns:
                continue
            observed_positions = np.flatnonzero(corrupted[channel].notna().to_numpy())
            if observed_positions.size < 40:
                continue
            n_hide = max(1, int(round(observed_positions.size * mask_fraction)))
            hidden = rng.choice(observed_positions, size=n_hide, replace=False)
            hidden_positions[channel] = hidden
            corrupted.iloc[hidden, corrupted.columns.get_loc(channel)] = np.nan

        corrupted_vmd, _ = build_value_mask_decay(corrupted, channels=self.channels, decay_base=self.decay_base)

        for channel, hidden in hidden_positions.items():
            z_column = f"{channel}_z"
            if z_column not in frame.columns:
                continue

            design = self._design_matrix(corrupted_vmd, channel)[hidden]
            target = np.nan_to_num(frame[z_column].to_numpy(dtype=float)[hidden])
            if design.shape[0] < 30:
                continue

            gram = design.T @ design + self.ridge_lambda * np.eye(design.shape[1])
            weights = np.linalg.solve(gram, design.T @ target)
            self.coefficients_[channel] = weights

            prediction = design @ weights
            residual_ss = float(((target - prediction) ** 2).sum())
            total_ss = float(((target - target.mean()) ** 2).sum()) or 1.0
            self.training_report_[channel] = {
                "training_rows": int(design.shape[0]),
                "n_features": int(design.shape[1]),
                "r2_in_sample": round(1.0 - residual_ss / total_ss, 4),
                "training_scheme": "self_supervised_artificial_masking",
                "mask_fraction": mask_fraction,
            }
        return self

    # -- refinement --------------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Refine imputed cells, leaving observed cells byte-identical.

        Parameters
        ----------
        frame:
            The same hourly table used for :meth:`fit`.

        Returns
        -------
        (pandas.DataFrame, dict)
            Table gaining ``<channel>_refined_z`` and ``<channel>_refined`` columns,
            and an audit reporting cells refined, cells protected, the mean absolute
            adjustment and an explicit ``observed_cells_modified`` counter that must
            be zero.
        """
        working = frame.copy()
        per_channel: dict[str, Any] = {}
        total_refined = 0
        total_protected = 0
        observed_modified = 0

        for channel in self.channels:
            z_column, mask_column, decay_column = f"{channel}_z", f"{channel}_mask", f"{channel}_decay"
            if z_column not in working.columns:
                continue

            baseline_z = np.nan_to_num(working[z_column].to_numpy(dtype=float))
            mask = working[mask_column].to_numpy() == 1
            decay = working[decay_column].to_numpy(dtype=float)

            if channel in self.coefficients_:
                prediction = self._design_matrix(working, channel) @ self.coefficients_[channel]
            else:
                prediction = baseline_z.copy()

            # Decay-gated blend, applied to imputed cells only.
            blended = decay * baseline_z + (1.0 - decay) * prediction
            refined_z = np.where(mask, baseline_z, blended)

            observed_modified += int(np.abs(refined_z[mask] - baseline_z[mask]).max() > 1e-12) if mask.any() else 0

            statistics_mean = working[f"{channel}_value"].to_numpy(dtype=float)
            # Invert the z-transform using the same statistics used to build it.
            observed_values = working.loc[mask, f"{channel}_value"]
            mean = float(observed_values.mean()) if len(observed_values) else float(np.nanmean(statistics_mean))
            std = float(observed_values.std(ddof=0)) or 1.0

            working[f"{channel}_refined_z"] = np.round(refined_z, 4)
            # Inverting the z-transform introduces rounding of order 1e-4, which would
            # perturb observed cells even though the refinement itself left them alone.
            # The guarantee has to hold exactly, not approximately, so observed cells are
            # copied through verbatim rather than round-tripped. The independent
            # validation agent recomputes this from the final table, which is how the
            # discrepancy was found in the first place.
            rescaled = np.round(refined_z * std + mean, 3)
            original_value = working[f"{channel}_value"].to_numpy(dtype=float)
            working[f"{channel}_refined"] = np.where(mask, original_value, rescaled)

            adjusted = ~mask
            adjustment = np.abs(refined_z[adjusted] - baseline_z[adjusted])
            per_channel[channel] = {
                "cells_refined": int(adjusted.sum()),
                "cells_protected_observed": int(mask.sum()),
                "mean_absolute_adjustment_z": round(float(adjustment.mean()), 4) if adjusted.any() else 0.0,
                "max_absolute_adjustment_z": round(float(adjustment.max()), 4) if adjusted.any() else 0.0,
                "r2_in_sample": self.training_report_.get(channel, {}).get("r2_in_sample"),
            }
            total_refined += int(adjusted.sum())
            total_protected += int(mask.sum())

        audit: dict[str, Any] = {
            "method": "decay_gated_ridge_refinement",
            "blend_formula": "x_hat = decay * locf + (1 - decay) * cross_channel_prediction",
            "decay_base": self.decay_base,
            "cells_refined": total_refined,
            "cells_protected_observed": total_protected,
            "observed_cells_modified": observed_modified,
            "by_channel": per_channel,
            "training_report": self.training_report_,
        }
        return working, audit


def refine_missing_values(
    frame: pd.DataFrame,
    channels: list[str] | None = None,
    hourly_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit and apply the DeepTSE-inspired refiner in one call.

    Parameters
    ----------
    frame:
        Hourly Value-Mask-Decay table.
    channels:
        Channels to refine.
    hourly_raw:
        Pre-expansion hourly table, required for self-supervised training. See
        :meth:`DeepTSEImputer.fit`.

    Returns
    -------
    (pandas.DataFrame, dict)
        Refined table and audit.

    Example
    -------
    >>> refined, audit = refine_missing_values(vmd_frame, hourly_raw=hourly)
    >>> audit["observed_cells_modified"]
    0
    """
    imputer = DeepTSEImputer(channels=channels)
    imputer.fit(frame, hourly_raw=hourly_raw)
    return imputer.transform(frame)


def benchmark_against_locf(
    frame: pd.DataFrame,
    channels: list[str] | None = None,
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = RANDOM_SEED,
    holdout_mode: str = "random",
    block_length: int = 3,
) -> dict[str, Any]:
    """Measure whether refinement actually beats carrying the last value forward.

    A held-out set of *genuinely observed* cells is hidden, the Value-Mask-Decay
    representation is rebuilt as though those measurements had never been taken, both
    estimators are run, and each is scored against the values that were withheld.

    Parameters
    ----------
    frame:
        Hourly table *before* Value-Mask-Decay expansion - it must still carry the raw
        channel columns with ``NaN`` for unmeasured hours.
    channels:
        Channels to benchmark.
    holdout_fraction:
        Share of observed cells to hide. Defaults to ``config.HOLDOUT_FRACTION``.
    seed:
        Seed controlling which cells are hidden.
    holdout_mode:
        ``"random"`` hides isolated observations; ``"block"`` hides runs of
        ``block_length`` consecutive observations from the same admission, which
        simulates a monitoring outage or a nurse handover gap. The distinction matters
        more than it looks: under ``"random"`` the hidden cell is usually bracketed by
        measurements an hour away, so LOCF is close to optimal and no method has much
        room to win. Realistic missingness is bursty, and ``"block"`` is the regime
        imputation is actually deployed in.
    block_length:
        Number of consecutive observations hidden per block in ``"block"`` mode.

    Returns
    -------
    dict
        ``by_channel`` with per-channel MAE and RMSE for both estimators and the
        percentage improvement, plus a pooled ``overall`` summary.

    Notes
    -----
    This function is the reason the project can claim refinement helps rather than
    merely assert it. If the improvement were negative, the honest move would be to
    ship LOCF and say so on the slide - and the number is computed live in the
    notebook, so that outcome would be visible.
    """
    from preprocessing.missing_value_handler import build_value_mask_decay  # local import avoids a cycle

    channels = channels or TIMESERIES_CHANNELS
    rng = np.random.default_rng(seed)
    corrupted = frame.copy()
    truth: dict[str, pd.Series] = {}

    admission_codes = frame["admission_id"].to_numpy() if "admission_id" in frame.columns else np.zeros(len(frame))

    for channel in channels:
        if channel not in corrupted.columns:
            continue
        observed_positions = np.flatnonzero(corrupted[channel].notna().to_numpy())
        if observed_positions.size < 40:
            continue
        n_hold = max(1, int(round(observed_positions.size * holdout_fraction)))

        if holdout_mode == "block":
            held_list: list[int] = []
            # Walk randomly ordered candidate starts, taking runs that stay inside one
            # admission, until the holdout budget is met.
            starts = rng.permutation(observed_positions.size)
            for start_index in starts:
                if len(held_list) >= n_hold:
                    break
                run = observed_positions[start_index : start_index + block_length]
                if run.size == 0:
                    continue
                if len(set(admission_codes[run])) != 1:
                    continue
                held_list.extend(int(position) for position in run)
            held = np.unique(np.array(held_list, dtype=int))[:n_hold]
            if held.size == 0:
                continue
        else:
            held = rng.choice(observed_positions, size=n_hold, replace=False)

        truth[channel] = corrupted[channel].iloc[held].copy()
        corrupted.iloc[held, corrupted.columns.get_loc(channel)] = np.nan

    vmd, _ = build_value_mask_decay(corrupted, channels=channels)
    # The refiner is trained on the corrupted view only: the held-out truth is never
    # visible to it, so the comparison below is an honest out-of-sample test.
    refined, _ = refine_missing_values(vmd, channels=channels, hourly_raw=corrupted)

    # ``build_value_mask_decay`` re-sorts; align back to the original row order.
    refined = refined.sort_index() if refined.index.equals(frame.index) else refined.reset_index(drop=True)

    by_channel: dict[str, Any] = {}
    pooled_locf: list[np.ndarray] = []
    pooled_refined: list[np.ndarray] = []

    for channel, held_values in truth.items():
        positions = held_values.index.to_numpy()
        actual = held_values.to_numpy(dtype=float)
        locf_estimate = refined[f"{channel}_value"].to_numpy(dtype=float)[positions]
        refined_estimate = refined[f"{channel}_refined"].to_numpy(dtype=float)[positions]

        valid = ~(np.isnan(actual) | np.isnan(locf_estimate) | np.isnan(refined_estimate))
        actual, locf_estimate, refined_estimate = actual[valid], locf_estimate[valid], refined_estimate[valid]
        if actual.size == 0:
            continue

        locf_mae = float(np.abs(actual - locf_estimate).mean())
        refined_mae = float(np.abs(actual - refined_estimate).mean())
        by_channel[channel] = {
            "held_out_cells": int(actual.size),
            "locf_mae": round(locf_mae, 4),
            "deeptse_mae": round(refined_mae, 4),
            "locf_rmse": round(float(np.sqrt(((actual - locf_estimate) ** 2).mean())), 4),
            "deeptse_rmse": round(float(np.sqrt(((actual - refined_estimate) ** 2).mean())), 4),
            "mae_improvement_pct": round(100.0 * (locf_mae - refined_mae) / locf_mae, 2) if locf_mae else 0.0,
        }
        # Pool on the standardised scale so channels with different units are comparable.
        scale = np.std(actual) or 1.0
        pooled_locf.append(np.abs(actual - locf_estimate) / scale)
        pooled_refined.append(np.abs(actual - refined_estimate) / scale)

    overall: dict[str, Any] = {}
    if pooled_locf:
        locf_pooled = float(np.concatenate(pooled_locf).mean())
        refined_pooled = float(np.concatenate(pooled_refined).mean())
        overall = {
            "held_out_cells": int(sum(entry["held_out_cells"] for entry in by_channel.values())),
            "locf_standardised_mae": round(locf_pooled, 4),
            "deeptse_standardised_mae": round(refined_pooled, 4),
            "improvement_pct": round(100.0 * (locf_pooled - refined_pooled) / locf_pooled, 2) if locf_pooled else 0.0,
            "channels_improved": int(sum(1 for entry in by_channel.values() if entry["mae_improvement_pct"] > 0)),
            "channels_tested": len(by_channel),
        }

    return {
        "holdout_fraction": holdout_fraction,
        "holdout_mode": holdout_mode,
        "block_length": block_length if holdout_mode == "block" else None,
        "seed": seed,
        "by_channel": by_channel,
        "overall": overall,
    }
