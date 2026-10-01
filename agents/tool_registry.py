"""
Module purpose:

The tool registry - the agent's hands.

An agent is an LLM for reasoning plus *tools* for acting. This module is the tool
half: every deterministic preprocessing function in the project is wrapped in a
uniform ``Tool`` record carrying a name, a natural-language description, a declared
precondition and a declared postcondition.

Two properties follow from this design, and both matter more than they first appear.

**The planner never writes code.** It selects tool names and argument dictionaries
from a fixed catalogue. An LLM that emits an unknown tool name gets a rejection, not
an execution. This is the deliberate difference between this project and a free-form
CodeAct agent: the reference implementation lets the model write arbitrary pandas and
constrains it with an import allow-list, which is powerful and open-ended; a clinical
pipeline trades that generality for a closed action space that can be audited before
it runs.

**Every tool returns an audit.** Each wrapped function has the signature
``frame -> (frame, audit_dict)``. The agent therefore accumulates a complete,
structured record of what happened without any tool needing to know that an agent
exists. The same functions are directly callable from a plain script.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from models.deeptse_imputer import refine_missing_values
from preprocessing.categorical_cleaner import standardise_categories
from preprocessing.date_cleaner import standardise_datetimes
from preprocessing.duplicate_handler import remove_duplicates
from preprocessing.missing_value_handler import build_value_mask_decay
from preprocessing.range_validator import validate_ranges
from preprocessing.timeseries_builder import build_hourly_timeseries
from preprocessing.unit_converter import convert_units

__all__ = ["Tool", "TOOL_REGISTRY", "describe_tools", "get_tool"]


@dataclass(frozen=True)
class Tool:
    """A single auditable preprocessing action available to the agent.

    Attributes
    ----------
    name:
        Stable identifier the planner emits.
    description:
        One-sentence natural-language summary, shown to the planning LLM.
    function:
        Callable with signature ``(frame, **kwargs) -> (frame, audit)``.
    requires:
        Human-readable precondition. Used by the agent to order the plan and to
        explain a rejection when a tool is requested out of sequence.
    produces:
        Human-readable postcondition.
    stage:
        ``"clean"`` for tools operating on the event table, ``"reshape"`` for the
        transition to hourly data, ``"impute"`` for the time-series stage.
    default_kwargs:
        Arguments applied when the planner supplies none.
    """

    name: str
    description: str
    function: Callable[..., tuple[pd.DataFrame, dict[str, Any]]]
    requires: str
    produces: str
    stage: str
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def run(self, frame: pd.DataFrame, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Execute the tool with defaults filled in.

        Parameters
        ----------
        frame:
            Current state of the table.
        **kwargs:
            Planner-supplied arguments, overriding ``default_kwargs``.

        Returns
        -------
        (pandas.DataFrame, dict)
            New table state and the tool's audit record.
        """
        merged = {**self.default_kwargs, **kwargs}
        return self.function(frame, **merged)


#: The closed action space. Adding a capability means adding an entry here; there is no
#: other way for the agent to affect the data.
TOOL_REGISTRY: dict[str, Tool] = {
    "remove_duplicates": Tool(
        name="remove_duplicates",
        description=(
            "Drop exactly duplicated observation rows and record the original row index "
            "of every row removed."
        ),
        function=remove_duplicates,
        requires="Raw event table as loaded from CSV.",
        produces="Event table with no exact duplicate rows.",
        stage="clean",
    ),
    "standardise_datetimes": Tool(
        name="standardise_datetimes",
        description=(
            "Parse intime, outtime and charttime from their mixed surface formats into "
            "true timestamps, reconcile one admission window per admission_id, and derive "
            "length of stay in hours."
        ),
        function=standardise_datetimes,
        requires="Duplicates removed, so that window reconciliation votes on distinct rows.",
        produces="datetime64 columns plus length_of_stay_hours.",
        stage="clean",
    ),
    "convert_units": Tool(
        name="convert_units",
        description=(
            "Normalise every measurement column to one canonical unit: temperature to "
            "Celsius, glucose to mg/dL, and strip redundant unit suffixes elsewhere."
        ),
        function=convert_units,
        requires="Raw measurement columns, possibly holding unit-suffixed strings.",
        produces="Float measurement columns in canonical units.",
        stage="clean",
    ),
    "standardise_categories": Tool(
        name="standardise_categories",
        description=(
            "Map categorical fields onto a controlled vocabulary and list any value the "
            "vocabulary does not cover."
        ),
        function=standardise_categories,
        requires="Categorical columns present.",
        produces="Categorical columns drawn from a closed vocabulary.",
        stage="clean",
    ),
    "validate_ranges": Tool(
        name="validate_ranges",
        description=(
            "Set clinically impossible measurements to missing, keeping the row, and log "
            "every rejection with its admission, time, channel and value."
        ),
        function=validate_ranges,
        requires="Units already canonical -- otherwise a Fahrenheit temperature is rejected for being out of Celsius range.",
        produces="Measurement columns containing only physiologically plausible values.",
        stage="clean",
    ),
    "build_hourly_timeseries": Tool(
        name="build_hourly_timeseries",
        description=(
            "Expand the irregular event table into one row per admission-hour across each "
            "admission window, averaging observations that share an hour."
        ),
        function=build_hourly_timeseries,
        requires="Parsed datetimes and reconciled admission windows.",
        produces="Regular hourly clinical time series.",
        stage="reshape",
    ),
    "build_value_mask_decay": Tool(
        name="build_value_mask_decay",
        description=(
            "Expand each channel into Value (neutral fill then LOCF), Mask (1 observed, "
            "0 imputed) and Decay (0.75 ** hours since the last real observation), then "
            "standardise."
        ),
        function=build_value_mask_decay,
        requires="Hourly time series.",
        produces="Model-ready Value-Mask-Decay representation.",
        stage="impute",
    ),
    "refine_missing_values": Tool(
        name="refine_missing_values",
        description=(
            "DeepTSE-inspired refinement of imputed cells only, blending the carried-forward "
            "value with a cross-channel prediction in proportion to the decay weight. Never "
            "modifies an observed cell."
        ),
        function=refine_missing_values,
        requires="Value-Mask-Decay representation.",
        produces="Refined estimates at every cell whose mask is zero.",
        stage="impute",
    ),
}

#: Canonical execution order. The agent's rule-based planner emits this; an LLM planner
#: is validated against it, and any deviation is reported rather than silently accepted.
CANONICAL_ORDER: tuple[str, ...] = (
    "remove_duplicates",
    "standardise_datetimes",
    "convert_units",
    "standardise_categories",
    "validate_ranges",
    "build_hourly_timeseries",
    "build_value_mask_decay",
    "refine_missing_values",
)


def get_tool(name: str) -> Tool:
    """Look up a tool by name.

    Parameters
    ----------
    name:
        Tool identifier.

    Returns
    -------
    Tool

    Raises
    ------
    KeyError
        If the name is not in the registry. The error message lists the valid names,
        which is what lets the agent recover on the next planning turn.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool {name!r}. Available tools: {sorted(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[name]


def describe_tools() -> str:
    """Render the tool catalogue as the text block handed to a planning LLM.

    Returns
    -------
    str
        One block per tool giving its name, description, precondition and
        postcondition.

    Example
    -------
    >>> print(describe_tools()[:60])
    remove_duplicates [clean]
      Drop exactly duplicated
    """
    blocks: list[str] = []
    for name in CANONICAL_ORDER:
        tool = TOOL_REGISTRY[name]
        blocks.append(
            f"{tool.name} [{tool.stage}]\n"
            f"  {tool.description}\n"
            f"  requires: {tool.requires}\n"
            f"  produces: {tool.produces}"
        )
    return "\n\n".join(blocks)
