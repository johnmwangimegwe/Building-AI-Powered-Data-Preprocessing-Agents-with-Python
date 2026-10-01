"""
Module purpose:

Two interchangeable planners for the preprocessing agent - the "brain" half.

``RuleBasedPlanner``
    Reads the data profile and emits a plan by deterministic reasoning over what the
    profile reports. No network, no API key, no latency, identical output every run.
    This is the default, because a conference demonstration that depends on a live
    API call over conference wifi is a demonstration that fails on stage.

``LLMPlanner``
    Sends the profile and the tool catalogue to a language model via LiteLLM and asks
    for a JSON plan. Used when an API key is present.

Both return the same structure, so the agent does not know or care which one it
holds. That substitutability is itself the lesson: the valuable part of an agent is
the **contract** - a validated plan over a closed tool catalogue - not the fact that
an LLM produced it. Whatever the planner proposes is checked against the registry
before anything runs, so a hallucinated tool name or a nonsensical ordering is caught
by the harness rather than by the data.

Author:
AI Medical Preprocessing Project (PyCon Kenya 2026)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from agents.tool_registry import CANONICAL_ORDER, TOOL_REGISTRY, describe_tools

__all__ = ["PlanStep", "RuleBasedPlanner", "LLMPlanner", "build_planner", "PERSONA", "TASK_TEMPLATE"]


PERSONA = """
You are an expert clinical data preprocessing agent.

You are given a compact profile of a messy hospital admission dataset and a fixed
catalogue of preprocessing tools. You do NOT write code and you do NOT see the raw
rows. You select tools, in order, and justify each choice from evidence in the
profile.

You MUST respect stated tool preconditions. In particular, units must be made
canonical before physiological ranges are validated, and timestamps must be parsed
before the data can be reshaped into a time series.

You MUST NOT invent tool names. If the profile shows a problem no tool addresses,
say so in the "concerns" field rather than improvising.
""".strip()


TASK_TEMPLATE = """
TASK
Prepare this hospital admission dataset for clinical time-series modelling.

REQUIREMENTS
1. Remove duplicated observations.
2. Standardise all timestamps to a single unambiguous representation.
3. Convert every measurement to one canonical unit per channel.
4. Standardise categorical fields onto a controlled vocabulary.
5. Reject clinically impossible values without discarding the rest of the row.
6. Reconstruct a regular hourly time series per admission.
7. Represent missingness explicitly as Value, Mask and Decay.
8. Refine imputed cells only, never observed ones.
9. Produce a validation report that a reviewer can check against the raw file.

AVAILABLE TOOLS
{tool_catalogue}

DATASET PROFILE
{profile}

OUTPUT
Return ONLY a JSON object of the form:
{{
  "plan": [
    {{"tool": "<tool name>", "reason": "<evidence from the profile>"}}
  ],
  "concerns": ["<anything the tools cannot address>"]
}}
""".strip()


@dataclass
class PlanStep:
    """One step of an agent plan.

    Attributes
    ----------
    tool:
        Name of a tool in :data:`agents.tool_registry.TOOL_REGISTRY`.
    reason:
        Evidence-backed justification, quoted in the agent trace and the HTML summary.
    kwargs:
        Optional arguments forwarded to the tool.
    """

    tool: str
    reason: str
    kwargs: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the step."""
        return {"tool": self.tool, "reason": self.reason, "kwargs": self.kwargs or {}}


class RuleBasedPlanner:
    """Deterministic planner that reasons directly over the data profile.

    Attributes
    ----------
    name:
        Identifier written into the agent trace.

    Example
    -------
    >>> plan, concerns = RuleBasedPlanner().plan(profile)
    >>> plan[0].tool
    'remove_duplicates'
    """

    name = "rule_based"

    def plan(self, profile: dict[str, Any]) -> tuple[list[PlanStep], list[str]]:
        """Select tools justified by observed defects in the profile.

        Parameters
        ----------
        profile:
            Output of :func:`preprocessing.data_profiler.profile_data`.

        Returns
        -------
        (list[PlanStep], list[str])
            The ordered plan and a list of concerns the tool catalogue cannot address.

        Notes
        -----
        A step is included only when the profile provides evidence for it, with two
        deliberate exceptions. Datetime standardisation and the hourly reconstruction
        run unconditionally, because the task requires a time series regardless of how
        tidy the incoming timestamps happen to be.
        """
        steps: list[PlanStep] = []
        concerns: list[str] = []

        duplicates = profile.get("duplicates", {}).get("exact_duplicate_rows", 0)
        if duplicates:
            steps.append(
                PlanStep(
                    "remove_duplicates",
                    f"Profile reports {duplicates:,} exactly duplicated rows "
                    f"({100.0 * duplicates / max(profile['n_rows'], 1):.1f}% of the table); "
                    "each inflates its patient-hour's weight in any downstream model.",
                )
            )

        n_formats = max((len(spellings) for spellings in profile.get("date_formats", {}).values()), default=0)
        steps.append(
            PlanStep(
                "standardise_datetimes",
                f"{n_formats} distinct datetime spellings detected across the timestamp columns; "
                "day-first values are ambiguous under default parsing and must be resolved by surface form.",
            )
        )

        units = profile.get("unit_inconsistencies", {})
        if units:
            detail = "; ".join(f"{column}: {', '.join(tokens)}" for column, tokens in list(units.items())[:4])
            steps.append(
                PlanStep(
                    "convert_units",
                    f"Mixed unit tokens found ({detail}). Range validation cannot run correctly until units agree.",
                )
            )

        variants = profile.get("categorical_variants", {})
        if variants:
            worst = max(variants.items(), key=lambda item: item[1]["distinct_raw_values"])
            steps.append(
                PlanStep(
                    "standardise_categories",
                    f"Categorical fields carry spelling variants (worst: {worst[0]} with "
                    f"{worst[1]['distinct_raw_values']} raw values); unmerged variants split cohort statistics.",
                )
            )

        implausible = profile.get("implausible_values", {})
        if implausible:
            total = sum(entry["out_of_range"] + entry["unparseable_text"] for entry in implausible.values())
            steps.append(
                PlanStep(
                    "validate_ranges",
                    f"{total:,} cells fall outside physiological bounds or cannot be parsed; "
                    "these are sensor or transcription faults and must become missing, not outliers.",
                )
            )

        steps.append(
            PlanStep(
                "build_hourly_timeseries",
                "Observations are event-based and irregularly spaced; modelling requires a regular "
                "hourly grid spanning each admission window.",
            )
        )
        steps.append(
            PlanStep(
                "build_value_mask_decay",
                "Hourly alignment exposes the true extent of missingness; Value-Mask-Decay preserves "
                "which cells were measured and how much confidence a carried-forward value deserves.",
            )
        )
        steps.append(
            PlanStep(
                "refine_missing_values",
                "Imputed cells with long gaps carry little confidence and can be improved using the "
                "simultaneous state of physiologically coupled channels.",
            )
        )

        high_missing = [
            column for column, pct in profile.get("missing_pct", {}).items() if pct > 25.0
        ]
        if high_missing:
            concerns.append(
                "Channels missing in more than a quarter of raw rows "
                f"({', '.join(sorted(high_missing))}); any imputation here is weakly constrained and "
                "should be flagged to a clinician before modelling."
            )

        return steps, concerns


class LLMPlanner:
    """Planner backed by a language model through LiteLLM.

    Parameters
    ----------
    model_name:
        LiteLLM model identifier, for example ``"gemini/gemini-2.5-flash"`` or
        ``"openai/gpt-4.1"``.
    temperature:
        Sampling temperature. Kept low because this is a selection task, not a
        creative one.
    fallback:
        Planner used when the model is unavailable or returns something unusable.
        Defaults to :class:`RuleBasedPlanner`, so the pipeline degrades to a working
        state rather than to an exception.

    Notes
    -----
    Only a *plan* crosses the network - never patient rows. The profile is a few
    hundred aggregate numbers. This is the property that makes agent-guided
    preprocessing defensible for clinical data at all, and it is worth saying out
    loud: pasting the dataset into a chat window would not be.
    """

    name = "llm"

    def __init__(
        self,
        model_name: str = "gemini/gemini-2.5-flash",
        temperature: float = 0.1,
        fallback: RuleBasedPlanner | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.fallback = fallback or RuleBasedPlanner()
        self.last_raw_response: str | None = None
        self.last_error: str | None = None

    @staticmethod
    def is_available() -> bool:
        """Report whether an API key and LiteLLM are both present.

        Returns
        -------
        bool
            ``True`` when a plan can actually be requested.
        """
        has_key = any(
            os.environ.get(variable)
            for variable in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        )
        if not has_key:
            return False
        try:
            import litellm  # noqa: F401
        except ImportError:
            return False
        return True

    def plan(self, profile: dict[str, Any]) -> tuple[list[PlanStep], list[str]]:
        """Ask the model for a plan, validate it, and fall back if necessary.

        Parameters
        ----------
        profile:
            Output of :func:`preprocessing.data_profiler.profile_data`.

        Returns
        -------
        (list[PlanStep], list[str])
            Validated plan and concerns. Steps naming unknown tools are dropped and
            recorded as concerns; if nothing usable survives, the fallback planner's
            output is returned instead.
        """
        from preprocessing.data_profiler import format_profile

        if not self.is_available():
            self.last_error = "No API key or LiteLLM not installed."
            steps, concerns = self.fallback.plan(profile)
            return steps, [*concerns, f"LLM planner unavailable ({self.last_error}); used rule-based plan."]

        prompt = TASK_TEMPLATE.format(tool_catalogue=describe_tools(), profile=format_profile(profile))

        try:
            from litellm import completion

            response = completion(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": PERSONA},
                    {"role": "user", "content": prompt},
                ],
            )
            self.last_raw_response = response.choices[0].message.content or ""
        except Exception as error:  # noqa: BLE001 - any client failure must degrade, not crash
            self.last_error = f"{type(error).__name__}: {error}"
            steps, concerns = self.fallback.plan(profile)
            return steps, [*concerns, f"LLM call failed ({self.last_error}); used rule-based plan."]

        parsed = self._extract_json(self.last_raw_response)
        if not parsed:
            self.last_error = "Model response contained no parseable JSON object."
            steps, concerns = self.fallback.plan(profile)
            return steps, [*concerns, f"{self.last_error} Used rule-based plan."]

        steps, concerns = [], list(parsed.get("concerns", []))
        for entry in parsed.get("plan", []):
            tool_name = str(entry.get("tool", "")).strip()
            if tool_name not in TOOL_REGISTRY:
                concerns.append(f"Planner proposed unknown tool {tool_name!r}; step rejected before execution.")
                continue
            steps.append(PlanStep(tool_name, str(entry.get("reason", "")).strip() or "No reason given."))

        if not steps:
            fallback_steps, fallback_concerns = self.fallback.plan(profile)
            return fallback_steps, [*concerns, *fallback_concerns, "No valid steps survived validation; used rule-based plan."]

        ordered = self._enforce_preconditions(steps)
        if [step.tool for step in ordered] != [step.tool for step in steps]:
            concerns.append("Proposed step order violated declared tool preconditions and was re-ordered before execution.")
        return ordered, concerns

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Pull the first JSON object out of a model response, fenced or bare."""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            candidate = brace.group(0) if brace else None
        if candidate is None:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _enforce_preconditions(steps: list[PlanStep]) -> list[PlanStep]:
        """Sort the proposed steps into the dependency-safe canonical order."""
        position = {name: index for index, name in enumerate(CANONICAL_ORDER)}
        return sorted(steps, key=lambda step: position.get(step.tool, len(position)))


def build_planner(prefer_llm: bool = False, model_name: str = "gemini/gemini-2.5-flash") -> Any:
    """Choose a planner according to what the environment can actually support.

    Parameters
    ----------
    prefer_llm:
        Request the LLM planner. Ignored when no API key is present.
    model_name:
        LiteLLM model identifier.

    Returns
    -------
    RuleBasedPlanner | LLMPlanner

    Example
    -------
    >>> build_planner().name
    'rule_based'
    """
    if prefer_llm and LLMPlanner.is_available():
        return LLMPlanner(model_name=model_name)
    return RuleBasedPlanner()
