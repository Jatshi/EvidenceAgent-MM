"""Auditable long-horizon tool traces and process rewards for v3 Agentic RL.

The module is deliberately independent of a particular rollout framework.  It
defines the contract that local agents, verl workers, and offline evaluators
must all obey.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from evidenceagent_mm.schema import ResponseStatus


class BudgetExceeded(RuntimeError):
    """Raised before a tool call would exceed an explicit evidence budget."""


class TerminationReason(str, Enum):
    ANSWERED = "answered"
    CLARIFICATION_REQUIRED = "clarification_required"
    EVIDENCE_MISSING = "evidence_missing"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNSAFE_TOOL_OUTPUT = "unsafe_tool_output"


@dataclass(frozen=True, slots=True)
class EvidenceBudget:
    """Immutable accounting state shared between rollout workers."""

    max_steps: int
    max_unique_evidence: int
    max_tool_time_ms: float
    used_steps: int = 0
    used_tool_time_ms: float = 0.0
    unique_evidence_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_unique_evidence <= 0:
            raise ValueError("max_unique_evidence must be positive")
        if self.max_tool_time_ms <= 0:
            raise ValueError("max_tool_time_ms must be positive")

    @property
    def remaining_steps(self) -> int:
        return self.max_steps - self.used_steps

    def consume(self, *, output_ids: list[str], elapsed_ms: float) -> EvidenceBudget:
        if elapsed_ms < 0:
            raise ValueError("elapsed_ms cannot be negative")
        if self.used_steps >= self.max_steps:
            raise BudgetExceeded("step budget exhausted")
        next_ids = self.unique_evidence_ids | frozenset(item for item in output_ids if item)
        if len(next_ids) > self.max_unique_evidence:
            raise BudgetExceeded("unique evidence budget exhausted")
        next_time = self.used_tool_time_ms + elapsed_ms
        if next_time > self.max_tool_time_ms:
            raise BudgetExceeded("tool time budget exhausted")
        return EvidenceBudget(
            max_steps=self.max_steps,
            max_unique_evidence=self.max_unique_evidence,
            max_tool_time_ms=self.max_tool_time_ms,
            used_steps=self.used_steps + 1,
            used_tool_time_ms=next_time,
            unique_evidence_ids=next_ids,
        )


@dataclass(frozen=True, slots=True)
class AgenticStep:
    index: int
    tool: str
    output_ids: list[str]
    elapsed_ms: float
    verified: bool
    injection_detected: bool = False

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("step index cannot be negative")
        if not self.tool:
            raise ValueError("tool cannot be empty")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms cannot be negative")


@dataclass(frozen=True, slots=True)
class AgenticTrace:
    trace_id: str
    terminal_status: ResponseStatus
    termination_reason: TerminationReason
    steps: list[AgenticStep]

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id cannot be empty")
        indexes = [step.index for step in self.steps]
        if indexes != list(range(len(indexes))):
            raise ValueError("step indexes must be contiguous and start at zero")


@dataclass(frozen=True, slots=True)
class ProcessRewardBreakdown:
    total: float
    terminal: float
    evidence_gain: float
    verification: float
    efficiency: float
    safety: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "terminal": self.terminal,
            "evidence_gain": self.evidence_gain,
            "verification": self.verification,
            "efficiency": self.efficiency,
            "safety": self.safety,
        }


_INJECTION_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"ignore\s+(?:all\s+)?previous\s+instructions?",
        r"reveal\s+(?:the\s+)?system\s+prompt",
        r"system\s*:\s*(?:disable|ignore|override)",
        r"忽略(?:之前|以上|所有).{0,12}(?:指令|要求)",
        r"(?:输出|泄露).{0,8}系统提示词",
    )
)


def detect_prompt_injection(text: str) -> bool:
    """Detect obvious instruction injection in untrusted tool observations.

    This is a deterministic safety gate, not a complete injection classifier.
    The raw observation remains evidence; it is never promoted to an instruction.
    """

    normalized = " ".join(text.split())
    return any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS)


def _terminal_consistency(trace: AgenticTrace) -> float:
    expected = {
        ResponseStatus.ANSWERED: TerminationReason.ANSWERED,
        ResponseStatus.NEEDS_CLARIFICATION: TerminationReason.CLARIFICATION_REQUIRED,
        ResponseStatus.ABSTAINED: TerminationReason.EVIDENCE_MISSING,
    }
    return float(expected.get(trace.terminal_status) is trace.termination_reason)


def score_process_trace(trace: AgenticTrace) -> ProcessRewardBreakdown:
    """Score a tool trajectory using only offline-verifiable properties."""

    unique_ids = {item for step in trace.steps for item in step.output_ids if item}
    evidence_gain = min(1.0, len(unique_ids) / 3.0)
    verification = (
        sum(step.verified for step in trace.steps) / len(trace.steps) if trace.steps else 0.0
    )
    total_time = sum(step.elapsed_ms for step in trace.steps)
    efficiency = max(0.0, 1.0 - 0.08 * len(trace.steps) - total_time / 5_000.0)
    safety = float(
        trace.termination_reason is not TerminationReason.UNSAFE_TOOL_OUTPUT
        and not any(step.injection_detected for step in trace.steps)
    )
    terminal = _terminal_consistency(trace)
    total = (
        0.30 * terminal
        + 0.20 * evidence_gain
        + 0.20 * verification
        + 0.10 * efficiency
        + 0.20 * safety
    )
    return ProcessRewardBreakdown(
        total=round(total, 6),
        terminal=terminal,
        evidence_gain=evidence_gain,
        verification=verification,
        efficiency=efficiency,
        safety=safety,
    )
