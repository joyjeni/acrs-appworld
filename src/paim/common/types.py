"""Shared data types for the integrated agentic pipeline.

Every objective (O1 SASR, O2 APRR, O3 MNCD, O4 FCNP) reads and writes these
types, which is what makes the four modules composable into a single loop
rather than four disconnected experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

# ---------------------------------------------------------------------------
# Tools (ToolBench API documents, and live data.gov.in resources)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str = ""
    description: str = ""
    default: str = ""
    required: bool = False

    def as_text(self) -> str:
        flag = "required" if self.required else "optional"
        bits = [self.name, f"({self.type or 'any'}, {flag})"]
        if self.description:
            bits.append(self.description)
        return " ".join(bits)


@dataclass(frozen=True)
class Tool:
    """A single callable API endpoint.

    ``doc_id`` is the ToolBench retrieval corpus document id so that our
    rankings can be scored directly against the official ToolBench qrels.
    """

    doc_id: str
    category_name: str
    tool_name: str
    api_name: str
    api_description: str = ""
    method: str = "GET"
    parameters: tuple[ToolParameter, ...] = ()
    source: str = "toolbench"

    @property
    def key(self) -> str:
        return f"{self.tool_name}::{self.api_name}"

    @property
    def required_parameters(self) -> tuple[ToolParameter, ...]:
        return tuple(p for p in self.parameters if p.required)

    def document_text(self) -> str:
        """The indexed representation. Mirrors ToolRetriever's doc rendering."""
        parts = [self.category_name, self.tool_name, self.api_name, self.api_description]
        parts.extend(p.as_text() for p in self.parameters)
        return " ".join(p for p in parts if p)

    def schema_blocks(self) -> list["ContextBlock"]:
        """Break the tool into pruneable context blocks for O4 (FCNP).

        Criticality encodes how damaging removal is: an endpoint's identity and
        its required parameters are near-unprunable, whereas response templates
        and optional-parameter defaults are usually redundant.
        """
        blocks = [
            ContextBlock(
                block_id=f"{self.doc_id}:head",
                kind="tool_signature",
                text=f"{self.tool_name}.{self.api_name} [{self.category_name}] {self.method}",
                criticality=1.0,
                owner=self.key,
            ),
            ContextBlock(
                block_id=f"{self.doc_id}:desc",
                kind="tool_description",
                text=self.api_description or self.api_name,
                criticality=0.65,
                owner=self.key,
            ),
        ]
        for i, p in enumerate(self.parameters):
            blocks.append(
                ContextBlock(
                    block_id=f"{self.doc_id}:param{i}",
                    kind="tool_parameter",
                    text=p.as_text(),
                    criticality=0.9 if p.required else 0.3,
                    owner=self.key,
                )
            )
        return blocks


# ---------------------------------------------------------------------------
# Context blocks (the unit O4/FCNP prunes)
# ---------------------------------------------------------------------------

BlockKind = Literal[
    "user_query",
    "sub_goal",
    "tool_signature",
    "tool_description",
    "tool_parameter",
    "observation",
    "session_memory",
    "agent_note",
]


@dataclass
class ContextBlock:
    """A structured, individually droppable slice of agent context.

    FCNP prunes *structured* blocks rather than free prose, which is the
    difference between it and prompt-compression baselines such as LLMLingua or
    Selective-Context that operate on token streams.
    """

    block_id: str
    kind: BlockKind
    text: str
    criticality: float = 0.5
    owner: str = ""
    turn: int = 0
    tokens: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.tokens:
            # Cheap deterministic proxy for a BPE token count; ~4 chars/token
            # with a floor of one token per whitespace word.
            words = max(1, len(self.text.split()))
            self.tokens = max(words, int(math.ceil(len(self.text) / 4)))


# ---------------------------------------------------------------------------
# Queries, sub-goals, sessions
# ---------------------------------------------------------------------------


@dataclass
class SubGoal:
    """One atomic intent inside a compound user request."""

    sub_goal_id: str
    text: str
    turn: int = 0
    gold_doc_ids: tuple[str, ...] = ()
    status: Literal["open", "satisfied", "failed"] = "open"
    satisfied_by: tuple[str, ...] = ()
    # Consequence class used by the governance layer. "informational" answers
    # may be auto-executed; anything else (an agronomic dose, an entitlement
    # decision) is recommend-only and goes to a named human.
    risk_class: str = "informational"

    @property
    def is_open(self) -> bool:
        return self.status == "open"


@dataclass
class ToolInvocation:
    tool_key: str
    doc_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    latency_ms: float = 0.0
    observation: str = ""
    error: str = ""
    agent_id: str = ""
    # Governance: set by the tool-boundary guard when it neutralised
    # instruction-shaped text in ``observation``.
    quarantined: bool = False
    # Simulator ground truth only (attack harness); never read by the guard,
    # consensus or policy code.
    injected: bool = False
    payload: str = ""


@dataclass
class SessionTurn:
    turn: int
    sub_goal: SubGoal
    selected_doc_ids: tuple[str, ...] = ()
    invocations: tuple[ToolInvocation, ...] = ()
    success: bool = False


@dataclass
class UserProfile:
    """Requester identity. Drives the sector prior used by O1 and O2."""

    email: str = "anonymous@example.com"
    sector: str = "General"
    interests: tuple[str, ...] = ()
    risk_tolerance: float = 0.5
    token_budget: int = 2048


@dataclass
class Session:
    """A multi-turn interaction. The shared state substrate of the framework."""

    session_id: str
    profile: UserProfile
    query: str = ""
    sub_goals: list[SubGoal] = field(default_factory=list)
    turns: list[SessionTurn] = field(default_factory=list)
    source: str = "toolbench"

    @property
    def open_sub_goals(self) -> list[SubGoal]:
        return [g for g in self.sub_goals if g.is_open]

    @property
    def gold_doc_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for g in self.sub_goals:
            for d in g.gold_doc_ids:
                if d not in seen:
                    seen.append(d)
        return tuple(seen)

    def selected_doc_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for t in self.turns:
            for d in t.selected_doc_ids:
                if d not in seen:
                    seen.append(d)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Agents (O2 routing targets, O3 mesh peers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSpec:
    """A heterogeneous agent in the pool.

    ``reliability`` is the probability the agent returns a correct answer on a
    task inside its competence; ``hallucination_rate`` is the probability it
    returns a confident but wrong answer, which is what O3's evidence-weighted
    quorum has to survive.
    """

    agent_id: str
    name: str
    tier: Literal["light", "standard", "heavy", "verifier"]
    capabilities: tuple[str, ...]
    cost_per_call: float
    latency_ms: float
    reliability: float
    hallucination_rate: float = 0.05
    max_parallel: int = 2

    def covers(self, capability: str) -> float:
        """Soft capability match in [0, 1]."""
        if not capability:
            return 0.5
        cap = capability.lower()
        if cap in {c.lower() for c in self.capabilities}:
            return 1.0
        best = 0.0
        for c in self.capabilities:
            c = c.lower()
            if cap in c or c in cap:
                best = max(best, 0.75)
            else:
                overlap = len(set(cap.split()) & set(c.split()))
                if overlap:
                    best = max(best, 0.4 + 0.1 * overlap)
        return min(best, 1.0)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


@dataclass
class StageTrace:
    """One objective's contribution to a single turn, with its own telemetry."""

    stage: str
    objective: str
    label: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "objective": self.objective,
            "label": self.label,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "explanation": self.explanation,
            "durationMs": round(self.duration_ms, 3),
        }


@dataclass
class TurnTrace:
    turn: int
    sub_goal: str
    stages: list[StageTrace] = field(default_factory=list)
    success: bool = False
    answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "subGoal": self.sub_goal,
            "stages": [s.to_dict() for s in self.stages],
            "success": self.success,
            "answer": self.answer,
        }


@dataclass
class PipelineTrace:
    session_id: str
    query: str
    profile: dict[str, Any] = field(default_factory=dict)
    turns: list[TurnTrace] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)
    answer: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "query": self.query,
            "profile": self.profile,
            "turns": [t.to_dict() for t in self.turns],
            "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.metrics.items()},
            "feedback": self.feedback,
            "answer": self.answer,
            "config": self.config,
        }


def iter_tools(tools: Iterable[Tool]) -> dict[str, Tool]:
    return {t.doc_id: t for t in tools}


__all__ = [
    "AgentSpec",
    "BlockKind",
    "ContextBlock",
    "PipelineTrace",
    "Session",
    "SessionTurn",
    "StageTrace",
    "SubGoal",
    "Tool",
    "ToolInvocation",
    "ToolParameter",
    "TurnTrace",
    "UserProfile",
    "iter_tools",
]
