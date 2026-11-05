"""Tool execution and per-agent claim formation inside the mesh.

Two execution back-ends share one interface:

``SimulatedToolExecutor``
    Deterministic synthetic responses derived from a ToolBench API schema. The
    ToolBench corpus documents 10k+ RapidAPI endpoints whose live keys are not
    redistributable, so trajectories over them are simulated. Faults are
    injected from a seeded stream, which makes robustness experiments
    replayable - the point of the simulator is to exercise the *pipeline*, and
    it is labelled as simulation everywhere it appears in a trace.

``DataGovExecutor`` (in ``paim.data.datagov``)
    Real HTTPS calls to ``api.data.gov.in``. This is the live-data leg of the
    evaluation: the same orchestrator, the same consensus, real responses.

Agent claim formation is where hallucination enters the system. A faithful
agent's claim is a digest of what the tool actually returned; a hallucinating
agent emits a *confident* claim digest that does not match the observation.
That is exactly the failure mode ``consensus.py`` has to survive, and majority
voting does not, because hallucinating agents can agree with each other.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from paim.common.rng import SeededStreams
from paim.common.types import AgentSpec, Tool, ToolInvocation


class ToolExecutor(Protocol):
    name: str
    live: bool

    def execute(self, tool: Tool, arguments: dict[str, Any], nonce: str = "") -> ToolInvocation: ...


def digest(payload: Any, length: int = 10) -> str:
    """Stable content digest used for claim identity and evidence tokens."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def evidence_token(doc_id: str, observation: str) -> str:
    """The canonical citation token for one tool observation."""
    return f"{doc_id}:{digest(observation, 6)}"


@dataclass
class SimulatedToolExecutor:
    """Schema-faithful synthetic tool responses with seeded fault injection."""

    seed: int = 20260905
    failure_rate: float = 0.08
    name: str = "simulated"
    live: bool = False
    calls: int = 0

    def __post_init__(self) -> None:
        self._streams = SeededStreams(self.seed)

    def execute(self, tool: Tool, arguments: dict[str, Any], nonce: str = "") -> ToolInvocation:
        self.calls += 1
        label = f"{tool.doc_id}|{nonce}"
        stream = self._streams.stream("exec", label)

        missing = [p.name for p in tool.required_parameters if p.name not in arguments]
        if missing:
            return ToolInvocation(
                tool_key=tool.key,
                doc_id=tool.doc_id,
                arguments=arguments,
                ok=False,
                latency_ms=12.0,
                error=f"missing required parameter(s): {', '.join(missing)}",
            )

        if stream.random() < self.failure_rate:
            return ToolInvocation(
                tool_key=tool.key,
                doc_id=tool.doc_id,
                arguments=arguments,
                ok=False,
                latency_ms=round(stream.uniform(80, 400), 2),
                error="upstream 503 (simulated transient failure)",
            )

        payload = {
            "endpoint": tool.key,
            "category": tool.category_name,
            "arguments": arguments,
            "records": stream.randint(1, 25),
            "checksum": digest([tool.doc_id, sorted(arguments.items())]),
        }
        return ToolInvocation(
            tool_key=tool.key,
            doc_id=tool.doc_id,
            arguments=arguments,
            ok=True,
            latency_ms=round(stream.uniform(60, 520), 2),
            observation=json.dumps(payload, sort_keys=True),
        )


@dataclass
class AgentReport:
    """One mesh peer's contribution to a sub-goal.

    ``hallucinated`` is ground truth from the simulator. It is used only for
    evaluation and is never visible to the consensus mechanism - asserted by
    ``tests/test_o3_mncd.py``.
    """

    agent_id: str
    sub_goal_id: str
    claim: str
    claim_key: str
    confidence: float
    evidence: tuple[str, ...] = ()
    ok: bool = True
    latency_ms: float = 0.0
    cost: float = 0.0
    invocations: tuple[ToolInvocation, ...] = ()
    hallucinated: bool = False
    # Simulator ground truth: the peer followed an instruction embedded in a
    # tool observation (indirect prompt injection). Evaluation-only.
    hijacked: bool = False
    error: str = ""

    def to_dict(self, include_ground_truth: bool = False) -> dict[str, Any]:
        out = {
            "agentId": self.agent_id,
            "subGoalId": self.sub_goal_id,
            "claim": self.claim,
            "claimKey": self.claim_key,
            "confidence": round(self.confidence, 4),
            "evidence": list(self.evidence),
            "ok": self.ok,
            "latencyMs": round(self.latency_ms, 2),
            "cost": round(self.cost, 6),
            "error": self.error,
        }
        if include_ground_truth:
            out["hallucinated"] = self.hallucinated
            out["hijacked"] = self.hijacked
        return out


def expected_quality(agent: AgentSpec, sub_goal_text: str, tools: Sequence[Tool]) -> float:
    """Ground-truth success probability. Mirrors ``o2_aprr.regret``."""
    caps = [t.category_name for t in tools] or [sub_goal_text[:40]]
    cap = max((agent.covers(c) for c in caps), default=0.0)
    return min(1.0, agent.reliability * (0.55 + 0.45 * cap))


@dataclass
class AgentRuntime:
    """Simulated agent behaviour: execute awarded calls, then form a claim."""

    executor: Any
    seed: int = 20260905
    _streams: SeededStreams = field(init=False)

    def __post_init__(self) -> None:
        self._streams = SeededStreams(self.seed)

    def execute(
        self,
        agent: AgentSpec,
        tools: Sequence[Tool],
        arguments: dict[str, dict[str, Any]] | None = None,
        nonce: str = "",
    ) -> list[ToolInvocation]:
        """Phase A: run the tool calls contract net awarded to this peer.

        Each awarded endpoint is called exactly once by exactly one peer, which
        is the whole point of contract-net allocation - duplicating every call
        across every peer is what makes debate-style deliberation expensive.
        """
        arguments = arguments or {}
        invocations: list[ToolInvocation] = []
        for tool in tools:
            inv = self.executor.execute(
                tool, arguments.get(tool.doc_id, {}), nonce=f"{agent.agent_id}|{nonce}"
            )
            inv.agent_id = agent.agent_id
            invocations.append(inv)
        return invocations

    def form_claim(
        self,
        agent: AgentSpec,
        sub_goal_id: str,
        sub_goal_text: str,
        pool: Sequence[ToolInvocation],
        tools: Sequence[Tool],
        own_invocations: Sequence[ToolInvocation] = (),
        nonce: str = "",
    ) -> AgentReport:
        """Phase C: reason over the mesh's *shared* observation pool.

        This is the step that makes agreement meaningful. Every recruited peer
        sees the same evidence (gathered in phase A, circulated in phase B), so
        two faithful peers necessarily produce the same claim digest, while a
        peer that fabricates diverges from the pool and cites nothing in it.
        Under the earlier design, where each peer only saw the calls it happened
        to win, faithful peers disagreed by construction and the quorum could
        never form.
        """
        invocations = list(own_invocations)
        latency = agent.latency_ms + sum(i.latency_ms for i in invocations)
        cost = agent.cost_per_call
        ok_obs = [i for i in pool if i.ok]
        stream = self._streams.stream("agent", agent.agent_id, sub_goal_id, nonce)

        if not ok_obs:
            return AgentReport(
                agent_id=agent.agent_id,
                sub_goal_id=sub_goal_id,
                claim="no groundable observation",
                claim_key="__failed__",
                confidence=0.05,
                ok=False,
                latency_ms=latency,
                cost=cost,
                invocations=tuple(invocations),
                error="; ".join(i.error for i in pool if i.error) or "all tool calls failed",
            )

        # Ground truth for this draw: the agent either answers correctly, or
        # fails; a failure becomes a *confident* fabrication with probability
        # proportional to the agent's hallucination propensity, and otherwise
        # becomes an honest low-confidence partial answer.
        succeeded = stream.random() < expected_quality(agent, sub_goal_text, tools)
        hallucinated = (not succeeded) and stream.random() < min(
            1.0, agent.hallucination_rate * 6.0
        )

        grounded_key = digest([sub_goal_id, sorted(i.observation for i in ok_obs)])
        evidence = tuple(evidence_token(i.doc_id, i.observation) for i in ok_obs)

        if succeeded:
            claim_key = grounded_key
            claim = f"grounded answer for '{_short(sub_goal_text)}' from {len(ok_obs)} observation(s)"
            confidence = min(0.99, 0.62 + 0.3 * agent.reliability)
        elif hallucinated:
            # Confidently wrong, and *cites nothing it actually retrieved*.
            claim_key = digest([sub_goal_id, "hallucination", agent.agent_id])
            claim = f"unsupported answer for '{_short(sub_goal_text)}'"
            confidence = min(0.99, 0.80 + 0.15 * stream.random())
            evidence = ()
        else:
            claim_key = digest([sub_goal_id, "abstain", agent.agent_id])
            claim = f"low-confidence partial answer for '{_short(sub_goal_text)}'"
            confidence = 0.25 + 0.2 * stream.random()

        # Indirect prompt injection: if an instruction payload survived into
        # the shared pool, a susceptible peer follows it. Every hijacked peer
        # produces the *same* attacker claim and cites real evidence, so this
        # is a correlated error. The extra draw happens only when a live
        # payload exists, so attack-free runs reproduce earlier results exactly.
        hijacked = False
        live = [i for i in ok_obs if i.payload and i.payload in i.observation]
        if live:
            susceptibility = min(0.95, INJECTION_SUSCEPTIBILITY + 0.5 * (1.0 - agent.reliability))
            if stream.random() < susceptibility:
                hijacked = True
                hallucinated = False
                claim_key = digest([sub_goal_id, "injected", sorted(i.payload for i in live)])
                claim = f"attacker-steered answer for '{_short(sub_goal_text)}'"
                confidence = min(0.99, 0.85 + 0.1 * stream.random())
                evidence = tuple(evidence_token(i.doc_id, i.observation) for i in ok_obs)

        return AgentReport(
            agent_id=agent.agent_id,
            sub_goal_id=sub_goal_id,
            claim=claim,
            claim_key=claim_key,
            confidence=confidence,
            evidence=evidence,
            ok=True,
            latency_ms=latency,
            cost=cost,
            invocations=tuple(invocations),
            hallucinated=hallucinated,
            hijacked=hijacked,
        )

    @staticmethod
    def grounded_key(sub_goal_id: str, observations: Sequence[str]) -> str:
        return digest([sub_goal_id, sorted(observations)])


INJECTION_SUSCEPTIBILITY = 0.55


def _short(text: str, n: int = 48) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "\u2026"


__all__ = [
    "AgentReport",
    "AgentRuntime",
    "SimulatedToolExecutor",
    "ToolExecutor",
    "digest",
    "evidence_token",
    "expected_quality",
]
