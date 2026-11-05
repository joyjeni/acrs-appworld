"""Contract-Net task allocation inside the mesh.

O2 (APRR) picks the *primary* agent for a sub-goal. Once the sub-goal is
decomposed into concrete tool calls, those calls still have to be spread over
the mesh: the primary cannot always execute them all (``max_parallel``), and
some calls fall outside its competence.

Contract Net (Smith, 1980) is the classical protocol for that, but the
FIPA-style version bids on a scalar utility. Here bids are **three-component
and evidence-aware**: capability match, spare capacity, and the bidder's live
in-session track record on that tool. That last component is the coupling to
the rest of the framework - a peer that has already failed this endpoint in
this session bids lower, so the mesh self-heals without a central scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from paim.common.session_state import SessionState
from paim.common.types import AgentSpec, Tool


@dataclass(frozen=True)
class TaskAnnouncement:
    task_id: str
    tool: Tool
    sub_goal_id: str
    deadline_ms: float = 5000.0


@dataclass(frozen=True)
class Bid:
    agent_id: str
    task_id: str
    capability: float
    capacity: float
    track_record: float
    price: float
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "taskId": self.task_id,
            "capability": round(self.capability, 4),
            "capacity": round(self.capacity, 4),
            "trackRecord": round(self.track_record, 4),
            "price": round(self.price, 6),
            "value": round(self.value, 5),
        }


@dataclass
class Award:
    task_id: str
    agent_id: str
    winning_bid: Bid
    all_bids: list[Bid] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "agentId": self.agent_id,
            "winningBid": self.winning_bid.to_dict(),
            "bids": [b.to_dict() for b in self.all_bids],
            "reason": self.reason,
        }


class ContractNet:
    """Announce -> bid -> award, with per-round load tracking."""

    def __init__(self, agents: Sequence[AgentSpec], lambda_price: float = 0.30) -> None:
        self.agents = list(agents)
        self.lambda_price = lambda_price
        self.load: dict[str, int] = {a.agent_id: 0 for a in self.agents}

    def reset_load(self) -> None:
        self.load = {a.agent_id: 0 for a in self.agents}

    def bid(self, agent: AgentSpec, ann: TaskAnnouncement, state: SessionState) -> Bid | None:
        used = self.load.get(agent.agent_id, 0)
        if used >= agent.max_parallel:
            return None  # honest refusal: no spare capacity

        capability = max(
            agent.covers(ann.tool.category_name),
            agent.covers(ann.tool.tool_name),
            0.15,
        )
        capacity = 1.0 - used / max(1, agent.max_parallel)
        failures = state.tool_failures.get(ann.tool.key, 0)
        track_record = 1.0 / (1.0 + failures)
        price = agent.cost_per_call

        value = (
            0.55 * capability
            + 0.20 * capacity
            + 0.25 * track_record * agent.reliability
            - self.lambda_price * min(1.0, price / 0.015)
        )
        return Bid(
            agent_id=agent.agent_id,
            task_id=ann.task_id,
            capability=capability,
            capacity=capacity,
            track_record=track_record,
            price=price,
            value=value,
        )

    def allocate(
        self, announcements: Sequence[TaskAnnouncement], state: SessionState
    ) -> list[Award]:
        awards: list[Award] = []
        for ann in announcements:
            bids = [b for b in (self.bid(a, ann, state) for a in self.agents) if b is not None]
            if not bids:
                self.reset_load()
                bids = [b for b in (self.bid(a, ann, state) for a in self.agents) if b is not None]
            if not bids:  # pragma: no cover - only if every agent has max_parallel = 0
                continue
            bids.sort(key=lambda b: (-b.value, b.agent_id))
            winner = bids[0]
            self.load[winner.agent_id] = self.load.get(winner.agent_id, 0) + 1
            runner_up = bids[1].value if len(bids) > 1 else None
            reason = (
                f"highest bid {winner.value:.3f}"
                if runner_up is None
                else f"highest bid {winner.value:.3f} vs runner-up {runner_up:.3f}"
            )
            awards.append(Award(ann.task_id, winner.agent_id, winner, bids, reason))
        return awards

    def stats(self) -> dict[str, Any]:
        total = sum(self.load.values())
        return {
            "assignments": total,
            "load": dict(self.load),
            "balance": round(
                1.0 - (max(self.load.values()) - min(self.load.values())) / total, 4
            )
            if total
            else 1.0,
        }


class HubAllocator(ContractNet):
    """Baseline: the orchestrator assigns everything to the primary agent."""

    def __init__(self, agents: Sequence[AgentSpec], primary_id: str) -> None:
        super().__init__(agents)
        self.primary_id = primary_id

    def allocate(self, announcements, state):  # type: ignore[override]
        awards = []
        primary = next((a for a in self.agents if a.agent_id == self.primary_id), self.agents[0])
        for ann in announcements:
            bid = Bid(primary.agent_id, ann.task_id, 1.0, 1.0, 1.0, primary.cost_per_call, 1.0)
            self.load[primary.agent_id] = self.load.get(primary.agent_id, 0) + 1
            awards.append(Award(ann.task_id, primary.agent_id, bid, [bid], "hub assignment"))
        return awards


__all__ = ["Award", "Bid", "ContractNet", "HubAllocator", "TaskAnnouncement"]
