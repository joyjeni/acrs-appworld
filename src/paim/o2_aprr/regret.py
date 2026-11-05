"""Realised-regret bookkeeping for the O2 router.

LinUCB's :math:`\\tilde{O}(d\\sqrt{T})` bound only holds under a linear-reward
assumption that our simulator does not satisfy exactly (rewards involve a
capability *max* and a Bernoulli success draw). So rather than asserting the
bound, we measure realised regret against the simulator's oracle - the agent
with the highest true expected reward for that sub-goal - and report the
empirical regret curve. A sub-linear curve is the claim; this module is the
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from paim.common.types import AgentSpec, Tool


def oracle_expected_reward(agent: AgentSpec, sub_goal_text: str, tools: Sequence[Tool]) -> float:
    """Ground-truth expected reward, available only inside the simulator.

    Must stay consistent with ``paim.o3_mncd.execution.expected_quality``.
    """
    caps = [t.category_name for t in tools] or [sub_goal_text[:40]]
    cap = max((agent.covers(c) for c in caps), default=0.0)
    quality = agent.reliability * (0.55 + 0.45 * cap) * (1.0 - agent.hallucination_rate)
    cost_penalty = 0.35 * min(1.0, agent.cost_per_call / 0.015)
    latency_penalty = 0.15 * min(1.0, agent.latency_ms / 2000.0)
    return max(0.0, quality - cost_penalty - latency_penalty)


@dataclass
class RegretTracker:
    chosen_rewards: list[float] = field(default_factory=list)
    oracle_rewards: list[float] = field(default_factory=list)
    chosen_agents: list[str] = field(default_factory=list)
    oracle_agents: list[str] = field(default_factory=list)

    def record(
        self,
        chosen: AgentSpec,
        candidates: Sequence[AgentSpec],
        sub_goal_text: str,
        tools: Sequence[Tool],
    ) -> float:
        best = max(candidates, key=lambda a: oracle_expected_reward(a, sub_goal_text, tools))
        r_chosen = oracle_expected_reward(chosen, sub_goal_text, tools)
        r_best = oracle_expected_reward(best, sub_goal_text, tools)
        self.chosen_rewards.append(r_chosen)
        self.oracle_rewards.append(r_best)
        self.chosen_agents.append(chosen.agent_id)
        self.oracle_agents.append(best.agent_id)
        return max(0.0, r_best - r_chosen)

    @property
    def instantaneous(self) -> list[float]:
        return [max(0.0, o - c) for o, c in zip(self.oracle_rewards, self.chosen_rewards)]

    @property
    def cumulative(self) -> list[float]:
        total = 0.0
        out = []
        for r in self.instantaneous:
            total += r
            out.append(total)
        return out

    @property
    def total_regret(self) -> float:
        return sum(self.instantaneous)

    @property
    def oracle_agreement(self) -> float:
        if not self.chosen_agents:
            return 0.0
        hits = sum(1 for c, o in zip(self.chosen_agents, self.oracle_agents) if c == o)
        return hits / len(self.chosen_agents)

    def summary(self) -> dict[str, float]:
        n = len(self.chosen_rewards)
        return {
            "decisions": float(n),
            "totalRegret": round(self.total_regret, 5),
            "avgRegret": round(self.total_regret / n, 5) if n else 0.0,
            "oracleAgreement": round(self.oracle_agreement, 5),
            "sqrtTNormalised": round(self.total_regret / (n ** 0.5), 5) if n else 0.0,
        }


__all__ = ["RegretTracker", "oracle_expected_reward"]
