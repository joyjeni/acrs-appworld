"""The heterogeneous agent pool that O2 routes over and O3 deliberates within.

Costs, latencies and reliabilities are declared configuration rather than
measured values: at proposal stage the contribution is the *routing mechanism*,
and the pool is the simulated environment it is evaluated in. Swapping this
table for measured per-model numbers is the only change needed to move from
simulation to a live model pool, and ``AgentPool.from_dicts`` exists for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from paim.common.types import AgentSpec

# Capability vocabulary is aligned with ToolBench category names plus the
# data.gov.in Agriculture domain used for the live-data walkthrough.
DEFAULT_AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        agent_id="a_light_lookup",
        name="Lookup Agent (light)",
        tier="light",
        capabilities=("data", "search", "database", "finance", "location", "mapping"),
        cost_per_call=0.0004,
        latency_ms=180.0,
        reliability=0.72,
        hallucination_rate=0.10,
        max_parallel=4,
    ),
    AgentSpec(
        agent_id="a_agri_specialist",
        name="Agriculture Data Specialist",
        tier="standard",
        capabilities=(
            "agriculture",
            "commodity price",
            "mandi",
            "crop",
            "weather",
            "open government data",
            "data",
        ),
        cost_per_call=0.0021,
        latency_ms=420.0,
        reliability=0.90,
        hallucination_rate=0.04,
        max_parallel=2,
    ),
    AgentSpec(
        agent_id="a_geo_specialist",
        name="Geospatial & Transport Specialist",
        tier="standard",
        capabilities=("location", "mapping", "transportation", "travel", "logistics", "weather"),
        cost_per_call=0.0018,
        latency_ms=380.0,
        reliability=0.87,
        hallucination_rate=0.05,
        max_parallel=2,
    ),
    AgentSpec(
        agent_id="a_reasoner_heavy",
        name="Compositional Reasoner (heavy)",
        tier="heavy",
        capabilities=(
            "reasoning",
            "composition",
            "multi step",
            "finance",
            "business",
            "data",
            "agriculture",
            "location",
            "media",
            "social",
        ),
        cost_per_call=0.0135,
        latency_ms=1750.0,
        reliability=0.94,
        hallucination_rate=0.02,
        max_parallel=1,
    ),
    AgentSpec(
        agent_id="a_verifier",
        name="Evidence Verifier",
        tier="verifier",
        capabilities=("verification", "grounding", "citation", "consistency"),
        cost_per_call=0.0026,
        latency_ms=310.0,
        reliability=0.92,
        hallucination_rate=0.01,
        max_parallel=3,
    ),
)


@dataclass
class AgentPool:
    agents: tuple[AgentSpec, ...] = DEFAULT_AGENTS

    @staticmethod
    def from_dicts(rows: Iterable[dict[str, Any]]) -> "AgentPool":
        return AgentPool(tuple(AgentSpec(**r) for r in rows))

    def __iter__(self):
        return iter(self.agents)

    def __len__(self) -> int:
        return len(self.agents)

    def __getitem__(self, agent_id: str) -> AgentSpec:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        raise KeyError(agent_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(a.agent_id for a in self.agents)

    def routable(self) -> tuple[AgentSpec, ...]:
        """Verifiers are recruited by O3, never selected as the primary solver."""
        return tuple(a for a in self.agents if a.tier != "verifier")

    def verifiers(self) -> tuple[AgentSpec, ...]:
        return tuple(a for a in self.agents if a.tier == "verifier")

    def by_tier(self, tier: str) -> tuple[AgentSpec, ...]:
        return tuple(a for a in self.agents if a.tier == tier)

    def escalation_ladder(self) -> Sequence[AgentSpec]:
        order = {"light": 0, "standard": 1, "heavy": 2, "verifier": 3}
        return sorted(self.routable(), key=lambda a: (order[a.tier], a.cost_per_call))

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "agentId": a.agent_id,
                "name": a.name,
                "tier": a.tier,
                "capabilities": list(a.capabilities),
                "costPerCall": a.cost_per_call,
                "latencyMs": a.latency_ms,
                "reliability": a.reliability,
                "hallucinationRate": a.hallucination_rate,
                "maxParallel": a.max_parallel,
            }
            for a in self.agents
        ]


__all__ = ["DEFAULT_AGENTS", "AgentPool"]
