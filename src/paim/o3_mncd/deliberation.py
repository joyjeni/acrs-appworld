"""The O3 deliberation round: recruit peers, allocate calls, gossip, certify.

This is the module the orchestrator calls. It ties together the three pieces of
O3 - mesh topology, contract-net allocation, evidence-weighted consensus - into
one auditable round per sub-goal.

Round structure
---------------
1. **Recruit.** Start from the primary agent chosen by O2, add its mesh
   neighbours up to the deliberation width, and add a verifier when the
   sub-goal is high-stakes. Recruitment is topology-driven, so the cost of
   deliberation is bounded by the mesh degree rather than the pool size.
2. **Allocate.** Announce one task per selected tool; peers bid; contract net
   awards. A peer with no spare capacity or a bad in-session record on that
   endpoint loses the bid.
3. **Execute.** Each peer runs its awarded calls and forms a claim.
4. **Gossip.** Peers exchange claim distributions over the mesh for
   ``rounds_needed()`` rounds. The convergence log is recorded; gossip does not
   change the certificate (beliefs are advisory), which keeps the safety
   argument independent of the mixing assumption.
5. **Certify.** Evidence-weighted quorum, or abstain and request escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from paim.common.session_state import SessionState
from paim.common.types import AgentSpec, Tool
from paim.o2_aprr.agents import AgentPool
from paim.o3_mncd.consensus import (
    ConsensusCertificate,
    ConsensusConfig,
    EvidenceWeightedConsensus,
)
from paim.o3_mncd.contract_net import ContractNet, TaskAnnouncement
from paim.o3_mncd.execution import AgentReport, AgentRuntime
from paim.o3_mncd.mesh import MeshTopology


@dataclass(frozen=True)
class DeliberationConfig:
    width: int = 3               # peers recruited per sub-goal (incl. primary)
    mesh_degree: int = 4
    gossip_mixing: float = 0.5
    recruit_verifier_when_high_stakes: bool = True
    max_tools_per_round: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "meshDegree": self.mesh_degree,
            "gossipMixing": self.gossip_mixing,
            "recruitVerifierWhenHighStakes": self.recruit_verifier_when_high_stakes,
            "maxToolsPerRound": self.max_tools_per_round,
        }


@dataclass
class DeliberationResult:
    certificate: ConsensusCertificate
    reports: list[AgentReport] = field(default_factory=list)
    awards: list[dict[str, Any]] = field(default_factory=list)
    peers: tuple[str, ...] = ()
    gossip_log: list[dict[str, float]] = field(default_factory=list)
    mesh_stats: dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    wall_latency_ms: float = 0.0
    n_tool_calls: int = 0
    n_tool_failures: int = 0

    def to_dict(self, include_ground_truth: bool = False) -> dict[str, Any]:
        return {
            "certificate": self.certificate.to_dict(),
            "reports": [r.to_dict(include_ground_truth) for r in self.reports],
            "awards": self.awards,
            "peers": list(self.peers),
            "gossipLog": self.gossip_log,
            "meshStats": self.mesh_stats,
            "totalCost": round(self.total_cost, 6),
            "totalLatencyMs": round(self.total_latency_ms, 2),
            "wallLatencyMs": round(self.wall_latency_ms, 2),
            "toolCalls": self.n_tool_calls,
            "toolFailures": self.n_tool_failures,
        }


class MeshDeliberation:
    objective = "O3"
    name = "MNCD"

    def __init__(
        self,
        pool: AgentPool | None = None,
        executor: Any = None,
        config: DeliberationConfig | None = None,
        consensus: EvidenceWeightedConsensus | None = None,
        seed: int = 20260905,
    ) -> None:
        from paim.o3_mncd.execution import SimulatedToolExecutor

        self.pool = pool or AgentPool()
        self.config = config or DeliberationConfig()
        self.consensus = consensus or EvidenceWeightedConsensus(ConsensusConfig())
        self.executor = executor or SimulatedToolExecutor(seed=seed)
        self.runtime = AgentRuntime(self.executor, seed=seed)
        self.mesh = MeshTopology(self.pool.ids, degree=self.config.mesh_degree)
        self.rounds = 0

    # -- step 1 ------------------------------------------------------------
    def recruit(self, primary_id: str, high_stakes: bool) -> list[AgentSpec]:
        peers = [primary_id]
        for nb in self.mesh.adjacency.get(primary_id, ()):
            if len(peers) >= self.config.width:
                break
            if self.pool[nb].tier == "verifier":
                continue
            peers.append(nb)
        # Top up from the ladder if the neighbourhood was too small.
        for a in self.pool.escalation_ladder():
            if len(peers) >= self.config.width:
                break
            if a.agent_id not in peers:
                peers.append(a.agent_id)
        if high_stakes and self.config.recruit_verifier_when_high_stakes:
            for v in self.pool.verifiers():
                if v.agent_id not in peers:
                    peers.append(v.agent_id)
                    break
        return [self.pool[p] for p in peers]

    # -- the round ---------------------------------------------------------
    def deliberate(
        self,
        sub_goal_id: str,
        sub_goal_text: str,
        tools: Sequence[Tool],
        primary_id: str,
        state: SessionState,
        high_stakes: bool = False,
        arguments: dict[str, dict[str, Any]] | None = None,
    ) -> DeliberationResult:
        self.rounds += 1
        nonce = f"t{state.turn}r{self.rounds}"
        tools = list(tools)[: self.config.max_tools_per_round]
        peers = self.recruit(primary_id, high_stakes)

        # 2. contract-net allocation of the concrete tool calls
        net = ContractNet(peers)
        announcements = [
            TaskAnnouncement(task_id=f"{sub_goal_id}:{t.doc_id}", tool=t, sub_goal_id=sub_goal_id)
            for t in tools
        ]
        awards = net.allocate(announcements, state)
        by_agent: dict[str, list[Tool]] = {p.agent_id: [] for p in peers}
        # Resolve awards through the announcement table, not by parsing the
        # task id: sub-goal ids may themselves contain ':' (e.g. KCC "kcc0:3").
        tool_by_task = {a.task_id: a.tool for a in announcements}
        for aw in awards:
            by_agent.setdefault(aw.agent_id, []).append(tool_by_task[aw.task_id])

        # 3a. execute: each awarded endpoint is called exactly once, by its winner
        own_invocations: dict[str, list[Any]] = {}
        pool: list[Any] = []
        for peer in peers:
            assigned = by_agent.get(peer.agent_id) or []
            invs = self.runtime.execute(peer, assigned, arguments=arguments, nonce=nonce)
            own_invocations[peer.agent_id] = invs
            pool.extend(invs)

        # 3b. circulate the evidence pool over the mesh, then let every peer
        # form its own claim over the *same* observations. Independent claims
        # over shared evidence is what makes agreement informative; peers
        # reasoning over disjoint evidence would disagree by construction.
        sub_mesh = MeshTopology(tuple(p.agent_id for p in peers), degree=self.config.mesh_degree)
        reports: list[AgentReport] = [
            self.runtime.form_claim(
                agent=peer,
                sub_goal_id=sub_goal_id,
                sub_goal_text=sub_goal_text,
                pool=pool,
                tools=tools,
                own_invocations=own_invocations.get(peer.agent_id, []),
                nonce=nonce,
            )
            for peer in peers
        ]

        # 4. gossip: claim distributions mix over the sub-mesh. Advisory only -
        # the certificate does not depend on the mixing assumption.
        beliefs = {r.agent_id: {r.claim_key: max(0.05, r.confidence)} for r in reports}
        _, gossip_log = sub_mesh.gossip(beliefs, mixing=self.config.gossip_mixing)

        # 5. certify
        reliabilities = {p.agent_id: p.reliability for p in peers}
        certificate = self.consensus.certify(sub_goal_id, reports, reliabilities)

        n_calls = len(pool)
        n_fail = sum(1 for i in pool if not i.ok)
        return DeliberationResult(
            certificate=certificate,
            reports=reports,
            awards=[a.to_dict() for a in awards],
            peers=tuple(p.agent_id for p in peers),
            gossip_log=gossip_log,
            mesh_stats=sub_mesh.stats(),
            total_cost=sum(r.cost for r in reports),
            total_latency_ms=sum(r.latency_ms for r in reports),
            wall_latency_ms=max((r.latency_ms for r in reports), default=0.0),
            n_tool_calls=n_calls,
            n_tool_failures=n_fail,
        )


__all__ = ["DeliberationConfig", "DeliberationResult", "MeshDeliberation"]
