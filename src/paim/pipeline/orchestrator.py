"""The integrated pipeline: all four objectives in one closed loop.

This module is the answer to "how are the objectives connected?". Each turn of
a session runs five stages and then closes three feedback edges.

Forward path (per sub-goal)
---------------------------
::

    S0  session state      intent is written into the decayed memory (O1 substrate)
    S1  O1  SASR           BM25/dense retrieve -> session-aware marginal-utility rerank
    S2  O4  FCNP           assemble typed context, PI-controlled prune under fidelity floor
    S3  O2  APRR           risk-adjusted bandit routing of the sub-goal to an agent
    S4  O3  MNCD           mesh recruit -> contract-net -> execute -> gossip -> certify
    S5  feedback           credit assignment back into O1, O2 and O4

Feedback edges (what makes it a loop, not a cascade)
----------------------------------------------------
* **O3 -> O2**  the consensus certificate *is* the bandit reward. The router is
  never trained on a label; it learns from whether the mesh could certify the
  answer at the cost it paid.
* **O3 -> O1**  failed tool calls raise the per-endpoint failure penalty and the
  session risk, so a flaky endpoint sinks in later turns of the same session.
* **O3 -> O2 (control)**  an abstained certificate triggers re-routing with a
  raised risk level, which makes O2's escalation policy fire. Abstention
  therefore buys a stronger agent rather than returning a guess.
* **O4 -> O1**  surviving observations are carried forward as context blocks and
  are what the next turn's session memory is built over; anything O4 pruned is
  genuinely gone, so O1 cannot depend on context the model never sees.
* **O1 -> O4**  the set of committed tools defines O4's schema-integrity
  hard-keep set, so retrieval decisions bound how far compression may go.
* **O1 -> O2**  the reranked, schema-intact tool set is the capability context
  the router builds its features from.

Ablations
---------
``PipelineConfig.ablations`` switches each objective back to its SOTA analogue
(stateless reranking, static routing, majority voting, fixed-ratio pruning) so
the same orchestrator produces both the baseline and the proposed system. This
is what makes the comparison in the proposal an ablation rather than two
different programs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from paim.common import metrics as M
from paim.common.session_state import SessionState
from paim.common.types import (
    ContextBlock,
    PipelineTrace,
    Session,
    SessionTurn,
    StageTrace,
    SubGoal,
    Tool,
    TurnTrace,
)
from paim.o1_sasr.cooccurrence import CooccurrenceGraph
from paim.o1_sasr.reranker import RerankWeights, SessionAwareReranker, StatelessReranker
from paim.o1_sasr.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from paim.o2_aprr.agents import AgentPool
from paim.o2_aprr.regret import RegretTracker
from paim.o2_aprr.router import APRRRouter, RoutingConfig, StaticHeavyRouter, is_high_stakes
from paim.o3_mncd.consensus import (
    ConsensusConfig,
    EvidenceWeightedConsensus,
    MajorityVoteConsensus,
    consensus_correct,
)
from paim.o3_mncd.deliberation import DeliberationConfig, MeshDeliberation
from paim.o3_mncd.execution import SimulatedToolExecutor
from paim.o4_fcnp.controller import ControllerConfig, FidelityController, FixedBudgetController
from paim.o4_fcnp.pruner import FCNPPruner, FixedRatioPruner
from paim.pipeline import context as ctx
from paim.governance.authority import ACRS_STAGES, Disposition, end_to_end_reliability
from paim.governance.layer import GovernanceConfig, GovernanceLayer


@dataclass(frozen=True)
class Ablations:
    """Per-objective switch between the proposed mechanism and its SOTA analogue."""

    session_aware_rerank: bool = True   # O1
    risk_aware_routing: bool = True     # O2
    mesh_consensus: bool = True         # O3
    fidelity_control: bool = True       # O4
    # G1 authority-bounded execution. Off by default so every v1/v2 number in
    # experiments/RESULTS.md reproduces bit-for-bit; governed runs opt in.
    governance: bool = False

    @staticmethod
    def sota_baseline() -> "Ablations":
        return Ablations(False, False, False, False)

    @staticmethod
    def governed() -> "Ablations":
        return Ablations(True, True, True, True, True)

    def label(self) -> str:
        objectives = {k: v for k, v in asdict(self).items() if k != "governance"}
        suffix = " + G1 governance" if self.governance else ""
        if all(objectives.values()):
            return "proposed (all objectives enabled)" + suffix
        if not any(objectives.values()):
            return "SOTA baseline (all objectives ablated)" + suffix
        on = [k for k, v in objectives.items() if v]
        return "partial: " + ", ".join(on) + suffix

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    retrieve_k: int = 40
    select_k: int = 3
    seed: int = 20260905
    max_reroutes: int = 1
    retriever: str = "bm25"            # bm25 | dense | hybrid
    rerank_weights: RerankWeights = field(default_factory=RerankWeights)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    deliberation: DeliberationConfig = field(default_factory=DeliberationConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    ablations: Ablations = field(default_factory=Ablations)
    tool_failure_rate: float = 0.08
    carry_observations: bool = True
    governance_config: GovernanceConfig = field(default_factory=GovernanceConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieveK": self.retrieve_k,
            "selectK": self.select_k,
            "seed": self.seed,
            "maxReroutes": self.max_reroutes,
            "retriever": self.retriever,
            "rerankWeights": self.rerank_weights.to_dict(),
            "routing": self.routing.to_dict(),
            "deliberation": self.deliberation.to_dict(),
            "consensus": self.consensus.to_dict(),
            "controller": self.controller.to_dict(),
            "ablations": self.ablations.to_dict(),
            "toolFailureRate": self.tool_failure_rate,
            "governance": self.governance_config.to_dict() if self.ablations.governance else None,
            "label": self.ablations.label(),
        }


class IntegratedPipeline:
    """One orchestrator, four objectives, one trace."""

    def __init__(
        self,
        tools: Mapping[str, Tool],
        train_trajectories: Sequence[Sequence[str]] = (),
        config: PipelineConfig | None = None,
        pool: AgentPool | None = None,
        executor: Any = None,
    ) -> None:
        self.tools = dict(tools)
        self.config = config or PipelineConfig()
        cfg = self.config
        self.pool = pool or AgentPool()

        # --- O1 -----------------------------------------------------------
        lexical = BM25Retriever().fit(self.tools.values())
        if cfg.retriever in {"dense", "hybrid"}:
            dense = DenseRetriever().fit(self.tools.values())
            self.retriever: Any = (
                dense if (cfg.retriever == "dense" and dense.available)
                else HybridRetriever(lexical, dense)
            )
        else:
            self.retriever = lexical
        self.graph = CooccurrenceGraph().fit(train_trajectories)
        self.reranker = (
            SessionAwareReranker(self.graph, cfg.rerank_weights)
            if cfg.ablations.session_aware_rerank
            else StatelessReranker()
        )

        # --- O2 -----------------------------------------------------------
        self.router = (
            APRRRouter(self.pool, cfg.routing)
            if cfg.ablations.risk_aware_routing
            else StaticHeavyRouter(self.pool, cfg.routing)
        )
        self.regret = RegretTracker()

        # --- O3 -----------------------------------------------------------
        self.executor = executor or SimulatedToolExecutor(
            seed=cfg.seed, failure_rate=cfg.tool_failure_rate
        )
        # --- G1 -----------------------------------------------------------
        self.governance: GovernanceLayer | None = (
            GovernanceLayer(cfg.governance_config) if cfg.ablations.governance else None
        )
        if self.governance is not None:
            # Tool output is data: every observation crosses the guard before
            # any peer reads it.
            self.executor = self.governance.wrap_executor(self.executor)
        consensus = (
            EvidenceWeightedConsensus(cfg.consensus)
            if cfg.ablations.mesh_consensus
            else MajorityVoteConsensus(cfg.consensus)
        )
        self.mesh = MeshDeliberation(
            pool=self.pool,
            executor=self.executor,
            config=cfg.deliberation,
            consensus=consensus,
            seed=cfg.seed,
        )

        # --- O4 -----------------------------------------------------------
        self.controller: Any = (
            FidelityController(FCNPPruner(), cfg.controller)
            if cfg.ablations.fidelity_control
            else FixedBudgetController(
                FixedRatioPruner(keep_ratio=0.5), 0.5, cfg.controller.fidelity_floor
            )
        )

    # ------------------------------------------------------------------
    def run(
        self,
        session: Session,
        arguments_for: Any = None,
        reset_controller: bool = True,
    ) -> PipelineTrace:
        """Execute a whole session and return its full trace.

        ``arguments_for(sub_goal_id) -> {doc_id: kwargs}`` supplies concrete
        call arguments; it is how the live data.gov.in scenarios inject their
        commodity/state filters. When omitted, required parameters are filled
        with their schema defaults, which is what the ToolBench simulation does.
        """
        cfg = self.config
        state = SessionState()
        if reset_controller:
            self.controller.reset()

        trace = PipelineTrace(
            session_id=session.session_id,
            query=session.query,
            profile={
                "email": session.profile.email,
                "sector": session.profile.sector,
                "interests": list(session.profile.interests),
                "riskTolerance": session.profile.risk_tolerance,
                "tokenBudget": session.profile.token_budget,
            },
            config=cfg.to_dict(),
        )
        carried: list[ContextBlock] = []
        per_turn_records: list[dict[str, Any]] = []

        for sub_goal in session.sub_goals:
            gov = self.governance
            if gov is not None and gov.kill_switch_engaged():
                turn_trace, record = self._unexecuted_turn(sub_goal, gov.blocked("kill switch engaged"))
            else:
                try:
                    turn_trace, carried, record = self._run_turn(
                        session, sub_goal, state, carried, arguments_for
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed is the point
                    if gov is None:
                        raise
                    turn_trace, record = self._unexecuted_turn(
                        sub_goal, gov.failed(f"{type(exc).__name__}: {exc}")
                    )
            trace.turns.append(turn_trace)
            session.turns.append(
                SessionTurn(
                    turn=sub_goal.turn,
                    sub_goal=sub_goal,
                    selected_doc_ids=tuple(record["selected"]),
                    success=turn_trace.success,
                )
            )
            per_turn_records.append(record)

        trace.metrics = self._session_metrics(session, per_turn_records, state)
        self.last_records = per_turn_records
        trace.feedback = self._feedback_summary(per_turn_records)
        trace.answer = self._compose_answer(session, per_turn_records)
        return trace

    # ------------------------------------------------------------------
    def _run_turn(
        self,
        session: Session,
        sub_goal: SubGoal,
        state: SessionState,
        carried: list[ContextBlock],
        arguments_for: Any,
    ) -> tuple[TurnTrace, list[ContextBlock], dict[str, Any]]:
        cfg = self.config
        profile = session.profile
        turn_trace = TurnTrace(turn=sub_goal.turn, sub_goal=sub_goal.text)

        # ---- S0: session state ------------------------------------------
        t0 = time.perf_counter()
        state.advance_turn()
        state.observe_intent(sub_goal.text)
        turn_trace.stages.append(
            StageTrace(
                stage="S0",
                objective="shared",
                label="Session state update",
                inputs={"subGoal": sub_goal.text},
                outputs={"state": state.snapshot()},
                metrics={"turn": float(state.turn), "risk": state.risk},
                explanation=[
                    "Intent written into the exponentially decayed session memory "
                    f"(decay={state.decay}); this vector is read by O1's session term, "
                    "O2's risk features and O4's memory block."
                ],
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        # ---- S1: O1 retrieval + session-aware rerank --------------------
        t0 = time.perf_counter()
        candidates = self.retriever.search(sub_goal.text, cfg.retrieve_k)
        rerank = self.reranker.rerank(
            sub_goal_text=sub_goal.text,
            candidates=candidates,
            tools=self.tools,
            state=state,
            profile=profile,
            committed=state.selected_doc_ids,
            top_k=cfg.select_k,
        )
        selected_tools = [self.tools[d] for d in rerank.selected]
        first_stage_ids = rerank.first_stage
        stage1_metrics = {
            "candidates": float(rerank.n_candidates),
            "selected": float(len(rerank.selected)),
        }
        if sub_goal.gold_doc_ids:
            gold = sub_goal.gold_doc_ids
            stage1_metrics.update(
                {
                    "ndcg@5": M.ndcg_at_k(rerank.ranked_ids, gold, 5),
                    "recall@5": M.true_recall_at_k(rerank.ranked_ids, gold, 5),
                    "mrr": M.mrr(rerank.ranked_ids, gold),
                    "firstStageNdcg@5": M.ndcg_at_k(first_stage_ids, gold, 5),
                    "firstStageRecall@5": M.true_recall_at_k(first_stage_ids, gold, 5),
                }
            )
        turn_trace.stages.append(
            StageTrace(
                stage="S1",
                objective="O1",
                label=f"SASR retrieve + rerank ({self.reranker.name})",
                inputs={"retriever": getattr(self.retriever, "name", "bm25"), "k": cfg.retrieve_k},
                outputs={
                    "selected": [
                        {"docId": t.doc_id, "tool": t.key, "category": t.category_name}
                        for t in selected_tools
                    ],
                    "firstStageTop5": first_stage_ids[:5],
                    "rerankedTop5": rerank.ranked_ids[:5],
                    "goldDocIds": list(sub_goal.gold_doc_ids),
                    "weights": rerank.weights,
                },
                metrics=stage1_metrics,
                explanation=rerank.explanations(3)
                + [f"co-occurrence graph: {self.graph.stats()}"],
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        # ---- S2: O4 context assembly + controlled pruning ---------------
        t0 = time.perf_counter()
        blocks = ctx.assemble(
            session=session,
            sub_goal=sub_goal,
            candidate_tools=selected_tools,
            state=state,
            carried_observations=carried if cfg.carry_observations else (),
        )
        prune, control = self.controller.step(
            blocks=blocks,
            goal_text=sub_goal.text,
            committed_tools=selected_tools,
            turn=sub_goal.turn,
            ceiling=profile.token_budget,
        )
        broken = set(prune.fidelity.broken_tools)
        callable_tools = [t for t in selected_tools if t.key not in broken]
        if not callable_tools and selected_tools:
            # The controller will raise the budget next turn; for this turn we
            # keep the top-ranked tool rather than emitting an empty call set.
            callable_tools = selected_tools[:1]
        turn_trace.stages.append(
            StageTrace(
                stage="S2",
                objective="O4",
                label=f"FCNP context governor ({getattr(self.controller, 'name', 'controller')})",
                inputs={
                    "blocks": len(blocks),
                    "originalTokens": prune.original_tokens,
                    "budget": prune.budget,
                    "tokenCeiling": profile.token_budget,
                },
                outputs={
                    "prune": prune.to_dict(),
                    "control": control.to_dict(),
                    "callableTools": [t.key for t in callable_tools],
                    "brokenTools": sorted(broken),
                },
                metrics={
                    "fidelity": prune.fidelity.fidelity,
                    "compression": prune.compression,
                    "keptTokens": float(prune.kept_tokens),
                    "schemaIntegrity": prune.fidelity.schema_integrity,
                    "floorSatisfied": 1.0
                    if prune.fidelity.fidelity >= self.config.controller.fidelity_floor
                    else 0.0,
                },
                explanation=[
                    f"budget {prune.budget} tok = {control.keep_fraction_target:.2f} x "
                    f"{prune.original_tokens} raw tok; fidelity {prune.fidelity.fidelity:.3f} "
                    f"vs floor {self.config.controller.fidelity_floor:.2f} -> "
                    f"controller action '{control.action}'",
                    f"schema integrity {prune.fidelity.schema_integrity:.2f}"
                    + (f", broken: {sorted(broken)}" if broken else ", all committed tools callable"),
                ],
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        # ---- S3: O2 routing ---------------------------------------------
        t0 = time.perf_counter()
        decision = self.router.route(sub_goal.text, callable_tools, state, profile)
        self.regret.record(
            self.pool[decision.agent_id], self.pool.routable(), sub_goal.text, callable_tools
        )
        turn_trace.stages.append(
            StageTrace(
                stage="S3",
                objective="O2",
                label=f"APRR routing ({self.router.name})",
                inputs={
                    "callableTools": [t.key for t in callable_tools],
                    "sessionRisk": state.risk,
                    "riskTolerance": profile.risk_tolerance,
                },
                outputs={
                    "decision": decision.to_dict(),
                    "agentName": self.pool[decision.agent_id].name,
                    "tier": self.pool[decision.agent_id].tier,
                },
                metrics={
                    "riskPosture": decision.risk_posture,
                    "escalated": 1.0 if decision.escalated else 0.0,
                    "topScore": decision.scores[0].score if decision.scores else 0.0,
                },
                explanation=[
                    f"posture {decision.risk_posture:+.3f} "
                    f"({'exploring' if decision.risk_posture > 0 else 'risk-averse'}); "
                    f"chose {self.pool[decision.agent_id].name}"
                    + (f"; {decision.escalation_reason}" if decision.escalation_reason else "")
                ],
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        # ---- S4: O3 mesh deliberation (with re-route on abstention) -----
        t0 = time.perf_counter()
        arguments = arguments_for(sub_goal.sub_goal_id) if callable(arguments_for) else None
        arguments = arguments or _default_arguments(callable_tools)
        delib = self.mesh.deliberate(
            sub_goal_id=sub_goal.sub_goal_id,
            sub_goal_text=sub_goal.text,
            tools=callable_tools,
            primary_id=decision.agent_id,
            state=state,
            high_stakes=decision.high_stakes,
            arguments=arguments,
        )
        reroutes = 0
        reroute_notes: list[str] = []
        turn_cost = delib.total_cost
        ceiling = (
            cfg.governance_config.max_cost_per_turn if self.governance is not None else float("inf")
        )
        # Bounded loop: step budget (max_reroutes) AND cost ceiling.
        while delib.certificate.abstained and reroutes < cfg.max_reroutes and turn_cost < ceiling:
            reroutes += 1
            # Feedback edge O3 -> O2: abstention raises risk, which makes the
            # router's escalation policy fire on the retry.
            state.risk = min(1.0, state.risk + 0.25)
            retry_decision = self.router.route(sub_goal.text, callable_tools, state, profile)
            reroute_notes.append(
                f"abstained ({delib.certificate.reason}); risk raised to {state.risk:.2f}, "
                f"re-routed to {self.pool[retry_decision.agent_id].name}"
            )
            decision = retry_decision
            delib = self.mesh.deliberate(
                sub_goal_id=sub_goal.sub_goal_id,
                sub_goal_text=sub_goal.text,
                tools=callable_tools,
                primary_id=decision.agent_id,
                state=state,
                high_stakes=True,
                arguments=arguments,
            )
            turn_cost += delib.total_cost

        cert = delib.certificate
        turn_trace.stages.append(
            StageTrace(
                stage="S4",
                objective="O3",
                label=f"MNCD deliberation ({self.mesh.consensus.name})",
                inputs={
                    "primary": decision.agent_id,
                    "highStakes": decision.high_stakes,
                    "arguments": {k: v for k, v in (arguments or {}).items()},
                },
                outputs=delib.to_dict(include_ground_truth=True),
                metrics={
                    "certified": 1.0 if cert.certified else 0.0,
                    "abstained": 1.0 if cert.abstained else 0.0,
                    "weightedShare": cert.weighted_share,
                    "quorumObserved": float(cert.quorum_observed),
                    "quorumRequired": float(cert.quorum_required),
                    "peers": float(len(delib.peers)),
                    "toolCalls": float(delib.n_tool_calls),
                    "toolFailures": float(delib.n_tool_failures),
                    "cost": delib.total_cost,
                    "wallLatencyMs": delib.wall_latency_ms,
                    "reroutes": float(reroutes),
                    "consensusCorrect": 1.0 if consensus_correct(cert, delib.reports) else 0.0,
                },
                explanation=[
                    f"peers {list(delib.peers)}; quorum {cert.quorum_observed}/"
                    f"{cert.quorum_required} (f={cert.tolerated_faults}); "
                    f"weighted share {cert.weighted_share:.2f}; {cert.reason}"
                ]
                + reroute_notes,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        # ---- G1: authority-bounded execution -----------------------------
        hijacked_keys = {r.claim_key for r in delib.reports if getattr(r, "hijacked", False)}
        if self.governance is not None:
            t0 = time.perf_counter()
            adj = self.governance.adjudicate(
                certificate=cert,
                reports=delib.reports,
                tools=callable_tools,
                risk_class=getattr(sub_goal, "risk_class", "informational"),
                stage_failures=0,
            )
            disposition = adj.disposition
            turn_trace.stages.append(
                StageTrace(
                    stage="G1",
                    objective="governance",
                    label="Policy-as-code adjudication (check_policy)",
                    inputs={"riskClass": getattr(sub_goal, "risk_class", "informational")},
                    outputs=adj.to_dict(),
                    metrics={
                        "execute": 1.0 if disposition is Disposition.EXECUTE else 0.0,
                        "review": 1.0 if disposition is Disposition.REVIEW else 0.0,
                        "theta": self.governance.theta,
                    },
                    explanation=[f"{disposition.value}: {adj.reason}"],
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            )
        else:
            disposition = Disposition.EXECUTE if cert.certified else Disposition.REVIEW
        executed = disposition is Disposition.EXECUTE
        correct = consensus_correct(cert, delib.reports)

        # ---- S5: credit assignment / feedback ---------------------------
        t0 = time.perf_counter()
        reward = _reward(cert.certified, delib.total_cost, delib.wall_latency_ms, cert.weighted_share)
        self.router.update(decision.agent_id, decision.context, reward)

        called_docs: list[str] = []
        for report in delib.reports:
            for inv in report.invocations:
                called_docs.append(inv.doc_id)
                tool = self.tools.get(inv.doc_id) or next(
                    (t for t in callable_tools if t.doc_id == inv.doc_id), None
                )
                if tool is None:
                    continue
                if inv.ok:
                    state.observe_selection(inv.doc_id, tool.key, tool.category_name)
                else:
                    state.observe_failure(tool.key)

        if cert.certified:
            state.observe_success()
            state.observe_satisfaction(sub_goal.text)
            sub_goal.status = "satisfied"
            sub_goal.satisfied_by = tuple(sorted(set(called_docs)))
        else:
            sub_goal.status = "failed"

        state.tokens_spent += prune.kept_tokens
        state.cost_spent += delib.total_cost

        surviving = ctx.carry_forward(prune.kept)
        writable = [inv for r in delib.reports for inv in r.invocations if inv.ok]
        if self.governance is not None:
            # R-MEM-01: quarantined observations never reach the next turn.
            writable = [inv for inv in writable if self.governance.allow_memory(inv)]
        new_observations = [
            ctx.observation_block(inv, sub_goal.turn, i) for i, inv in enumerate(writable)
        ]
        carried = (surviving + new_observations)[-8:] if cfg.carry_observations else []

        turn_trace.success = cert.certified
        turn_trace.answer = cert.claim if cert.certified else f"abstained: {cert.reason}"
        turn_trace.stages.append(
            StageTrace(
                stage="S5",
                objective="shared",
                label="Credit assignment (closes the loop)",
                inputs={"certified": cert.certified, "cost": round(delib.total_cost, 6)},
                outputs={
                    "reward": round(reward, 5),
                    "state": state.snapshot(),
                    "carriedObservations": len(carried),
                    "routerStats": self.router.stats(),
                },
                metrics={"reward": reward, "sessionRisk": state.risk},
                explanation=[
                    f"O3 -> O2: certificate turned into bandit reward {reward:.3f} for "
                    f"{self.pool[decision.agent_id].name}",
                    f"O3 -> O1: {sum(1 for r in delib.reports for i in r.invocations if not i.ok)} "
                    f"failed call(s) recorded as endpoint penalties; session risk now {state.risk:.2f}",
                    f"O4 -> O1: {len(carried)} observation block(s) carried into the next turn's "
                    "context and session memory",
                ],
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        record = {
            "turn": sub_goal.turn,
            "subGoalId": sub_goal.sub_goal_id,
            "selected": rerank.selected,
            "rankedIds": rerank.ranked_ids,
            "firstStageIds": first_stage_ids,
            "gold": list(sub_goal.gold_doc_ids),
            "certified": cert.certified,
            "abstained": cert.abstained,
            "consensusCorrect": consensus_correct(cert, delib.reports),
            "escalated": decision.escalated,
            "reroutes": reroutes,
            "agent": decision.agent_id,
            "tier": self.pool[decision.agent_id].tier,
            "cost": delib.total_cost,
            "latencyMs": delib.wall_latency_ms,
            "toolCalls": delib.n_tool_calls,
            "toolFailures": delib.n_tool_failures,
            "keptTokens": prune.kept_tokens,
            "originalTokens": prune.original_tokens,
            "fidelity": prune.fidelity.fidelity,
            "schemaIntegrity": prune.fidelity.schema_integrity,
            "hallucinatingPeers": sum(1 for r in delib.reports if r.hallucinated),
            "hijackedPeers": sum(1 for r in delib.reports if getattr(r, "hijacked", False)),
            "quarantinedCalls": sum(
                1 for r in delib.reports for i in r.invocations if getattr(i, "quarantined", False)
            ),
            "weightedShare": cert.weighted_share,
            "disposition": disposition.value,
            "executed": executed,
            "executedCorrect": executed and correct,
            "executedWrong": executed and not correct,
            "attackSucceeded": executed and cert.claim_key in hijacked_keys,
            "riskClass": getattr(sub_goal, "risk_class", "informational"),
            "failed": False,
        }
        return turn_trace, carried, record

    def _unexecuted_turn(self, sub_goal: SubGoal, adj: Any) -> tuple[TurnTrace, dict[str, Any]]:
        """A turn that never reached certification (kill switch or a stage failure)."""
        tt = TurnTrace(turn=sub_goal.turn, sub_goal=sub_goal.text)
        tt.stages.append(
            StageTrace(
                stage="G1", objective="governance", label="Fail-closed disposition",
                inputs={}, outputs=adj.to_dict(), metrics={"execute": 0.0},
                explanation=[f"{adj.disposition.value}: {adj.reason}"], duration_ms=0.0,
            )
        )
        tt.success = False
        tt.answer = f"{adj.disposition.value.lower()}: {adj.reason}"
        sub_goal.status = "failed"
        record = {
            "turn": sub_goal.turn, "subGoalId": sub_goal.sub_goal_id, "selected": [],
            "rankedIds": [], "firstStageIds": [], "gold": list(sub_goal.gold_doc_ids),
            "certified": False, "abstained": True, "consensusCorrect": False,
            "escalated": False, "reroutes": 0, "agent": "", "tier": "", "cost": 0.0,
            "latencyMs": 0.0, "toolCalls": 0, "toolFailures": 0, "keptTokens": 0,
            "originalTokens": 0, "fidelity": 0.0, "schemaIntegrity": 0.0,
            "hallucinatingPeers": 0, "hijackedPeers": 0, "quarantinedCalls": 0,
            "weightedShare": 0.0, "disposition": adj.disposition.value, "executed": False,
            "executedCorrect": False, "executedWrong": False, "attackSucceeded": False,
            "riskClass": getattr(sub_goal, "risk_class", "informational"), "failed": True,
        }
        return tt, record

    # ------------------------------------------------------------------
    def _session_metrics(
        self, session: Session, records: Sequence[dict[str, Any]], state: SessionState
    ) -> dict[str, float]:
        if not records:
            return {}
        gold_turns = [r for r in records if r["gold"]]
        out: dict[str, float] = {
            "turns": float(len(records)),
            "certifiedTurns": float(sum(1 for r in records if r["certified"])),
            "sessionSuccessRate": M.mean(1.0 if r["certified"] else 0.0 for r in records),
            "abstentionRate": M.mean(1.0 if r["abstained"] else 0.0 for r in records),
            "consensusAccuracy": M.mean(
                1.0 if r["consensusCorrect"] else 0.0 for r in records
            ),
            "escalationRate": M.mean(1.0 if r["escalated"] else 0.0 for r in records),
            "rerouteRate": M.mean(float(r["reroutes"]) for r in records),
            "totalCost": sum(r["cost"] for r in records),
            "totalWallLatencyMs": sum(r["latencyMs"] for r in records),
            "toolCalls": float(sum(r["toolCalls"] for r in records)),
            "toolFailureRate": (
                sum(r["toolFailures"] for r in records) / sum(r["toolCalls"] for r in records)
                if sum(r["toolCalls"] for r in records)
                else 0.0
            ),
            "keptTokens": float(sum(r["keptTokens"] for r in records)),
            "originalTokens": float(sum(r["originalTokens"] for r in records)),
            "meanFidelity": M.mean(r["fidelity"] for r in records),
            "minFidelity": min(r["fidelity"] for r in records),
            "meanSchemaIntegrity": M.mean(r["schemaIntegrity"] for r in records),
            "hallucinatingPeerEncounters": float(
                sum(r["hallucinatingPeers"] for r in records)
            ),
            "sessionRiskFinal": state.risk,
        }
        orig = out["originalTokens"]
        out["compression"] = 1.0 - out["keptTokens"] / orig if orig else 0.0
        out["floorSatisfactionRate"] = M.mean(
            1.0 if r["fidelity"] >= self.config.controller.fidelity_floor else 0.0
            for r in records
        )

        if gold_turns:
            out.update(
                {
                    "ndcg@5": M.mean(M.ndcg_at_k(r["rankedIds"], r["gold"], 5) for r in gold_turns),
                    "recall@5": M.mean(
                        M.true_recall_at_k(r["rankedIds"], r["gold"], 5) for r in gold_turns
                    ),
                    "mrr": M.mean(M.mrr(r["rankedIds"], r["gold"]) for r in gold_turns),
                    "firstStageNdcg@5": M.mean(
                        M.ndcg_at_k(r["firstStageIds"], r["gold"], 5) for r in gold_turns
                    ),
                    "firstStageRecall@5": M.mean(
                        M.true_recall_at_k(r["firstStageIds"], r["gold"], 5) for r in gold_turns
                    ),
                    "selectionF1": M.mean(M.set_f1(r["selected"], r["gold"]) for r in gold_turns),
                }
            )
            session_gold = session.gold_doc_ids
            session_selected = [d for r in records for d in r["selected"]]
            out["sessionToolSetF1"] = M.set_f1(session_selected, session_gold)

        out.update({f"regret_{k}": v for k, v in self.regret.summary().items()})
        n = float(len(records))
        out.update(
            {
                "executeRate": sum(1 for r in records if r["disposition"] == "EXECUTE") / n,
                "reviewRate": sum(1 for r in records if r["disposition"] == "REVIEW") / n,
                "blockRate": sum(1 for r in records if r["disposition"] == "BLOCK") / n,
                "executedErrorRate": sum(1 for r in records if r["executedWrong"]) / n,
                "executedCorrectRate": sum(1 for r in records if r["executedCorrect"]) / n,
                "attackSuccessRate": sum(1 for r in records if r["attackSucceeded"]) / n,
                "hijackedPeerEncounters": float(sum(r["hijackedPeers"] for r in records)),
                "quarantinedCalls": float(sum(r["quarantinedCalls"] for r in records)),
                "stageFailureRate": sum(1 for r in records if r["failed"]) / n,
            }
        )
        return out

    def _feedback_summary(self, records: Sequence[dict[str, Any]]) -> list[str]:
        if not records:
            return []
        certified = sum(1 for r in records if r["certified"])
        reroutes = sum(r["reroutes"] for r in records)
        failures = sum(r["toolFailures"] for r in records)
        halluc = sum(r["hallucinatingPeers"] for r in records)
        return [
            f"O3 -> O2: {len(records)} bandit update(s) from consensus certificates; "
            f"{certified}/{len(records)} certified.",
            f"O3 -> O2 (control): {reroutes} abstention-triggered re-route(s) with raised risk.",
            f"O3 -> O1: {failures} failed tool call(s) written back as endpoint penalties.",
            f"O4 -> O1: pruned observations carried forward; overall compression "
            f"{1 - sum(r['keptTokens'] for r in records) / max(1, sum(r['originalTokens'] for r in records)):.1%}.",
            f"Robustness: {halluc} confidently-wrong peer report(s) encountered and screened by "
            "evidence-weighted quorum.",
        ]

    def _compose_answer(self, session: Session, records: Sequence[dict[str, Any]]) -> str:
        lines = []
        for sg, r in zip(session.sub_goals, records):
            status = "certified" if r["certified"] else "abstained"
            tools = ", ".join(self.tools[d].key for d in r["selected"] if d in self.tools)
            lines.append(f"[turn {sg.turn}] {status} - {sg.text[:70]} via {tools or 'no tool'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def controller_summary(self) -> dict[str, Any]:
        return self.controller.summary()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "tools": len(self.tools),
            "retriever": getattr(self.retriever, "name", "bm25"),
            "cooccurrenceGraph": self.graph.stats(),
            "mesh": self.mesh.mesh.stats(),
            "agents": self.pool.to_dicts(),
            "router": self.router.stats(),
            "regret": self.regret.summary(),
            "controller": self.controller.summary(),
            "executor": getattr(self.executor, "stats", lambda: {"executor": self.executor.name})(),
            "config": self.config.to_dict(),
        }


def _default_arguments(tools: Sequence[Tool]) -> dict[str, dict[str, Any]]:
    """Fill required parameters from schema defaults (ToolBench simulation)."""
    out: dict[str, dict[str, Any]] = {}
    for tool in tools:
        args: dict[str, Any] = {}
        for p in tool.required_parameters:
            args[p.name] = p.default or f"<{p.name}>"
        out[tool.doc_id] = args
    return out


def _reward(certified: bool, cost: float, latency_ms: float, share: float) -> float:
    """Bandit reward from the consensus certificate, cost and latency."""
    base = (0.75 + 0.25 * share) if certified else 0.10 * share
    penalty = 0.30 * min(1.0, cost / 0.04) + 0.12 * min(1.0, latency_ms / 3000.0)
    return max(0.0, min(1.0, base - penalty))


__all__ = ["Ablations", "IntegratedPipeline", "PipelineConfig"]
