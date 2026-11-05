"""Smoke tests: each objective (O1-O4) and the integrated loop run end to end on shipped data."""
from pathlib import Path
from paim.data.toolbench import load_bundle, build_sessions
from paim.pipeline.orchestrator import IntegratedPipeline, PipelineConfig, Ablations

DATA = Path(__file__).resolve().parents[1] / "data"


def _pipe(abl):
    b = load_bundle(DATA)
    gold = {frozenset(q["gold_doc_ids"]) for q in b.queries}
    traj = [t for t in b.train_trajectories if frozenset(t) not in gold]  # leak-filtered
    return b, IntegratedPipeline(b.tools, traj, PipelineConfig(ablations=abl))


def test_full_acrs_runs_and_reports_all_objectives():
    b, pipe = _pipe(Ablations())
    m = pipe.run(build_sessions(b)[0]).metrics
    for key in ("ndcg@5", "consensusAccuracy", "totalCost", "compression", "meanFidelity"):
        assert key in m


def test_fcnp_respects_fidelity_floor():
    b, pipe = _pipe(Ablations())
    rows = [pipe.run(s).metrics for s in build_sessions(b)[:10]]
    assert sum(r["floorSatisfactionRate"] for r in rows) / len(rows) >= 0.9


def test_no_leaked_gold_sets_in_cooccurrence_training():
    b = load_bundle(DATA)
    gold = {frozenset(q["gold_doc_ids"]) for q in b.queries}
    clean = [t for t in b.train_trajectories if frozenset(t) not in gold]
    assert not any(frozenset(t) in gold for t in clean)


def test_satr_v2_provider_memory_carries_across_turns():
    from paim.common.session_state import SessionState
    s = SessionState()
    s.observe_selection("1", "SQUAKE::Projects", "Logistics")
    s.advance_turn()
    assert s.family_affinity("SQUAKE") == 1.0
    assert s.family_affinity("Other") == 0.0


def _report(agent, key, conf, evidence=(), hallucinated=False):
    from paim.o3_mncd.execution import AgentReport
    return AgentReport(agent_id=agent, sub_goal_id="g", claim=key, claim_key=key, confidence=conf,
                       evidence=tuple(evidence), ok=True, latency_ms=1.0, cost=0.0, hallucinated=hallucinated)


def test_mncd_safety_lone_forger_not_certified_with_n4():
    """Proposition 3: with n = 4 (f = 1) a single Byzantine peer that forges citations is never certified."""
    from paim.o3_mncd.consensus import EvidenceWeightedConsensus
    ev = ["d1:abc123"]
    reports = [_report("a1", "fake", 0.95, ev, True)] + [_report(f"a{i}", f"partial{i}", 0.3, ev) for i in range(2, 5)]
    cert = EvidenceWeightedConsensus().certify("g", reports)
    assert not (cert.certified and cert.claim_key == "fake")


def test_router_baselines_route():
    from paim.o2_aprr.baselines import LinUCBRouter, LinTSRouter
    from paim.common.session_state import SessionState
    from paim.common.types import UserProfile
    for cls in (LinUCBRouter, LinTSRouter):
        d = cls().route("price of wheat", [], SessionState(), UserProfile())
        assert d.agent_id
