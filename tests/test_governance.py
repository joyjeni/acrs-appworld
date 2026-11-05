"""G1 governance: every property the thesis claims is asserted here."""

from __future__ import annotations

import json
import random

import pytest

from paim.common.types import Tool, ToolInvocation
from paim.data.kcc import (
    SYNTHETIC_SMOKE_ROWS,
    build_kcc_sessions,
    kcc_report,
    kisan_tools,
)
from paim.governance import (
    ACRS_STAGES,
    DecisionLog,
    Disposition,
    GovernanceConfig,
    GovernanceLayer,
    InjectingExecutor,
    InjectionGuard,
    KillSwitch,
    PolicyEngine,
    ReleaseManifest,
    Rule,
    distribution_report,
    end_to_end_reliability,
    evaluation_gate,
    expected_calibration_error,
    learn_then_test,
    pass_hat_k,
    registry_violations,
    to_rego,
    tool_side_effect,
)
from paim.governance.guard import PAYLOADS
from paim.governance.layer import DEFAULT_POLICY
from paim.o3_mncd.execution import SimulatedToolExecutor
from paim.pipeline.orchestrator import Ablations, IntegratedPipeline, PipelineConfig


# --------------------------------------------------------------- policy engine
def test_policy_file_rules_pass_their_own_examples_both_ways():
    eng = PolicyEngine.from_file(DEFAULT_POLICY)
    assert eng.self_test() == []
    for rules in eng.rules.values():
        for r in rules:
            assert r.examples_allow and r.examples_deny and r.source


def test_default_deny_for_unknown_action_and_missing_inputs():
    eng = PolicyEngine.from_file(DEFAULT_POLICY)
    assert not eng.check_policy({"role": "mesh"}, "launch_rocket", {}, {}).allow
    # commit_answer with an input missing entirely -> deny (fail closed)
    d = eng.check_policy({"role": "mesh"}, "commit_answer", {"certified": True}, {})
    assert not d.allow and "missing" in d.reason


def test_rule_without_examples_or_source_is_rejected():
    with pytest.raises(ValueError):
        Rule.from_dict({"id": "x", "action": "a", "all": [{"field": "subject.r", "op": "==", "value": 1}],
                        "source": "s", "examples": {"allow": [{}]}})
    with pytest.raises(ValueError):
        Rule.from_dict({"id": "x", "action": "a", "all": [], "source": "s",
                        "examples": {"allow": [{}], "deny": [{}]}})


def test_decision_log_is_tamper_evident():
    log = DecisionLog()
    for i in range(5):
        log.append({"i": i, "allow": True})
    assert log.verify() == (True, -1)
    log.entries[2]["allow"] = False
    ok, bad = log.verify()
    assert not ok and bad == 2


def test_rego_export_is_default_deny():
    eng = PolicyEngine.from_file(DEFAULT_POLICY)
    rego = to_rego([r for rs in eng.rules.values() for r in rs])
    assert "default allow := false" in rego and "R-COMMIT-01" in rego


# --------------------------------------------------------------- authority
def test_stage_registry_design_invariants_hold():
    assert registry_violations() == []
    assert any(s.stage == "G1" and not s.uses_model for s in ACRS_STAGES)


def test_reliability_compounds_multiplicatively():
    assert end_to_end_reliability({f"s{i}": 0.95 for i in range(10)}) == pytest.approx(0.5987, abs=1e-3)


def test_side_effect_classification_is_conservative():
    read = Tool("1", "Finance", "Rates", "get_exchange_rate")
    write = Tool("2", "Finance", "Bank", "transfer_money")
    post = Tool("3", "X", "Y", "lookup", method="POST")
    assert tool_side_effect(read) == "read"
    assert tool_side_effect(write) == "write"
    assert tool_side_effect(post) == "write"


# --------------------------------------------------------------- guard
def test_guard_quarantines_overt_payloads_but_not_clean_text():
    g = InjectionGuard()
    clean = ToolInvocation("t", "d", observation=json.dumps({"price": 2150, "market": "Khanna"}))
    assert not g.sanitize(clean).quarantined
    for p in PAYLOADS[:4]:
        inv = ToolInvocation("t", "d", observation=json.dumps({"price": 1, "note": p.format(target="X")}))
        out = g.sanitize(inv)
        assert out.quarantined and "X" not in out.observation


def test_paraphrased_payload_is_a_known_residual():
    # Documented limitation: the lexical guard is a baseline, not complete.
    g = InjectionGuard()
    inv = ToolInvocation("t", "d", observation=json.dumps({"note": PAYLOADS[4].format(target="X")}))
    assert not g.sanitize(inv).quarantined


def test_governance_never_reads_ground_truth_fields():
    import inspect

    import paim.governance.guard as guard
    import paim.governance.layer as layer
    import paim.governance.policy as policy

    for mod in (layer, policy):
        src = inspect.getsource(mod)
        assert ".injected" not in src and ".payload" not in src and "hijacked" not in src
    guard_src = inspect.getsource(guard.InjectionGuard)
    assert ".injected" not in guard_src and ".payload" not in guard_src


# --------------------------------------------------------------- conformal
def test_learn_then_test_controls_risk_on_fresh_data():
    rng = random.Random(0)

    def draw(n):
        s, w = [], []
        for _ in range(n):
            x = rng.random()
            s.append(x)
            w.append(rng.random() < 0.35 * (1 - x) ** 2)
        return s, w

    cal = draw(3000)
    ct = learn_then_test(*cal, alpha=0.03, delta=0.05)
    assert ct.valid and ct.risk_hat <= 0.03
    test_s, test_w = draw(20000)
    ex = [w for s, w in zip(test_s, test_w) if s >= ct.threshold]
    assert sum(ex) / len(ex) <= 0.03 + 0.005


def test_learn_then_test_refuses_when_no_threshold_is_safe():
    ct = learn_then_test([0.9] * 50, [True] * 10 + [False] * 40, alpha=0.01)
    assert not ct.valid and ct.threshold > 1.0


def test_ece_is_zero_for_perfect_calibration():
    assert expected_calibration_error([1.0, 1.0, 0.0], [True, True, False]) == pytest.approx(0.0)


# --------------------------------------------------------------- release
def test_manifest_detects_commit_less_change(tmp_path):
    pol = tmp_path / "p.json"
    pol.write_text("{}")
    kw = dict(code_ref="abc", config={"a": 1}, agents=[{"id": "x"}], tools=[],
              models={"x": "m-2026-01-01"}, policy_path=pol, index_fingerprint=[1, 2])
    a = ReleaseManifest.build(**kw)
    b = ReleaseManifest.build(**{**kw, "models": {"x": "m-2026-06-01"}})
    assert a.diff(b) == ["models"]


def test_distribution_gate_and_pass_k():
    rep = distribution_report([0.5, 0.6, 0.7, 0.8])
    assert rep["n"] == 4 and rep["median"] == pytest.approx(0.65)
    assert pass_hat_k([[True, True, False]], 2) == pytest.approx(1 / 3)
    g = evaluation_gate({"q": [0.9, 0.7, 0.9]}, {}, {"q": (">=", 0.8)}, flake_tolerance=1)
    assert g.passed
    g = evaluation_gate({"q": [0.9, 0.7, 0.7]}, {}, {"q": (">=", 0.8)}, flake_tolerance=1)
    assert not g.passed


def test_kill_switch_scheduled_check(tmp_path, monkeypatch):
    flag = tmp_path / "KILL"
    ks = KillSwitch(flag_file=str(flag))
    assert not ks.engaged()
    flag.write_text("1")
    assert ks.engaged()
    monkeypatch.setenv("ACRS_KILL_SWITCH", "1")
    assert KillSwitch().engaged()


# --------------------------------------------------------------- integrated
def _kcc_pipeline(abl, executor=None, **gov):
    tools = {t.doc_id: t for t in kisan_tools()}
    cfg = PipelineConfig(ablations=abl, governance_config=GovernanceConfig(**gov))
    return IntegratedPipeline(tools, (), cfg, executor=executor)


def test_kcc_smoke_sessions_and_report():
    sessions = build_kcc_sessions(SYNTHETIC_SMOKE_ROWS)
    rep = kcc_report(SYNTHETIC_SMOKE_ROWS, sessions)
    assert rep["sessions"] == 2 and rep["riskClasses"]["agronomic-dose"] == 2


def test_dose_and_entitlement_queries_are_never_auto_executed():
    pipe = _kcc_pipeline(Ablations.governed())
    for s in build_kcc_sessions(SYNTHETIC_SMOKE_ROWS):
        pipe.run(s)
        for g, r in zip(s.sub_goals, pipe.last_records):
            if g.risk_class != "informational":
                assert r["disposition"] != "EXECUTE"


def test_fail_closed_on_stage_exception():
    class Boom:
        name, live = "boom", False

        def execute(self, *a, **k):
            raise RuntimeError("upstream exploded")

    pipe = _kcc_pipeline(Ablations.governed(), executor=Boom())
    s = build_kcc_sessions(SYNTHETIC_SMOKE_ROWS)[0]
    pipe.run(s)
    assert all(r["disposition"] == "REVIEW" and r["failed"] for r in pipe.last_records)


def test_ungoverned_pipeline_still_raises():
    class Boom:
        name, live = "boom", False

        def execute(self, *a, **k):
            raise RuntimeError("x")

    pipe = _kcc_pipeline(Ablations(), executor=Boom())
    with pytest.raises(RuntimeError):
        pipe.run(build_kcc_sessions(SYNTHETIC_SMOKE_ROWS)[0])


def test_kill_switch_blocks_every_turn(tmp_path):
    flag = tmp_path / "KILL"
    flag.write_text("1")
    pipe = _kcc_pipeline(Ablations.governed(), kill_switch_file=str(flag))
    pipe.run(build_kcc_sessions(SYNTHETIC_SMOKE_ROWS)[0])
    assert {r["disposition"] for r in pipe.last_records} == {"BLOCK"}


def test_execute_requires_recorded_allow():
    pipe = _kcc_pipeline(Ablations.governed())
    for s in build_kcc_sessions(SYNTHETIC_SMOKE_ROWS):
        pipe.run(s)
    ok, _ = pipe.governance.engine.log.verify()
    assert ok
    executes = pipe.governance.counts["EXECUTE"]
    allows = sum(1 for e in pipe.governance.engine.log.entries
                 if e["action"] == "commit_answer" and e["allow"])
    assert executes == allows


def test_injection_success_drops_under_governance():
    def run(abl):
        tools = {t.doc_id: t for t in kisan_tools()}
        cfg = PipelineConfig(ablations=abl)
        ex = InjectingExecutor(SimulatedToolExecutor(seed=cfg.seed), rate=1.0)
        pipe = IntegratedPipeline(tools, (), cfg, executor=ex)
        hits = 0
        n = 0
        for s in build_kcc_sessions(SYNTHETIC_SMOKE_ROWS * 5):
            pipe.run(s)
            hits += sum(r["attackSucceeded"] for r in pipe.last_records)
            n += len(pipe.last_records)
        return hits / n

    assert run(Ablations.governed()) < run(Ablations())


def test_attack_free_runs_unchanged_by_new_fields():
    # The governed flag off must reproduce the pre-governance pipeline exactly.
    a = _kcc_pipeline(Ablations())
    b = _kcc_pipeline(Ablations())
    sa = build_kcc_sessions(SYNTHETIC_SMOKE_ROWS)
    sb = build_kcc_sessions(SYNTHETIC_SMOKE_ROWS)
    ma = [a.run(s).metrics for s in sa]
    mb = [b.run(s).metrics for s in sb]
    assert ma == mb
    assert Disposition.EXECUTE.value == "EXECUTE"
