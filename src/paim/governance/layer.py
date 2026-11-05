"""GovernanceLayer: the fifth stage (G1) that turns a certificate into a disposition.

The four objectives answer *what is the best answer and how cheaply can we get
it*. G1 answers *is the system allowed to act on it without a person*. It is
not a fifth algorithm competing with O1-O4; it is the authority contract that
binds them (gap G5) and the only place a disposition is produced:

    certificate (O3) + tool provenance (guard) + side effect (O1 schema)
      + risk class (domain) + stage health  --check_policy-->  EXECUTE | REVIEW | BLOCK

Properties tested in tests/test_governance.py:

* fail closed  - any exception in a turn yields REVIEW, never EXECUTE;
* no skipped check - :meth:`adjudicate` cannot return EXECUTE without a
  recorded ``allow`` decision id from the policy engine;
* kill switch  - engaged => BLOCK before any stage runs;
* bounded loop - re-routing stops at the step budget or the cost ceiling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from paim.common.types import Tool, ToolInvocation
from paim.governance.authority import Disposition
from paim.governance.guard import GuardedExecutor, InjectionGuard
from paim.governance.policy import PolicyDecision, PolicyEngine
from paim.governance.release import KillSwitch

DEFAULT_POLICY = Path(__file__).resolve().parents[3] / "policies" / "acrs_policies.json"

_WRITE_VERBS = re.compile(
    r"\b(post|put|patch|delete|remove|create|add|update|send|transfer|pay|payment|book|"
    r"order|purchase|cancel|submit|upload|register|subscribe|withdraw)\b",
    re.IGNORECASE,
)


def tool_side_effect(tool: Tool) -> str:
    """'write' if the endpoint can change external state, else 'read'.

    HTTP method first; ToolBench marks many writes as GET, so the API name is
    also checked against state-changing verbs. Conservative by construction:
    a false 'write' costs a review, a false 'read' could cost an action.
    """
    if (tool.method or "GET").upper() not in {"GET", "HEAD", "OPTIONS"}:
        return "write"
    name = f"{tool.api_name} {tool.tool_name}".replace("_", " ")
    return "write" if _WRITE_VERBS.search(name) else "read"


@dataclass(frozen=True)
class GovernanceConfig:
    policy_path: str = str(DEFAULT_POLICY)
    theta: float | None = None          # None -> policy file default; set from Learn-then-Test
    guard: bool = True
    max_cost_per_turn: float = 0.05      # cost ceiling for the abstain -> re-route loop
    kill_switch_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": Path(self.policy_path).name,
            "theta": self.theta,
            "guard": self.guard,
            "maxCostPerTurn": self.max_cost_per_turn,
        }


@dataclass
class Adjudication:
    disposition: Disposition
    decision: PolicyDecision | None
    reason: str
    resource: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "reason": self.reason,
            "decision": self.decision.to_dict() if self.decision else None,
            "resource": self.resource,
        }


class GovernanceLayer:
    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self.config = config or GovernanceConfig()
        # Rule examples document rule *semantics* at the file's default
        # parameters, so the self-test runs before any calibrated override.
        failures = PolicyEngine.from_file(self.config.policy_path).self_test()
        if failures:
            raise RuntimeError(f"policy self-test failed: {failures}")
        params = {"theta": self.config.theta} if self.config.theta is not None else {}
        self.engine = PolicyEngine.from_file(self.config.policy_path, params=params)
        self.guard = InjectionGuard() if self.config.guard else None
        self.kill = KillSwitch(flag_file=self.config.kill_switch_file)
        self.counts = {d.value: 0 for d in Disposition}

    # ----------------------------------------------------------------- hooks
    def wrap_executor(self, executor: Any) -> Any:
        return GuardedExecutor(executor, self.guard) if self.guard else executor

    @property
    def theta(self) -> float:
        return float(self.engine.params["theta"])

    def kill_switch_engaged(self) -> bool:
        return self.kill.engaged()

    def allow_memory(self, inv: ToolInvocation) -> bool:
        d = self.engine.check_policy(
            {"role": "pipeline"},
            "write_memory",
            {"ok": bool(inv.ok), "quarantined": bool(getattr(inv, "quarantined", False)),
             "tool": inv.tool_key},
            {"kill_switch": self.kill.engaged()},
        )
        return d.allow

    def adjudicate(
        self,
        *,
        certificate: Any,
        reports: Sequence[Any],
        tools: Sequence[Tool],
        risk_class: str = "informational",
        stage_failures: int = 0,
    ) -> Adjudication:
        from paim.o3_mncd.execution import evidence_token

        quarantined_tokens = {
            evidence_token(inv.doc_id, inv.observation)
            for r in reports
            for inv in r.invocations
            if getattr(inv, "quarantined", False)
        }
        cited = set(getattr(certificate, "evidence", ()) or ())
        side = "write" if any(tool_side_effect(t) == "write" for t in tools) else "read"
        resource = {
            "certified": bool(certificate.certified),
            "weighted_share": float(certificate.weighted_share),
            "quorum_ok": certificate.quorum_observed >= certificate.quorum_required,
            "quarantined_evidence": len(cited & quarantined_tokens),
            "side_effect": side,
            "risk_class": risk_class,
            "claim_key": certificate.claim_key,
        }
        state = {"stage_failures": int(stage_failures), "kill_switch": self.kill.engaged()}
        decision = self.engine.check_policy({"role": "mesh"}, "commit_answer", resource, state)
        if decision.allow:
            disp, why = Disposition.EXECUTE, decision.reason
        elif state["kill_switch"]:
            disp, why = Disposition.BLOCK, "kill switch engaged"
        else:
            # Recommend, do not decide: everything not cleared goes to a person.
            disp, why = Disposition.REVIEW, decision.reason
        # Invariant: EXECUTE only with a recorded allow.
        if disp is Disposition.EXECUTE and not (decision.allow and decision.decision_id):
            disp, why = Disposition.REVIEW, "no recorded policy decision"
        self.counts[disp.value] += 1
        return Adjudication(disp, decision, why, resource)

    def blocked(self, reason: str) -> Adjudication:
        self.counts[Disposition.BLOCK.value] += 1
        return Adjudication(Disposition.BLOCK, None, reason)

    def failed(self, reason: str) -> Adjudication:
        self.counts[Disposition.REVIEW.value] += 1
        return Adjudication(Disposition.REVIEW, None, f"fail-closed: {reason}")

    def stats(self) -> dict[str, Any]:
        return {
            "dispositions": dict(self.counts),
            "policy": self.engine.stats(),
            "guard": self.guard.stats() if self.guard else None,
            "theta": self.theta,
        }


__all__ = ["Adjudication", "GovernanceConfig", "GovernanceLayer", "tool_side_effect"]
