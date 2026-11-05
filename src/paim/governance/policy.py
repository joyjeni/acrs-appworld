"""Policy-as-Code for ACRS: a deny-by-default decision engine outside the agents.

Why it lives outside the agents
-------------------------------
An agent that can be talked out of a rule (prompt injection, a confident
hallucination, a colluding peer) cannot also be the thing enforcing the rule.
The engine below is therefore plain code with no model in it. Every agent
action that changes state - committing an answer, executing a tool, writing
memory back into SATR's session prior, releasing a build - must obtain an
``allow`` from :meth:`PolicyEngine.check_policy` first.

Standard every rule meets (enforced by :meth:`Rule.validate`)
--------------------------------------------------------------
1. **Denies by default.** A rule lists what must *all* be true to allow; if no
   rule for the action allows, the answer is deny.
2. **Every condition must hold.** Conditions are a conjunction, never "any".
3. **Names who, what, on what, in what state** - the four inputs of
   ``check_policy(subject, action, resource, state)``.
4. **Traced to a source.** Each rule carries a ``source`` string (paper,
   requirement id or design decision) so an examiner can ask "why".
5. **Tested both ways.** Each rule carries at least one ``allow`` and one
   ``deny`` example, and :meth:`PolicyEngine.self_test` runs them.

Audit
-----
Every decision is appended to a hash-chained log (each entry commits to the
previous entry's SHA-256). :meth:`DecisionLog.verify` detects any edited,
dropped or reordered entry, which answers "who decided, and can you prove it
afterwards?" without trusting the process that wrote the log.

The policy file format is JSON so the package keeps its numpy-only footprint
(Kaggle "Internet: off" compatible). The same rules can be exported to OPA/Rego
by :func:`to_rego` for deployments that already run an OPA sidecar.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_OPS = {"==", "!=", ">=", "<=", ">", "<", "in", "not_in", "is_true", "is_false"}


def _lookup(path: str, env: Mapping[str, Any]) -> Any:
    """Resolve ``subject.role`` / ``resource.weighted_share`` style paths."""
    cur: Any = env
    for part in path.split("."):
        if isinstance(cur, Mapping):
            if part not in cur:
                return _MISSING
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return _MISSING
            cur = getattr(cur, part)
    return cur


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


_MISSING = _Missing()


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: Any = None

    def holds(self, env: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[bool, str]:
        left = _lookup(self.field, env)
        right = self.value
        if isinstance(right, str) and right.startswith("$"):
            right = params.get(right[1:], _MISSING)
        if left is _MISSING:
            # Fail closed: an input the rule needs but was not supplied is a deny.
            return False, f"{self.field} missing"
        if right is _MISSING:
            return False, f"parameter {self.value} missing"
        try:
            if self.op == "==":
                ok = left == right
            elif self.op == "!=":
                ok = left != right
            elif self.op == ">=":
                ok = float(left) >= float(right)
            elif self.op == "<=":
                ok = float(left) <= float(right)
            elif self.op == ">":
                ok = float(left) > float(right)
            elif self.op == "<":
                ok = float(left) < float(right)
            elif self.op == "in":
                ok = left in right
            elif self.op == "not_in":
                ok = left not in right
            elif self.op == "is_true":
                ok = left is True
            elif self.op == "is_false":
                ok = left is False
            else:  # pragma: no cover - rejected by validate()
                ok = False
        except (TypeError, ValueError):
            return False, f"{self.field} not comparable"
        return ok, f"{self.field} {self.op} {right!r}" + ("" if ok else f" (was {left!r})")

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    action: str
    description: str
    source: str
    conditions: tuple[Condition, ...]
    examples_allow: tuple[Mapping[str, Any], ...] = ()
    examples_deny: tuple[Mapping[str, Any], ...] = ()
    version: str = "1"

    def validate(self) -> None:
        if not self.conditions:
            raise ValueError(f"{self.rule_id}: a rule with no conditions would allow everything")
        for c in self.conditions:
            if c.op not in _OPS:
                raise ValueError(f"{self.rule_id}: unknown operator {c.op!r}")
            if c.field.split(".")[0] not in {"subject", "resource", "state"}:
                raise ValueError(f"{self.rule_id}: field {c.field!r} must name subject/resource/state")
        if not self.source.strip():
            raise ValueError(f"{self.rule_id}: every rule must be traced to a source")
        if not self.examples_allow or not self.examples_deny:
            raise ValueError(f"{self.rule_id}: every rule must be tested both ways")

    def evaluate(self, env: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[bool, list[str]]:
        notes = []
        allow = True
        for c in self.conditions:
            ok, note = c.holds(env, params)
            notes.append(("ok " if ok else "FAIL ") + note)
            allow = allow and ok
        return allow, notes

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Rule":
        r = Rule(
            rule_id=d["id"],
            action=d["action"],
            description=d.get("description", ""),
            source=d.get("source", ""),
            conditions=tuple(Condition(**c) for c in d["all"]),
            examples_allow=tuple(d.get("examples", {}).get("allow", ())),
            examples_deny=tuple(d.get("examples", {}).get("deny", ())),
            version=str(d.get("version", "1")),
        )
        r.validate()
        return r


@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    allow: bool
    action: str
    rule_id: str
    reason: str
    notes: tuple[str, ...]
    latency_us: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisionId": self.decision_id,
            "allow": self.allow,
            "action": self.action,
            "ruleId": self.rule_id,
            "reason": self.reason,
            "notes": list(self.notes),
            "latencyUs": round(self.latency_us, 1),
        }


class DecisionLog:
    """Append-only, SHA-256 hash-chained decision log."""

    GENESIS = "0" * 64

    def __init__(self, path: str | Path | None = None) -> None:
        self.entries: list[dict[str, Any]] = []
        self.path = Path(path) if path else None

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else self.GENESIS

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        body = {"seq": len(self.entries), "prev": self.head, **record}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        entry = {**body, "hash": digest}
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        return entry

    def verify(self) -> tuple[bool, int]:
        """Return (intact, index of first bad entry or -1)."""
        prev = self.GENESIS
        for i, e in enumerate(self.entries):
            body = {k: v for k, v in e.items() if k != "hash"}
            if body.get("prev") != prev or body.get("seq") != i:
                return False, i
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
            if digest != e["hash"]:
                return False, i
            prev = e["hash"]
        return True, -1

    def __len__(self) -> int:
        return len(self.entries)


class PolicyEngine:
    """Deny-by-default ``check_policy`` with a hash-chained decision log."""

    def __init__(
        self,
        rules: Iterable[Rule],
        params: Mapping[str, Any] | None = None,
        log: DecisionLog | None = None,
        policy_version: str = "",
    ) -> None:
        self.rules: dict[str, list[Rule]] = {}
        for r in rules:
            r.validate()
            self.rules.setdefault(r.action, []).append(r)
        self.params = dict(params or {})
        self.log = log or DecisionLog()
        self.policy_version = policy_version
        self._counter = 0

    @classmethod
    def from_file(cls, path: str | Path, **kw: Any) -> "PolicyEngine":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = [Rule.from_dict(d) for d in raw["rules"]]
        version = raw.get("version", "") + ":" + hashlib.sha256(
            Path(path).read_bytes()
        ).hexdigest()[:12]
        params = {**raw.get("params", {}), **kw.pop("params", {})}
        return cls(rules, params=params, policy_version=version, **kw)

    def check_policy(
        self,
        subject: Mapping[str, Any],
        action: str,
        resource: Mapping[str, Any],
        state: Mapping[str, Any],
        record: bool = True,
    ) -> PolicyDecision:
        t0 = time.perf_counter()
        env = {"subject": subject, "resource": resource, "state": state}
        candidates = self.rules.get(action, [])
        allow, rule_id, reason, notes = False, "default-deny", f"no rule allows '{action}'", []
        for rule in candidates:
            ok, n = rule.evaluate(env, self.params)
            notes.extend(f"[{rule.rule_id}] {x}" for x in n)
            if ok:
                allow, rule_id, reason = True, rule.rule_id, rule.description
                break
            rule_id, reason = rule.rule_id, "denied: " + "; ".join(x[5:] for x in n if x.startswith("FAIL"))
        self._counter += 1
        decision = PolicyDecision(
            decision_id=f"pd-{self._counter:07d}",
            allow=allow,
            action=action,
            rule_id=rule_id,
            reason=reason,
            notes=tuple(notes),
            latency_us=(time.perf_counter() - t0) * 1e6,
        )
        if record:
            self.log.append(
                {
                    "decisionId": decision.decision_id,
                    "policyVersion": self.policy_version,
                    "action": action,
                    "subject": dict(subject),
                    "resource": {k: v for k, v in resource.items() if _is_jsonable(v)},
                    "state": {k: v for k, v in state.items() if _is_jsonable(v)},
                    "allow": allow,
                    "ruleId": rule_id,
                    "reason": reason,
                }
            )
        return decision

    def self_test(self) -> list[dict[str, Any]]:
        """Run every rule's allow/deny examples. Returns failures (empty = pass)."""
        failures = []
        for action, rules in self.rules.items():
            for rule in rules:
                for expect, examples in ((True, rule.examples_allow), (False, rule.examples_deny)):
                    for ex in examples:
                        d = self.check_policy(
                            ex.get("subject", {}), action, ex.get("resource", {}),
                            ex.get("state", {}), record=False,
                        )
                        if d.allow != expect:
                            failures.append(
                                {"rule": rule.rule_id, "expected": expect, "got": d.allow,
                                 "example": ex, "reason": d.reason}
                            )
        return failures

    def stats(self) -> dict[str, Any]:
        allows = sum(1 for e in self.log.entries if e["allow"])
        return {
            "policyVersion": self.policy_version,
            "rules": sum(len(v) for v in self.rules.values()),
            "decisions": len(self.log),
            "allowed": allows,
            "denied": len(self.log) - allows,
            "logIntact": self.log.verify()[0],
        }


def _is_jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, tuple))


def to_rego(rules: Sequence[Rule], package: str = "acrs.authz") -> str:
    """Export rules to OPA Rego (default deny, one ``allow`` body per rule)."""
    out = [f"package {package}", "", "import rego.v1", "", "default allow := false", ""]
    for r in rules:
        out.append(f"# {r.rule_id}: {r.description}")
        out.append(f"# source: {r.source}")
        out.append("allow if {")
        out.append(f'\tinput.action == "{r.action}"')
        for c in r.conditions:
            lhs = "input." + c.field
            rhs = c.value
            if isinstance(rhs, str) and rhs.startswith("$"):
                rhs_s = "data.params." + rhs[1:]
            else:
                rhs_s = json.dumps(rhs)
            if c.op == "is_true":
                out.append(f"\t{lhs} == true")
            elif c.op == "is_false":
                out.append(f"\t{lhs} == false")
            elif c.op == "in":
                out.append(f"\t{lhs} in {rhs_s}")
            elif c.op == "not_in":
                out.append(f"\tnot {lhs} in {rhs_s}")
            else:
                out.append(f"\t{lhs} {c.op} {rhs_s}")
        out.append("}")
        out.append("")
    return "\n".join(out)


__all__ = ["Condition", "DecisionLog", "PolicyDecision", "PolicyEngine", "Rule", "to_rego"]
