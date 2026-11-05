"""Release discipline for an agentic system: pin it, gate it, be able to stop it.

Four ways agentic failure hides from a conventional pipeline - silent (well
formed output while judgment degrades), irreversible, cross-tenant, and
commit-less (a model, index or prompt changes with no diff). The controls:

* :class:`ReleaseManifest` pins the *seven* parts of a release - code, config,
  prompts/agent specs, tool definitions, model identifiers, guardrail policy,
  retrieval index - by SHA-256, so the exact agent can be rebuilt and any
  commit-less change shows up as a manifest diff.
* :func:`distribution_report` treats a build as a distribution, not a score:
  median, low percentile and n over seeds, gated with a flake tolerance.
* :func:`evaluation_gate` blocks promotion on quality, cost and p95 latency
  regressions. It automates the *evidence* for a release; the decision to
  promote stays with a named approver (policy R-REL-01).
* :class:`KillSwitch` removes the system's ability to act without a deploy.
  Engaged -> every turn is BLOCK. It is checked before each turn and is tested
  on a schedule (tests/test_governance.py), because an untested switch is an
  unverified one.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _sha(obj: Any) -> str:
    if isinstance(obj, (bytes, bytearray)):
        data = bytes(obj)
    elif isinstance(obj, Path):
        data = obj.read_bytes() if obj.is_file() else str(obj).encode()
    else:
        data = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


@dataclass
class ReleaseManifest:
    code: str
    config: str
    agents: str
    tools: str
    models: str
    policy: str
    index: str
    runtime: dict[str, str] = field(default_factory=dict)

    PARTS = ("code", "config", "agents", "tools", "models", "policy", "index")

    @classmethod
    def build(
        cls,
        *,
        code_ref: str,
        config: Mapping[str, Any],
        agents: Sequence[Mapping[str, Any]],
        tools: Iterable[Any],
        models: Mapping[str, str],
        policy_path: str | Path,
        index_fingerprint: Any,
    ) -> "ReleaseManifest":
        tool_rows = sorted(
            (getattr(t, "doc_id", str(t)), getattr(t, "document_text", lambda: str(t))())
            for t in tools
        )
        try:
            import numpy  # type: ignore

            np_v = numpy.__version__
        except Exception:  # pragma: no cover
            np_v = "absent"
        return cls(
            code=code_ref,
            config=_sha(config),
            agents=_sha(list(agents)),
            tools=_sha(tool_rows),
            models=_sha(dict(models)),
            policy=_sha(Path(policy_path)),
            index=_sha(index_fingerprint),
            runtime={"python": platform.python_version(), "numpy": np_v},
        )

    def digest(self) -> str:
        return _sha({p: getattr(self, p) for p in self.PARTS})

    def diff(self, other: "ReleaseManifest") -> list[str]:
        return [p for p in self.PARTS if getattr(self, p) != getattr(other, p)]

    def to_dict(self) -> dict[str, Any]:
        return {**{p: getattr(self, p) for p in self.PARTS}, "runtime": self.runtime, "digest": self.digest()}


def _pct(xs: Sequence[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(len(xs) - 1, int(i) + 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def distribution_report(values: Sequence[float]) -> dict[str, float]:
    values = [float(v) for v in values if v == v and v not in (float("inf"), float("-inf"))]
    return {
        "n": float(len(values)),
        "median": _pct(values, 0.5),
        "p10": _pct(values, 0.10),
        "p90": _pct(values, 0.90),
        "mean": st.mean(values) if values else float("nan"),
        "sd": st.pstdev(values) if len(values) > 1 else 0.0,
    }


def pass_hat_k(outcomes: Sequence[Sequence[bool]], k: int) -> float:
    """tau-bench pass^k: probability that k i.i.d. trials of a task ALL succeed.

    ``outcomes[task]`` is the list of n trial outcomes for one task. Unbiased
    estimator C(c, k) / C(n, k) with c successes (Yao et al., 2024).
    """
    from math import comb

    vals = []
    for trials in outcomes:
        n, c = len(trials), sum(1 for t in trials if t)
        if n < k:
            continue
        vals.append(comb(c, k) / comb(n, k))
    return st.mean(vals) if vals else float("nan")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons)}


def evaluation_gate(
    candidate: Mapping[str, Sequence[float]],
    baseline: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, tuple[str, float]],
    flake_tolerance: int = 0,
    max_cost_ratio: float = 1.5,
    max_p95_latency_ratio: float = 1.5,
) -> GateResult:
    """Block on quality floors (per seed, with flake tolerance), cost and p95 latency."""
    reasons = []
    for metric, (direction, bound) in thresholds.items():
        vals = list(candidate.get(metric, ()))
        if not vals:
            reasons.append(f"{metric}: no samples")
            continue
        bad = sum(1 for v in vals if (v < bound if direction == ">=" else v > bound))
        if bad > flake_tolerance:
            reasons.append(f"{metric}: {bad}/{len(vals)} runs violate {direction} {bound}")
    for metric, limit in (("cost", max_cost_ratio), ("latency_p95", max_p95_latency_ratio)):
        c, b = candidate.get(metric), baseline.get(metric)
        if c and b and st.median(b) > 0 and st.median(c) / st.median(b) > limit:
            reasons.append(f"{metric}: x{st.median(c) / st.median(b):.2f} vs baseline exceeds x{limit}")
    return GateResult(not reasons, tuple(reasons))


@dataclass
class KillSwitch:
    """Engaged when env ACRS_KILL_SWITCH=1 or the flag file exists. Fail closed."""

    flag_file: str | None = None
    env_var: str = "ACRS_KILL_SWITCH"
    forced: bool = False

    def engaged(self) -> bool:
        if self.forced:
            return True
        if os.environ.get(self.env_var, "").strip() in {"1", "true", "on", "yes"}:
            return True
        if self.flag_file:
            try:
                return Path(self.flag_file).exists()
            except OSError:
                return True  # cannot read the switch -> treat as engaged
        return False


__all__ = [
    "GateResult",
    "KillSwitch",
    "ReleaseManifest",
    "distribution_report",
    "evaluation_gate",
    "pass_hat_k",
]
