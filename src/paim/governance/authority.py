"""Capability vs authority: which ACRS stage may recommend, which may decide.

Two questions, asked per stage and in this order:

1. **Should a model be involved at all?** (the decision sequence)
   - no judgment in the step            -> plain code, no model
   - steps knowable in advance          -> a workflow pattern
   - the model must choose what is next -> an agent (needs a termination rule)
2. **What may it decide once it is there?** Capability and authority are
   independent. "Human in the loop" is not an answer; it resolves into one of
   four oversight designs, chosen per decision:

   ===========================  ==========================================
   APPROVE_BEFORE_EXECUTE       irreversible step; nothing runs unsigned
   REVIEW_AFTER_EXECUTE         reversal is genuinely cheap
   SAMPLE_AUDIT                 error cost measured; it sets the audit rate
   ESCALATE_ON_UNCERTAINTY      only when confidence is *calibrated*
   ===========================  ==========================================

The registry below is the ACRS answer for every stage. It is data, not prose,
so tests can assert it (e.g. "no stage that decides is model-based", "every
model-based stage names its failure mode and detector").

Reliability compounds multiplicatively across stages, so the stage list also
drives :func:`end_to_end_reliability` - the number reported next to every
pipeline result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping


class Oversight(str, Enum):
    NONE = "none (deterministic)"
    APPROVE_BEFORE_EXECUTE = "approve-before-execute"
    REVIEW_AFTER_EXECUTE = "review-after-execute"
    SAMPLE_AUDIT = "sample-audit"
    ESCALATE_ON_UNCERTAINTY = "escalate-on-uncertainty"


class Disposition(str, Enum):
    """Outcome of one governed turn.

    EXECUTE  the certified answer/action proceeds (clear, policy-allowed result)
    REVIEW   held for a named human; the system recommends, a person decides
    BLOCK    a policy denies it or the kill switch is engaged
    A failure in any stage can only ever produce REVIEW or BLOCK (fail closed).
    """

    EXECUTE = "EXECUTE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class Pattern(str, Enum):
    PLAIN_CODE = "plain code, no model"
    PROMPT_CHAINING = "prompt chaining"
    ROUTING = "routing"
    PARALLEL_VOTING = "parallelization (voting)"
    PARALLEL_SECTIONING = "parallelization (sectioning)"
    ORCHESTRATOR_WORKERS = "orchestrator-workers"
    EVALUATOR_OPTIMIZER = "evaluator-optimizer"
    AGENT = "agent (bounded loop)"


class ComponentType(str, Enum):
    INGEST = "ingestion/normalisation"
    RULES = "matching/rules"
    RECOMMEND = "model-assisted recommendation"
    APPROVAL = "human approval gate"
    PUBLISH = "publish/write-back"


@dataclass(frozen=True)
class StageSpec:
    stage: str
    objective: str
    component: ComponentType
    pattern: Pattern
    uses_model: bool
    may_recommend: bool
    may_decide: str            # "no" | "bounded" | "deterministic-after-approval" | "yes (deterministic)"
    oversight: Oversight
    accountable_owner: str     # set by the deploying organisation, not by the engineer
    why_model: str
    failure_mode: str
    detector: str
    audit: str

    def to_dict(self) -> dict[str, str | bool]:
        d = asdict(self)
        for k in ("component", "pattern", "oversight"):
            d[k] = getattr(self, k).value
        return d


ACRS_STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        "S0", "shared", ComponentType.INGEST, Pattern.PLAIN_CODE, False, False,
        "yes (deterministic)", Oversight.NONE, "platform owner",
        "No judgment: tokenise intent and decay session memory.",
        "stale memory after topic switch", "topic-switch test in O1 suite",
        "state snapshot in StageTrace",
    ),
    StageSpec(
        "S1", "O1 SATR", ComponentType.RULES, Pattern.PLAIN_CODE, False, True,
        "no", Oversight.NONE, "platform owner",
        "Deterministic scoring (BM25/dense + session marginal utility); an encoder "
        "embeds text but chooses nothing.",
        "turn-amnesic or provider-locked ranking", "held-out NDCG@5 per turn index, CI",
        "first-stage and reranked top-k logged",
    ),
    StageSpec(
        "S2", "O4 FCNP", ComponentType.RULES, Pattern.PLAIN_CODE, False, True,
        "no", Oversight.NONE, "platform owner",
        "PI controller over typed blocks; schema blocks of committed tools are hard-kept.",
        "pruning a block the answer needed", "fidelity floor Phi* satisfaction rate",
        "kept/pruned block ids, controller action",
    ),
    StageSpec(
        "S3", "O2 APRR", ComponentType.RECOMMEND, Pattern.ROUTING, True, True,
        "bounded", Oversight.SAMPLE_AUDIT, "service owner",
        "Which agent tier is cheapest-sufficient is learned from feedback; categories "
        "are distinct and need different capabilities.",
        "front-door misroute (cheap agent on hard sub-goal)",
        "regret and oracle agreement measured separately from answer quality",
        "posture, scores, escalation reason",
    ),
    StageSpec(
        "S4", "O3 MNCD", ComponentType.RECOMMEND, Pattern.PARALLEL_VOTING, True, True,
        "bounded", Oversight.ESCALATE_ON_UNCERTAINTY, "service owner",
        "Agents form claims over shared tool evidence; agreement is only informative "
        "when it is evidence-weighted.",
        "correlated error (colluding or injected peers agree on the same wrong claim)",
        "agreement against a labelled sample; calibration (ECE) of weighted share",
        "votes, evidence tokens, quorum, certificate",
    ),
    StageSpec(
        "G1", "governance", ComponentType.APPROVAL, Pattern.PLAIN_CODE, False, False,
        "yes (deterministic)", Oversight.APPROVE_BEFORE_EXECUTE, "named human reviewer",
        "check_policy is plain code by design: a guardrail an agent can argue with is not one.",
        "rule skipped or bypassed", "transition refused without a recorded decision id",
        "hash-chained decision log",
    ),
    StageSpec(
        "S5", "shared", ComponentType.PUBLISH, Pattern.PLAIN_CODE, False, False,
        "deterministic-after-approval", Oversight.REVIEW_AFTER_EXECUTE, "platform owner",
        "Write-back (bandit reward, memory carry-forward) executes only what G1 allowed.",
        "poisoned memory (quarantined observation written into SATR prior)",
        "write-back policy R-MEM-01 decisions",
        "carried block ids and reward",
    ),
)


def stage(stage_id: str) -> StageSpec:
    for s in ACRS_STAGES:
        if s.stage == stage_id:
            return s
    raise KeyError(stage_id)


def registry_violations(stages: Iterable[StageSpec] = ACRS_STAGES) -> list[str]:
    """Design invariants the thesis claims. Empty list means the design holds."""
    out = []
    for s in stages:
        if s.uses_model and s.may_decide not in {"no", "bounded"}:
            out.append(f"{s.stage}: a model-based stage may at most decide within bounds")
        if s.uses_model and s.oversight == Oversight.NONE:
            out.append(f"{s.stage}: model-based stage without an oversight design")
        if s.uses_model and (not s.failure_mode or not s.detector):
            out.append(f"{s.stage}: model-based stage must name failure mode and detector")
        if s.component == ComponentType.APPROVAL and s.uses_model:
            out.append(f"{s.stage}: the approval gate must not be a model")
    comps = {s.component for s in stages}
    missing = set(ComponentType) - comps
    if missing:
        out.append("missing component types: " + ", ".join(sorted(c.value for c in missing)))
    return out


def end_to_end_reliability(per_stage: Mapping[str, float]) -> float:
    """Reliability of a chain is the product of its stages, not their mean."""
    r = 1.0
    for v in per_stage.values():
        r *= max(0.0, min(1.0, float(v)))
    return r


__all__ = [
    "ACRS_STAGES",
    "ComponentType",
    "Disposition",
    "Oversight",
    "Pattern",
    "StageSpec",
    "end_to_end_reliability",
    "registry_violations",
    "stage",
]
