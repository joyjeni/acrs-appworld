"""O3 - evidence-weighted quorum consensus over mesh peer reports.

Positioning
-----------
================================  ======================================================
SOTA                              Limitation this objective attacks
================================  ======================================================
Self-consistency (Wang et al.)    Plain majority vote over samples of *one* model.
Multi-agent debate (Du et al.)    Rounds of all-to-all critique, then majority. Agents
                                  converge on a confident wrong answer as readily as on
                                  a right one - debate amplifies shared priors.
LLM-as-a-judge                    A single judge is a single point of failure and is
                                  itself susceptible to confident-sounding text.
Chain-of-Agents / hub planners     Sequential hand-offs; no quorum, so one bad link is
                                  unrecoverable.
================================  ======================================================

The shared weakness: **votes are counted, not weighed against evidence.** k
hallucinating agents that agree beat k-1 correct agents.

Proposed
--------
Weigh each report by how *groundable* it is, then require a Byzantine-style
quorum on the weighted result.

For a report *r* on sub-goal *g*:

.. math::

    w_r = \\big(\\eta_e\\,e_r + \\eta_c\\,c_r + \\eta_p\\,\\rho_r\\big)\\cdot g_r

* :math:`e_r` **evidence support**: the fraction of the report's citations that
  match observations actually returned by a tool call this turn. A fabricated
  claim cites nothing, so :math:`e_r = 0` and the whole weight collapses.
* :math:`c_r` self-reported confidence, which alone is *anti*-informative here
  (fabrications are the most confident reports) and so carries a small weight.
* :math:`\\rho_r` the peer's live posterior reliability in this session.
* :math:`g_r` **corroboration gate**: 1 if at least one other peer independently
  produced the same claim digest, else :math:`\\kappa < 1`. This is what stops a
  single confident peer from certifying anything on its own.

A claim is **certified** iff (i) it wins the weighted vote, (ii) its weighted
share is at least :math:`\\theta`, and (iii) the number of reports backing it is
at least the quorum :math:`\\min(n, 2f+1)` for the tolerated fault count *f*,
where *f* is itself capped by the classical :math:`n \\ge 3f+1` bound. Otherwise
the mechanism **abstains** and returns an escalation request rather than a
guess - abstention is a first-class outcome, which is the safety property the
majority-vote baselines cannot express.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from paim.o3_mncd.execution import AgentReport, evidence_token
from paim.o3_mncd.mesh import byzantine_quorum, max_tolerated_faults


@dataclass(frozen=True)
class ConsensusConfig:
    eta_evidence: float = 0.55
    eta_confidence: float = 0.15
    eta_reliability: float = 0.30
    corroboration_penalty: float = 0.35   # kappa
    certification_threshold: float = 0.50  # theta
    target_faults: int = 1                 # requested f

    def to_dict(self) -> dict[str, float]:
        return {
            "etaEvidence": self.eta_evidence,
            "etaConfidence": self.eta_confidence,
            "etaReliability": self.eta_reliability,
            "corroborationPenalty": self.corroboration_penalty,
            "certificationThreshold": self.certification_threshold,
            "targetFaults": float(self.target_faults),
        }


@dataclass
class WeightedVote:
    agent_id: str
    claim_key: str
    evidence_support: float
    confidence: float
    reliability: float
    corroborated: bool
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "claimKey": self.claim_key,
            "evidenceSupport": round(self.evidence_support, 4),
            "confidence": round(self.confidence, 4),
            "reliability": round(self.reliability, 4),
            "corroborated": self.corroborated,
            "weight": round(self.weight, 5),
        }


@dataclass
class ConsensusCertificate:
    """The auditable output of a deliberation round."""

    sub_goal_id: str
    certified: bool
    claim: str
    claim_key: str
    weighted_share: float
    supporting_agents: tuple[str, ...]
    dissenting_agents: tuple[str, ...]
    abstained: bool
    quorum_required: int
    quorum_observed: int
    tolerated_faults: int
    votes: list[WeightedVote] = field(default_factory=list)
    reason: str = ""
    evidence: tuple[str, ...] = ()

    @property
    def escalate(self) -> bool:
        return self.abstained

    def to_dict(self) -> dict[str, Any]:
        return {
            "subGoalId": self.sub_goal_id,
            "certified": self.certified,
            "claim": self.claim,
            "claimKey": self.claim_key,
            "weightedShare": round(self.weighted_share, 5),
            "supportingAgents": list(self.supporting_agents),
            "dissentingAgents": list(self.dissenting_agents),
            "abstained": self.abstained,
            "quorumRequired": self.quorum_required,
            "quorumObserved": self.quorum_observed,
            "toleratedFaults": self.tolerated_faults,
            "votes": [v.to_dict() for v in self.votes],
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


class EvidenceWeightedConsensus:
    """The O3 contribution."""

    objective = "O3"
    name = "MNCD evidence-weighted quorum"

    def __init__(self, config: ConsensusConfig | None = None) -> None:
        self.config = config or ConsensusConfig()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def observed_evidence(reports: Sequence[AgentReport]) -> set[str]:
        """The union of evidence tokens that a tool call actually produced.

        Built from invocation observations, *not* from the agents' citation
        lists, so a peer cannot manufacture evidence by claiming it.
        """
        out: set[str] = set()
        for r in reports:
            for inv in r.invocations:
                if inv.ok and inv.observation:
                    out.add(evidence_token(inv.doc_id, inv.observation))
        return out

    def evidence_support(self, report: AgentReport, observed: set[str]) -> float:
        if not report.evidence:
            return 0.0
        hits = sum(1 for e in report.evidence if e in observed)
        return hits / len(report.evidence)

    # -- main --------------------------------------------------------------
    def certify(
        self,
        sub_goal_id: str,
        reports: Sequence[AgentReport],
        reliabilities: dict[str, float] | None = None,
    ) -> ConsensusCertificate:
        cfg = self.config
        reliabilities = reliabilities or {}
        usable = [r for r in reports if r.ok]
        n = len(usable)

        f = min(cfg.target_faults, max_tolerated_faults(n))
        quorum_required = byzantine_quorum(n, f) if n else 1

        if n == 0:
            return ConsensusCertificate(
                sub_goal_id=sub_goal_id,
                certified=False,
                claim="",
                claim_key="",
                weighted_share=0.0,
                supporting_agents=(),
                dissenting_agents=tuple(r.agent_id for r in reports),
                abstained=True,
                quorum_required=quorum_required,
                quorum_observed=0,
                tolerated_faults=f,
                reason="no peer produced a groundable report",
            )

        observed = self.observed_evidence(usable)
        key_counts: dict[str, int] = defaultdict(int)
        for r in usable:
            key_counts[r.claim_key] += 1

        votes: list[WeightedVote] = []
        weight_by_key: dict[str, float] = defaultdict(float)
        agents_by_key: dict[str, list[str]] = defaultdict(list)

        for r in usable:
            support = self.evidence_support(r, observed)
            rel = reliabilities.get(r.agent_id, 0.7)
            corroborated = key_counts[r.claim_key] > 1
            base = (
                cfg.eta_evidence * support
                + cfg.eta_confidence * r.confidence
                + cfg.eta_reliability * rel
            )
            weight = base * (1.0 if corroborated else cfg.corroboration_penalty)
            votes.append(
                WeightedVote(
                    agent_id=r.agent_id,
                    claim_key=r.claim_key,
                    evidence_support=support,
                    confidence=r.confidence,
                    reliability=rel,
                    corroborated=corroborated,
                    weight=weight,
                )
            )
            weight_by_key[r.claim_key] += weight
            agents_by_key[r.claim_key].append(r.agent_id)

        total_weight = sum(weight_by_key.values())
        winner_key = max(weight_by_key.items(), key=lambda kv: (kv[1], kv[0]))[0]
        share = weight_by_key[winner_key] / total_weight if total_weight > 0 else 0.0
        quorum_observed = key_counts[winner_key]
        supporters = tuple(sorted(agents_by_key[winner_key]))
        dissenters = tuple(sorted(r.agent_id for r in usable if r.claim_key != winner_key))
        winner_report = next(r for r in usable if r.claim_key == winner_key)

        reasons: list[str] = []
        if share < cfg.certification_threshold:
            reasons.append(
                f"weighted share {share:.2f} < threshold {cfg.certification_threshold:.2f}"
            )
        if quorum_observed < quorum_required:
            reasons.append(
                f"quorum {quorum_observed} < required {quorum_required} (n={n}, f={f})"
            )
        if weight_by_key[winner_key] <= 1e-9:
            reasons.append("winning claim carries no evidence-backed weight")

        certified = not reasons
        return ConsensusCertificate(
            sub_goal_id=sub_goal_id,
            certified=certified,
            claim=winner_report.claim,
            claim_key=winner_key,
            weighted_share=share,
            supporting_agents=supporters,
            dissenting_agents=dissenters,
            abstained=not certified,
            quorum_required=quorum_required,
            quorum_observed=quorum_observed,
            tolerated_faults=f,
            votes=sorted(votes, key=lambda v: (-v.weight, v.agent_id)),
            reason="; ".join(reasons) if reasons else "certified",
            evidence=tuple(sorted(observed)),
        )


class MajorityVoteConsensus(EvidenceWeightedConsensus):
    """Baseline: unweighted majority over claim digests (self-consistency)."""

    name = "Majority vote (self-consistency)"

    def certify(self, sub_goal_id, reports, reliabilities=None):  # type: ignore[override]
        usable = [r for r in reports if r.ok]
        n = len(usable)
        if n == 0:
            return ConsensusCertificate(
                sub_goal_id, False, "", "", 0.0, (), (), True, 1, 0, 0,
                reason="no reports",
            )
        counts: dict[str, list[str]] = defaultdict(list)
        for r in usable:
            counts[r.claim_key].append(r.agent_id)
        winner_key = max(counts.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
        winner = next(r for r in usable if r.claim_key == winner_key)
        share = len(counts[winner_key]) / n
        return ConsensusCertificate(
            sub_goal_id=sub_goal_id,
            certified=True,  # majority voting never abstains
            claim=winner.claim,
            claim_key=winner_key,
            weighted_share=share,
            supporting_agents=tuple(sorted(counts[winner_key])),
            dissenting_agents=tuple(
                sorted(r.agent_id for r in usable if r.claim_key != winner_key)
            ),
            abstained=False,
            quorum_required=1,
            quorum_observed=len(counts[winner_key]),
            tolerated_faults=0,
            reason="plurality of claim digests",
        )


class HighestConfidenceConsensus(EvidenceWeightedConsensus):
    """Baseline: trust the most confident peer (LLM-as-judge stand-in)."""

    name = "Highest confidence (single judge)"

    def certify(self, sub_goal_id, reports, reliabilities=None):  # type: ignore[override]
        usable = [r for r in reports if r.ok]
        if not usable:
            return ConsensusCertificate(
                sub_goal_id, False, "", "", 0.0, (), (), True, 1, 0, 0, reason="no reports"
            )
        winner = max(usable, key=lambda r: (r.confidence, r.agent_id))
        return ConsensusCertificate(
            sub_goal_id=sub_goal_id,
            certified=True,
            claim=winner.claim,
            claim_key=winner.claim_key,
            weighted_share=1.0,
            supporting_agents=(winner.agent_id,),
            dissenting_agents=tuple(
                sorted(r.agent_id for r in usable if r.agent_id != winner.agent_id)
            ),
            abstained=False,
            quorum_required=1,
            quorum_observed=1,
            tolerated_faults=0,
            reason="single highest-confidence report",
        )


def consensus_correct(certificate: ConsensusCertificate, reports: Iterable[AgentReport]) -> bool:
    """Evaluation-only oracle: did the certified claim come from a faithful peer?"""
    truth = {
        r.claim_key
        for r in reports
        if r.ok and not r.hallucinated and not getattr(r, "hijacked", False) and r.confidence > 0.5
    }
    if certificate.abstained:
        return False
    return certificate.claim_key in truth


__all__ = [
    "ConsensusCertificate",
    "ConsensusConfig",
    "EvidenceWeightedConsensus",
    "HighestConfidenceConsensus",
    "MajorityVoteConsensus",
    "WeightedVote",
    "consensus_correct",
]
