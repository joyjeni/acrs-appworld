"""O2 - Adaptive Priority and Risk-aware Routing (APRR).

Positioning
-----------
==============================  ========================================================
SOTA                            Limitation this objective attacks
==============================  ========================================================
AutoGen / CAMEL / MetaGPT       Hand-wired static topologies: which agent handles which
                                step is fixed by the developer, not learned.
RouteLLM, FrugalGPT             Learn a router, but the decision is *query-level* and the
                                objective is a scalar cost-quality trade-off with no
                                notion of session risk or per-decision uncertainty.
Mixture-of-Agents (MoA)         Queries every agent at every layer - maximal cost, and
                                no routing decision at all.
LLM-as-router prompting         No calibration, no regret guarantee, no reproducibility.
==============================  ========================================================

Proposed
--------
A **contextual bandit router with a profile-conditioned risk posture and a
hierarchical escalation policy**.

For each open sub-goal, build a 13-dimensional context :math:`x_{a}` per
candidate agent (capability match, tier, cost, latency, live session risk,
remaining budget, in-session failure rate, ...). LinUCB maintains
:math:`\\hat{\\theta}_a` and :math:`A_a^{-1}` per agent and gives

.. math::

    \\hat{q}_a = \\hat{\\theta}_a^\\top x_a, \\qquad
    \\sigma_a = \\sqrt{x_a^\\top A_a^{-1} x_a}

The routing score is then **risk-adjusted**, not merely optimistic:

.. math::

    J_a = \\hat{q}_a + (\\alpha - \\gamma)\\,\\sigma_a
          - \\lambda_c \\tilde{c}_a - \\lambda_\\ell \\tilde{\\ell}_a
          - \\lambda_h h_a

The :math:`(\\alpha - \\gamma)\\sigma_a` term is the novel part. Classical
LinUCB is unconditionally optimistic (:math:`+\\alpha\\sigma`); mean-variance
control is unconditionally pessimistic (:math:`-\\gamma\\sigma`). Here the sign
is set by the requester's ``risk_tolerance`` and the live session risk, so the
same router explores on a low-stakes exploratory session and becomes pessimistic
once the session has started failing or the user is risk-averse. With
:math:`\\gamma = 0` it degrades exactly to LinUCB and inherits its
:math:`\\tilde{O}(d\\sqrt{T})` regret bound; ``regret.py`` measures realised
regret against the simulator's oracle so the claim is checked, not asserted.

**Escalation.** If :math:`\\max_a J_a` is below the admission threshold
:math:`\\tau`, or if the sub-goal is flagged high-stakes, the router refuses the
cheap answer and escalates one rung up the tier ladder
(light -> standard -> heavy), recording the reason in the trace. This is what
turns a cost optimiser into a *safety* mechanism: the fallback is always a more
capable agent, never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from paim.common.session_state import SessionState
from paim.common.types import AgentSpec, Tool, UserProfile
from paim.o2_aprr.agents import AgentPool
from paim.o2_aprr.linalg import Matrix, dot, identity, quad_form, sherman_morrison_update

CONTEXT_DIM = 13

_TIERS = ("light", "standard", "heavy")
_HIGH_STAKES_HINTS = (
    "price",
    "payment",
    "medical",
    "legal",
    "loan",
    "subsidy",
    "compliance",
    "safety",
    "emergency",
    "yield",
    "forecast",
)


@dataclass(frozen=True)
class RoutingConfig:
    alpha: float = 0.35          # exploration coefficient
    gamma_base: float = 0.30     # risk-aversion coefficient at risk_tolerance = 0
    lambda_cost: float = 0.55
    lambda_latency: float = 0.25
    lambda_halluc: float = 0.60
    admission_threshold: float = 0.42
    ridge: float = 1.0
    learning_rate: float = 1.0
    max_escalations: int = 2

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "gammaBase": self.gamma_base,
            "lambdaCost": self.lambda_cost,
            "lambdaLatency": self.lambda_latency,
            "lambdaHalluc": self.lambda_halluc,
            "admissionThreshold": self.admission_threshold,
            "ridge": self.ridge,
        }


@dataclass
class AgentScore:
    agent_id: str
    q_hat: float
    sigma: float
    cost_term: float
    latency_term: float
    halluc_term: float
    score: float
    capability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "qHat": round(self.q_hat, 5),
            "sigma": round(self.sigma, 5),
            "costTerm": round(self.cost_term, 5),
            "latencyTerm": round(self.latency_term, 5),
            "hallucTerm": round(self.halluc_term, 5),
            "capability": round(self.capability, 5),
            "score": round(self.score, 5),
        }


@dataclass
class RoutingDecision:
    agent_id: str
    scores: list[AgentScore]
    escalated: bool
    escalation_reason: str
    risk_posture: float
    high_stakes: bool
    context: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "scores": [s.to_dict() for s in self.scores],
            "escalated": self.escalated,
            "escalationReason": self.escalation_reason,
            "riskPosture": round(self.risk_posture, 5),
            "highStakes": self.high_stakes,
        }


def _norm_cost(cost: float) -> float:
    return min(1.0, cost / 0.015)


def _norm_latency(ms: float) -> float:
    return min(1.0, ms / 2000.0)


def is_high_stakes(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in _HIGH_STAKES_HINTS)


def build_context(
    agent: AgentSpec,
    sub_goal_text: str,
    tools: Sequence[Tool],
    state: SessionState,
    profile: UserProfile,
) -> list[float]:
    """13-d routing context. Order is part of the public contract (parity test)."""
    capability_terms = [t.category_name for t in tools] or [profile.sector]
    cap = max((agent.covers(c) for c in capability_terms), default=0.0)
    goal_cap = agent.covers(sub_goal_text[:80])

    failures = state.tool_failures
    fail_rate = min(1.0, sum(failures.values()) / 5.0) if failures else 0.0
    budget_left = 0.0
    if profile.token_budget > 0:
        budget_left = max(0.0, 1.0 - state.tokens_spent / profile.token_budget)

    return [
        1.0,                                              # 0 bias
        cap,                                              # 1 tool-category capability match
        1.0 if agent.tier == "light" else 0.0,            # 2
        1.0 if agent.tier == "standard" else 0.0,         # 3
        1.0 if agent.tier == "heavy" else 0.0,            # 4
        _norm_cost(agent.cost_per_call),                  # 5
        _norm_latency(agent.latency_ms),                  # 6
        state.risk,                                       # 7 live session risk
        budget_left,                                      # 8 remaining token budget
        fail_rate,                                        # 9 in-session failure pressure
        min(1.0, len(sub_goal_text.split()) / 40.0),      # 10 sub-goal complexity
        goal_cap,                                         # 11 capability match to goal text
        min(1.0, len(tools) / 10.0),                      # 12 candidate tool-set size
    ]


class APRRRouter:
    """Risk-adjusted LinUCB router with hierarchical escalation."""

    objective = "O2"
    name = "APRR"

    def __init__(self, pool: AgentPool | None = None, config: RoutingConfig | None = None) -> None:
        self.pool = pool or AgentPool()
        self.config = config or RoutingConfig()
        d = CONTEXT_DIM
        self._a_inv: dict[str, Matrix] = {
            a.agent_id: identity(d, 1.0 / self.config.ridge) for a in self.pool
        }
        self._b: dict[str, list[float]] = {a.agent_id: [0.0] * d for a in self.pool}
        self._theta: dict[str, list[float]] = {a.agent_id: [0.0] * d for a in self.pool}
        self.n_pulls: dict[str, int] = {a.agent_id: 0 for a in self.pool}
        self.n_decisions = 0
        self.n_escalations = 0
        self._warm_start()

    # -- learning ----------------------------------------------------------
    def _warm_start(self) -> None:
        """Seed each agent's posterior with its declared reliability.

        Without this, a cold LinUCB router routes the first few sub-goals
        essentially at random, which is fine asymptotically but makes a
        proposal-stage demo look broken. The prior is weak (one pseudo-
        observation) so real feedback overwhelms it quickly.
        """
        for agent in self.pool:
            x = [0.0] * CONTEXT_DIM
            x[0] = 1.0
            x[1] = 1.0
            self._observe(agent.agent_id, x, agent.reliability, weight=1.0)

    def _observe(self, agent_id: str, x: Sequence[float], reward: float, weight: float = 1.0) -> None:
        xs = [v * (weight ** 0.5) for v in x]
        self._a_inv[agent_id] = sherman_morrison_update(self._a_inv[agent_id], xs)
        b = self._b[agent_id]
        for i, v in enumerate(x):
            b[i] += reward * v * weight
        self._theta[agent_id] = [
            dot(row, b) for row in self._a_inv[agent_id]
        ]

    def update(self, agent_id: str, context: Sequence[float], reward: float) -> None:
        """Feed back an execution outcome (called by the orchestrator after O3)."""
        if agent_id not in self._a_inv:
            return
        self.n_pulls[agent_id] += 1
        self._observe(agent_id, context, max(0.0, min(1.0, reward)), self.config.learning_rate)

    # -- decision ----------------------------------------------------------
    def risk_posture(self, state: SessionState, profile: UserProfile) -> float:
        """Signed uncertainty coefficient (alpha - gamma).

        Positive => optimistic/exploring. Negative => pessimistic/risk-averse.
        """
        gamma = self.config.gamma_base * (1.0 - profile.risk_tolerance) + 0.45 * state.risk
        return self.config.alpha - gamma

    def score_agents(
        self,
        sub_goal_text: str,
        tools: Sequence[Tool],
        state: SessionState,
        profile: UserProfile,
        allowed: Sequence[AgentSpec] | None = None,
    ) -> tuple[list[AgentScore], dict[str, list[float]]]:
        agents = list(allowed if allowed is not None else self.pool.routable())
        posture = self.risk_posture(state, profile)
        cfg = self.config
        out: list[AgentScore] = []
        contexts: dict[str, list[float]] = {}

        for agent in agents:
            x = build_context(agent, sub_goal_text, tools, state, profile)
            contexts[agent.agent_id] = x
            q_hat = dot(self._theta[agent.agent_id], x)
            sigma = quad_form(self._a_inv[agent.agent_id], x) ** 0.5
            cost_term = cfg.lambda_cost * _norm_cost(agent.cost_per_call)
            lat_term = cfg.lambda_latency * _norm_latency(agent.latency_ms)
            hal_term = cfg.lambda_halluc * agent.hallucination_rate
            score = q_hat + posture * sigma - cost_term - lat_term - hal_term
            out.append(
                AgentScore(
                    agent_id=agent.agent_id,
                    q_hat=q_hat,
                    sigma=sigma,
                    cost_term=-cost_term,
                    latency_term=-lat_term,
                    halluc_term=-hal_term,
                    score=score,
                    capability=x[1],
                )
            )

        out.sort(key=lambda s: (-s.score, s.agent_id))
        return out, contexts

    def route(
        self,
        sub_goal_text: str,
        tools: Sequence[Tool],
        state: SessionState,
        profile: UserProfile,
    ) -> RoutingDecision:
        self.n_decisions += 1
        high_stakes = is_high_stakes(sub_goal_text)
        posture = self.risk_posture(state, profile)

        allowed = list(self.pool.routable())
        reason = ""
        escalated = False

        # Guard rail: a failing session or an explicitly high-stakes sub-goal may
        # not be served by the cheapest tier at all.
        if high_stakes or state.risk >= 0.5:
            filtered = [a for a in allowed if a.tier != "light"]
            if filtered:
                allowed = filtered
                escalated = True
                reason = (
                    "high-stakes sub-goal: light tier excluded"
                    if high_stakes
                    else f"session risk {state.risk:.2f} >= 0.50: light tier excluded"
                )

        scores, contexts = self.score_agents(sub_goal_text, tools, state, profile, allowed)
        if not scores:  # pragma: no cover - pool is never empty in practice
            raise RuntimeError("APRR: empty agent pool")

        ladder = list(self.pool.escalation_ladder())
        escalations = 0
        chosen = scores[0]
        while (
            chosen.score < self.config.admission_threshold
            and escalations < self.config.max_escalations
        ):
            current_tier = self.pool[chosen.agent_id].tier
            higher = _next_tier(current_tier)
            if higher is None:
                reason = reason or (
                    f"score {chosen.score:.3f} < tau {self.config.admission_threshold:.2f} "
                    "but already at top tier"
                )
                break
            candidates = [a for a in ladder if a.tier == higher]
            if not candidates:
                break
            allowed = candidates
            scores, contexts = self.score_agents(sub_goal_text, tools, state, profile, allowed)
            chosen = scores[0]
            escalations += 1
            escalated = True
            reason = (
                f"score below admission threshold {self.config.admission_threshold:.2f}: "
                f"escalated {current_tier} -> {higher}"
            )

        if escalated:
            self.n_escalations += 1

        return RoutingDecision(
            agent_id=chosen.agent_id,
            scores=scores,
            escalated=escalated,
            escalation_reason=reason,
            risk_posture=posture,
            high_stakes=high_stakes,
            context=contexts[chosen.agent_id],
        )

    def context_for(
        self,
        agent_id: str,
        sub_goal_text: str,
        tools: Sequence[Tool],
        state: SessionState,
        profile: UserProfile,
    ) -> list[float]:
        return build_context(self.pool[agent_id], sub_goal_text, tools, state, profile)

    def stats(self) -> dict[str, Any]:
        return {
            "decisions": self.n_decisions,
            "escalations": self.n_escalations,
            "escalationRate": round(self.n_escalations / self.n_decisions, 4)
            if self.n_decisions
            else 0.0,
            "pulls": dict(self.n_pulls),
        }


def _next_tier(tier: str) -> str | None:
    if tier not in _TIERS:
        return None
    idx = _TIERS.index(tier)
    return _TIERS[idx + 1] if idx + 1 < len(_TIERS) else None


# ---------------------------------------------------------------------------
# Baselines for the ablation table
# ---------------------------------------------------------------------------


class StaticHeavyRouter(APRRRouter):
    """Always route to the most capable agent (the MoA / 'just use GPT-4' policy)."""

    name = "Static heavy (no routing)"

    def route(self, sub_goal_text, tools, state, profile):  # type: ignore[override]
        self.n_decisions += 1
        heavy = self.pool.by_tier("heavy") or self.pool.routable()
        agent = heavy[0]
        scores, contexts = self.score_agents(sub_goal_text, tools, state, profile, [agent])
        return RoutingDecision(
            agent_id=agent.agent_id,
            scores=scores,
            escalated=False,
            escalation_reason="static policy",
            risk_posture=0.0,
            high_stakes=is_high_stakes(sub_goal_text),
            context=contexts[agent.agent_id],
        )


class CapabilityMatchRouter(APRRRouter):
    """Greedy capability match with no learning, cost model or escalation."""

    name = "Capability match (no learning)"

    def route(self, sub_goal_text, tools, state, profile):  # type: ignore[override]
        self.n_decisions += 1
        scores, contexts = self.score_agents(sub_goal_text, tools, state, profile)
        best = max(scores, key=lambda s: (s.capability, -s.score))
        return RoutingDecision(
            agent_id=best.agent_id,
            scores=scores,
            escalated=False,
            escalation_reason="capability match only",
            risk_posture=0.0,
            high_stakes=is_high_stakes(sub_goal_text),
            context=contexts[best.agent_id],
        )

    def update(self, agent_id, context, reward):  # type: ignore[override]
        self.n_pulls[agent_id] = self.n_pulls.get(agent_id, 0) + 1


class RoundRobinRouter(APRRRouter):
    """Load-balancing baseline: ignores context entirely."""

    name = "Round robin"

    def __init__(self, pool=None, config=None) -> None:
        super().__init__(pool, config)
        self._cursor = 0

    def route(self, sub_goal_text, tools, state, profile):  # type: ignore[override]
        self.n_decisions += 1
        routable = self.pool.routable()
        agent = routable[self._cursor % len(routable)]
        self._cursor += 1
        scores, contexts = self.score_agents(sub_goal_text, tools, state, profile, [agent])
        return RoutingDecision(
            agent_id=agent.agent_id,
            scores=scores,
            escalated=False,
            escalation_reason="round robin",
            risk_posture=0.0,
            high_stakes=is_high_stakes(sub_goal_text),
            context=contexts[agent.agent_id],
        )

    def update(self, agent_id, context, reward):  # type: ignore[override]
        self.n_pulls[agent_id] = self.n_pulls.get(agent_id, 0) + 1


__all__ = [
    "APRRRouter",
    "AgentScore",
    "CONTEXT_DIM",
    "CapabilityMatchRouter",
    "RoundRobinRouter",
    "RoutingConfig",
    "RoutingDecision",
    "StaticHeavyRouter",
    "build_context",
    "is_high_stakes",
]
