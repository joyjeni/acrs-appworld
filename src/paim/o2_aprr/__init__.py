"""O2 - Adaptive Priority and Risk-aware Routing (APRR)."""

from paim.o2_aprr.agents import DEFAULT_AGENTS, AgentPool
from paim.o2_aprr.regret import RegretTracker, oracle_expected_reward
from paim.o2_aprr.router import (
    CONTEXT_DIM,
    APRRRouter,
    AgentScore,
    CapabilityMatchRouter,
    RoundRobinRouter,
    RoutingConfig,
    RoutingDecision,
    StaticHeavyRouter,
    build_context,
    is_high_stakes,
)

OBJECTIVE = {
    "id": "O2",
    "code": "APRR",
    "title": "Adaptive Priority and Risk-aware Routing across a heterogeneous agent pool",
    "statement": (
        "To design and analyse an online routing policy that assigns each sub-goal of an "
        "agentic session to the agent maximising a risk-adjusted utility over expected quality, "
        "monetary cost, latency and hallucination exposure, with a hierarchical escalation "
        "guarantee that high-stakes or low-confidence sub-goals are never served by the "
        "cheapest tier, and to characterise its realised regret against an oracle policy."
    ),
    "sota": [
        "AutoGen / CAMEL / MetaGPT: static, developer-wired agent topologies",
        "RouteLLM / FrugalGPT: query-level cost-quality routers, no session state or risk posture",
        "Mixture-of-Agents: queries every agent every layer, maximal cost, no routing decision",
        "LLM-as-a-router prompting: uncalibrated, no regret analysis, not reproducible",
    ],
    "novelty": [
        "Sub-goal-level (not query-level) routing over a heterogeneous pool",
        "Signed uncertainty coefficient (alpha - gamma): the same router explores or turns "
        "pessimistic depending on requester risk tolerance and live session risk",
        "Explicit hallucination-exposure term in the routing objective",
        "Hierarchical escalation with an admission threshold, so the fallback is a stronger "
        "agent rather than a low-confidence answer",
        "Rewards are closed-loop: they come from O3's consensus certificate, not from a label",
        "Realised regret measured against a simulator oracle rather than an asserted bound",
    ],
}

__all__ = [
    "CONTEXT_DIM",
    "DEFAULT_AGENTS",
    "OBJECTIVE",
    "APRRRouter",
    "AgentPool",
    "AgentScore",
    "CapabilityMatchRouter",
    "RegretTracker",
    "RoundRobinRouter",
    "RoutingConfig",
    "RoutingDecision",
    "StaticHeavyRouter",
    "build_context",
    "is_high_stakes",
    "oracle_expected_reward",
]
