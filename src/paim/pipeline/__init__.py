"""The integrated pipeline that composes O1-O4 into one closed loop."""

from paim.pipeline.orchestrator import Ablations, IntegratedPipeline, PipelineConfig

STAGES = (
    {
        "id": "S0",
        "objective": "shared",
        "label": "Session state update",
        "detail": "Write the incoming intent into the exponentially decayed session memory.",
    },
    {
        "id": "S1",
        "objective": "O1",
        "label": "SASR retrieve + session-aware rerank",
        "detail": "BM25/bi-encoder first stage, then marginal-utility reranking conditioned on "
        "session memory, tool co-occurrence and the requester profile.",
    },
    {
        "id": "S2",
        "objective": "O4",
        "label": "FCNP context governor",
        "detail": "Assemble typed context blocks and prune to a PI-controlled token budget "
        "subject to a fidelity floor gated on tool-schema integrity.",
    },
    {
        "id": "S3",
        "objective": "O2",
        "label": "APRR risk-adjusted routing",
        "detail": "Contextual-bandit routing of the sub-goal with a profile-conditioned risk "
        "posture and hierarchical escalation.",
    },
    {
        "id": "S4",
        "objective": "O3",
        "label": "MNCD mesh deliberation",
        "detail": "Recruit mesh peers, allocate calls by contract net, execute, gossip, and "
        "certify with an evidence-weighted Byzantine quorum (or abstain).",
    },
    {
        "id": "S5",
        "objective": "shared",
        "label": "Credit assignment",
        "detail": "Certificate becomes the bandit reward; failures become endpoint penalties; "
        "surviving observations become the next turn's context.",
    },
)

FEEDBACK_EDGES = (
    {
        "from": "O3",
        "to": "O2",
        "label": "consensus certificate -> bandit reward",
        "detail": "The router is never trained on a label; it learns from whether the mesh "
        "could certify the answer at the cost it paid.",
    },
    {
        "from": "O3",
        "to": "O1",
        "label": "tool failures -> endpoint penalty + session risk",
        "detail": "A flaky endpoint sinks in later turns of the same session.",
    },
    {
        "from": "O3",
        "to": "O2",
        "label": "abstention -> forced escalation",
        "detail": "An abstained certificate raises session risk and re-routes, so abstention "
        "buys a stronger agent instead of returning a guess.",
    },
    {
        "from": "O4",
        "to": "O1",
        "label": "surviving observations -> session memory",
        "detail": "Anything the pruner dropped is genuinely gone, so O1 cannot depend on "
        "context the model never sees.",
    },
    {
        "from": "O1",
        "to": "O4",
        "label": "committed tools -> schema hard-keep set",
        "detail": "Retrieval decisions bound how far compression is allowed to go.",
    },
    {
        "from": "O1",
        "to": "O2",
        "label": "reranked tool set -> routing context",
        "detail": "Tool categories become the capability features of the routing context.",
    },
)

__all__ = ["FEEDBACK_EDGES", "STAGES", "Ablations", "IntegratedPipeline", "PipelineConfig"]
