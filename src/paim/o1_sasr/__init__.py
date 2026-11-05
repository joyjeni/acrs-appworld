"""O1 - Session-Aware Sequential Tool Retrieval and Reranking (SASR)."""

from paim.o1_sasr.cooccurrence import CooccurrenceGraph
from paim.o1_sasr.reranker import (
    RerankResult,
    RerankWeights,
    ScoredCandidate,
    SessionAwareReranker,
    StatelessReranker,
)
from paim.o1_sasr.retriever import (
    BM25Retriever,
    Candidate,
    DenseRetriever,
    HybridRetriever,
)

OBJECTIVE = {
    "id": "O1",
    "code": "SASR",
    "title": "Session-Aware Sequential Tool Retrieval and Reranking",
    "statement": (
        "To formulate tool retrieval in multi-turn agentic sessions as a marginal-utility "
        "ranking problem conditioned on an explicit, temporally decayed session state, and to "
        "show that conditioning on session state and inter-tool co-occurrence structure "
        "improves multi-tool selection over stateless bi-encoder retrieval on ToolBench."
    ),
    "sota": [
        "ToolRetriever (ToolBench, Qin et al. 2023): stateless Sentence-BERT bi-encoder",
        "ToolRerank (2024): cross-encoder on (query, api) pairs only, still turn-independent",
        "API-Bank / ToolLLM: dialogue history kept in the prompt, never a ranking feature",
        "MMR / submodular selection: diversity against candidates, not against served intents",
    ],
    "novelty": [
        "Session intent memory with per-turn exponential decay used directly as a ranking feature",
        "PPMI tool co-occurrence graph fit on training trajectories, giving compositional continuation priors",
        "Explicit 'already served' vector so satisfied sub-goals stop attracting duplicate tools",
        "Requester-profile / sector prior that solves the cold-start turn before any history exists",
        "Per-session failure and saturation penalties fed back from execution (O3) outcomes",
        "Greedy submodular marginal-gain selection giving a (1-1/e) guarantee and a full scorable permutation",
    ],
}

__all__ = [
    "OBJECTIVE",
    "BM25Retriever",
    "Candidate",
    "CooccurrenceGraph",
    "DenseRetriever",
    "HybridRetriever",
    "RerankResult",
    "RerankWeights",
    "ScoredCandidate",
    "SessionAwareReranker",
    "StatelessReranker",
]
