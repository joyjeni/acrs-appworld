"""O4 - Fidelity-Controlled Neural Context Pruning (FCNP)."""

from paim.o4_fcnp.controller import (
    ControllerConfig,
    ControlStep,
    FidelityController,
    FixedBudgetController,
    verify_monotone_fidelity,
)
from paim.o4_fcnp.fidelity import FidelityEstimator, FidelityReport, FidelityWeights
from paim.o4_fcnp.pruner import (
    BlockScore,
    FCNPPruner,
    FixedRatioPruner,
    NoPruner,
    PruneResult,
    PruneWeights,
    TailTruncationPruner,
)

OBJECTIVE = {
    "id": "O4",
    "code": "FCNP",
    "title": "Fidelity-Controlled Neural Context Pruning for agentic context windows",
    "statement": (
        "To formulate agent context management as constrained online optimisation - minimise "
        "retained tokens subject to a task-fidelity floor - and to realise it as a closed-loop "
        "controller over typed context blocks whose fidelity measure gates on tool-schema "
        "integrity, so that compression can never silently break a callable tool signature."
    ),
    "sota": [
        "LLMLingua / LongLLMLingua: fixed-ratio, perplexity-driven token compression of prose",
        "Selective-Context: self-information filtering over a token stream",
        "H2O / SnapKV / StreamingLLM: KV-cache eviction inside the model, not over assembled context",
        "RAG top-k truncation: document-granular, so one verbose schema evicts the whole history",
    ],
    "novelty": [
        "Compression target is derived from a fidelity constraint rather than chosen by hand",
        "Typed context blocks (signature / required parameter / observation / session memory) "
        "instead of an undifferentiated token stream",
        "Schema integrity enters fidelity multiplicatively, so breaking a required parameter "
        "drives fidelity to zero and forces the controller to buy the tokens back",
        "PI control with anti-windup and a dead band, giving a fixed point at the smallest "
        "budget that satisfies the floor",
        "Operates inside the session loop: pruned observations feed back into O1's session memory",
        "Monotonicity premise of the fixed-point argument is verified empirically, not assumed",
    ],
}

__all__ = [
    "OBJECTIVE",
    "BlockScore",
    "ControlStep",
    "ControllerConfig",
    "FCNPPruner",
    "FidelityController",
    "FidelityEstimator",
    "FidelityReport",
    "FidelityWeights",
    "FixedBudgetController",
    "FixedRatioPruner",
    "NoPruner",
    "PruneResult",
    "PruneWeights",
    "TailTruncationPruner",
    "verify_monotone_fidelity",
]
