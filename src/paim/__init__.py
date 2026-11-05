"""PAIM - the integrated agentic AI framework behind the PhD research proposal.

Four objectives, one closed loop:

* **O1 SASR** - Session-Aware Sequential Tool Retrieval and Reranking
* **O2 APRR** - Adaptive Priority and Risk-aware Routing
* **O3 MNCD** - Mesh-Networked Consensus and Deliberation
* **O4 FCNP** - Fidelity-Controlled Neural Context Pruning

See :mod:`paim.objectives` for the objective statements and
:mod:`paim.pipeline` for how the four are composed.
"""

from paim.objectives import OBJECTIVES, OVERALL_OBJECTIVE, RESEARCH_QUESTIONS
from paim.pipeline import Ablations, IntegratedPipeline, PipelineConfig

__version__ = "0.1.0"

__all__ = [
    "OBJECTIVES",
    "OVERALL_OBJECTIVE",
    "RESEARCH_QUESTIONS",
    "Ablations",
    "IntegratedPipeline",
    "PipelineConfig",
    "__version__",
]
