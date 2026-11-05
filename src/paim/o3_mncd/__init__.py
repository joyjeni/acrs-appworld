"""O3 - Mesh-Networked Consensus and Deliberation (MNCD)."""

from paim.o3_mncd.consensus import (
    ConsensusCertificate,
    ConsensusConfig,
    EvidenceWeightedConsensus,
    HighestConfidenceConsensus,
    MajorityVoteConsensus,
    consensus_correct,
)
from paim.o3_mncd.contract_net import Award, Bid, ContractNet, HubAllocator, TaskAnnouncement
from paim.o3_mncd.deliberation import DeliberationConfig, DeliberationResult, MeshDeliberation
from paim.o3_mncd.execution import (
    AgentReport,
    AgentRuntime,
    SimulatedToolExecutor,
    digest,
    evidence_token,
    expected_quality,
)
from paim.o3_mncd.mesh import (
    CompleteTopology,
    MeshTopology,
    StarTopology,
    byzantine_quorum,
    max_tolerated_faults,
    spectral_gap_estimate,
)

OBJECTIVE = {
    "id": "O3",
    "code": "MNCD",
    "title": "Mesh-Networked Consensus and Deliberation among heterogeneous agents",
    "statement": (
        "To replace hub-and-spoke and all-to-all multi-agent coordination with a "
        "degree-bounded peer mesh in which concrete tool calls are allocated by contract-net "
        "bidding and answers are certified by an evidence-weighted Byzantine quorum that can "
        "abstain, and to quantify the resulting robustness to confidently wrong agents at a "
        "bounded message and cost budget."
    ),
    "sota": [
        "Self-consistency (Wang et al. 2023): unweighted majority over samples of one model",
        "Multi-agent debate (Du et al. 2023): all-to-all critique then majority; O(n^2) messages",
        "LLM-as-a-judge: single arbiter, single point of failure, confidence-biased",
        "Chain-of-Agents / hub planners: sequential hand-offs with no quorum",
    ],
    "novelty": [
        "Degree-bounded connected mesh (ring + logarithmic chords) with O(n*deg) messages per round",
        "Contract-net allocation whose bids include the peer's live in-session record on that endpoint",
        "Evidence-weighted voting: weight collapses to zero when a claim's citations do not match "
        "observations a tool actually returned, so k agreeing fabricators cannot win",
        "Corroboration gate preventing any single peer from certifying a claim alone",
        "Abstention as a first-class outcome that requests escalation instead of guessing",
        "Explicit Byzantine quorum min(n, 2f+1) with f capped by the n >= 3f+1 bound",
    ],
}

__all__ = [
    "OBJECTIVE",
    "AgentReport",
    "AgentRuntime",
    "Award",
    "Bid",
    "CompleteTopology",
    "ConsensusCertificate",
    "ConsensusConfig",
    "ContractNet",
    "DeliberationConfig",
    "DeliberationResult",
    "EvidenceWeightedConsensus",
    "HighestConfidenceConsensus",
    "HubAllocator",
    "MajorityVoteConsensus",
    "MeshDeliberation",
    "MeshTopology",
    "SimulatedToolExecutor",
    "StarTopology",
    "TaskAnnouncement",
    "byzantine_quorum",
    "consensus_correct",
    "digest",
    "evidence_token",
    "expected_quality",
    "max_tolerated_faults",
    "spectral_gap_estimate",
]
