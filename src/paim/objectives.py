"""Objective definitions: the research content, in one machine-readable place.

Everything downstream reads from here - the slide generator
(``slides/build_deck.py``), the exported JSON the web app renders
(``scripts/export_web_bundle.py``), and the docs. Editing an objective statement
here changes the deck, the site and the documentation together, which is the
only way a proposal, an implementation and a set of slides stay consistent.

Each objective carries the same six fields, deliberately, so the deck can render
them with one template:

``statement``   the formal objective, in the "To <verb> ..." form examiners expect
``questions``   the research questions it answers
``sota``        what the state of the art does, and the specific limitation attacked
``novelty``     the claimed contributions, one per bullet
``method``      how it is realised in the implementation
``validation``  what would falsify the claim, and on what data
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Overall
# ---------------------------------------------------------------------------

TITLE = (
    "A Session-Aware, Risk-Routed and Fidelity-Controlled Framework for "
    "Reliable Large-Scale Tool Use in Multi-Agent LLM Systems"
)

OVERALL_OBJECTIVE: dict[str, Any] = {
    "aim": (
        "To design, implement and evaluate an integrated agentic framework in which tool "
        "retrieval, agent routing, multi-agent deliberation and context compression are "
        "coupled through a shared session state and closed feedback loops, so that a "
        "multi-turn agent operating over a corpus of thousands of tools remains accurate, "
        "auditable and bounded in cost."
    ),
    "problem": (
        "Contemporary tool-using agents are assembled from four independently optimised "
        "components. Retrieval is stateless, routing is hand-wired, deliberation counts votes "
        "instead of weighing evidence, and context is compressed to a ratio someone guessed. "
        "Each is defensible alone; composed, their failures multiply. A stateless retriever "
        "re-proposes an endpoint the session already exhausted; a static router sends a "
        "high-stakes sub-goal to a cheap agent; majority voting certifies whatever the "
        "confident agents agreed on; and fixed-ratio compression deletes the one required "
        "parameter that made the call executable. None of the four components can see the "
        "others' state, so none can compensate."
    ),
    "thesis": (
        "These four failures share one cause - the absence of an explicit, shared session "
        "state that every component both reads and writes - and therefore admit one coupled "
        "solution rather than four independent ones."
    ),
    "objectives_summary": (
        "Four specific objectives (O1-O4) each contribute a mechanism, and a fifth "
        "integration objective (O5) establishes that the mechanisms compose into a working "
        "end-to-end pipeline with measurable feedback rather than a sequential cascade."
    ),
    "expected_contributions": [
        "A formal model of an agentic session as a shared state read and written by all "
        "coordination components, with the six feedback edges made explicit and instrumented.",
        "Four mechanisms - session-aware marginal-utility tool reranking, risk-adjusted "
        "bandit routing with hierarchical escalation, evidence-weighted Byzantine quorum "
        "consensus over a degree-bounded mesh, and PI-controlled context pruning under a "
        "schema-gated fidelity floor.",
        "ToolBench-Sessions: a label-preserving multi-turn re-derivation of the ToolBench G1 "
        "retrieval benchmark that makes session-awareness measurable without new annotation.",
        "A reproducible reference implementation in which every objective can be ablated back "
        "to its state-of-the-art analogue inside the same orchestrator, so all comparisons are "
        "ablations rather than reimplementations.",
        "An end-to-end validation on two legs - the 10,439-endpoint ToolBench corpus for ranking, and live "
        "Indian open-government Agriculture data from data.gov.in for execution. Agriculture records "
        "are never invented; ToolBench RapidAPI keys are not redistributable so those endpoints stay ranking-only.",
    ],
    "scope_and_limits": [
        "The proposal stage establishes pipeline correctness and integration, not competitive "
        "end-task accuracy; agent behaviour is simulated from declared reliability and "
        "hallucination rates, and every trace labels itself as such.",
        "ToolBench endpoints are called through a schema-faithful simulator because RapidAPI "
        "keys are not redistributable; the data.gov.in leg is genuinely live and is what "
        "demonstrates the pipeline against a real API.",
        "The regret and fixed-point arguments are stated under their assumptions and checked "
        "empirically; they are not presented as proofs for the full non-linear system.",
    ],
}

RESEARCH_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "RQ1",
        "question": (
            "Does conditioning tool retrieval on an explicit, decayed session state and on "
            "learned inter-tool co-occurrence improve multi-tool selection over stateless "
            "bi-encoder retrieval, on multi-turn sessions derived from ToolBench?"
        ),
        "objective": "O1",
    },
    {
        "id": "RQ2",
        "question": (
            "Can a contextual-bandit router with a profile-conditioned risk posture achieve "
            "sub-linear realised regret against an oracle while guaranteeing that high-stakes "
            "or low-confidence sub-goals are never served by the cheapest agent tier?"
        ),
        "objective": "O2",
    },
    {
        "id": "RQ3",
        "question": (
            "Does weighing agent reports by verifiable evidence, rather than counting votes, "
            "keep multi-agent consensus correct in the presence of confidently wrong agents, "
            "at a bounded message and cost budget?"
        ),
        "objective": "O3",
    },
    {
        "id": "RQ4",
        "question": (
            "Can context compression be driven by a fidelity constraint instead of a chosen "
            "ratio, such that the controller provably backs off when compression would break "
            "a callable tool signature?"
        ),
        "objective": "O4",
    },
    {
        "id": "RQ5",
        "question": (
            "When the four mechanisms are composed through a shared session state, do the "
            "feedback edges actually carry signal - does an execution failure change a later "
            "retrieval decision, and does an abstention change a later routing decision - and "
            "does the integrated pipeline run end to end on live data.gov.in Agriculture tools "
            "(ToolBench endpoints remain ranking-only because RapidAPI keys are not redistributable)?"
        ),
        "objective": "O5",
    },
]


# ---------------------------------------------------------------------------
# Specific objectives
# ---------------------------------------------------------------------------

OBJECTIVES: list[dict[str, Any]] = [
    {
        "id": "O1",
        "code": "SASR",
        "repo": "session-aware-toolbench-rerank",
        "module": "paim.o1_sasr",
        "title": "Session-Aware Sequential Tool Retrieval and Reranking",
        "short": "Make retrieval remember the session.",
        "statement": (
            "To formulate tool retrieval in a multi-turn agentic session as a marginal-utility "
            "ranking problem conditioned on an explicit, temporally decayed session state, and "
            "to establish that conditioning on session state, learned inter-tool co-occurrence "
            "and the requester profile improves multi-tool selection over stateless bi-encoder "
            "retrieval on the ToolBench G1 corpus."
        ),
        "questions": ["RQ1"],
        "sota": [
            {
                "work": "ToolRetriever (ToolBench, Qin et al. 2023)",
                "does": "Fine-tuned Sentence-BERT bi-encoder; one embedding of the instruction, "
                "one nearest-neighbour lookup over the API corpus.",
                "limitation": "Stateless. Turn 5 of a session is scored exactly like turn 1, so "
                "the ranker cannot know which endpoints the session already exhausted.",
            },
            {
                "work": "ToolRerank (2024)",
                "does": "Adds a cross-encoder second stage over (query, api) pairs.",
                "limitation": "Reranking features are still turn-independent; no session term.",
            },
            {
                "work": "API-Bank, ToolLLM planners",
                "does": "Keep dialogue history inside the prompt.",
                "limitation": "History competes with tool schemas for context budget and is "
                "never used as an explicit ranking feature.",
            },
            {
                "work": "MMR / submodular diversity",
                "does": "Diversifies a candidate list.",
                "limitation": "Diversity is measured against other candidates, not against the "
                "intents the session has already satisfied.",
            },
        ],
        "novelty": [
            "Session intent memory with per-turn exponential decay, used directly as a ranking "
            "feature rather than as prompt text.",
            "PPMI-weighted tool co-occurrence graph fitted on training trajectories only, "
            "supplying compositional continuation priors for multi-tool queries.",
            "An explicit 'already served' vector, so a satisfied sub-goal stops attracting "
            "duplicate endpoints - the failure mode MMR cannot express.",
            "A requester-profile and sector prior that resolves the cold-start turn, before any "
            "session history exists.",
            "Per-endpoint failure and saturation penalties written back from O3's execution "
            "outcomes, so a flaky endpoint sinks within the same session.",
            "Greedy marginal-gain selection over a submodular set objective, giving a (1-1/e) "
            "guarantee and a full candidate permutation that is directly scorable against qrels.",
        ],
        "method": [
            "First stage: Okapi BM25 over rendered endpoint documents (dependency-free and "
            "reproducible offline), with an optional Sentence-BERT bi-encoder and reciprocal-"
            "rank fusion when weights are available.",
            "Second stage: seven-term additive score - relevance, goal match, session affinity, "
            "co-occurrence affinity, profile prior, redundancy penalty, served penalty, failure "
            "penalty - each term recorded per candidate so every ranking is explainable.",
            "Selection: greedy marginal gain, recomputing the committed-set-dependent terms "
            "after each pick.",
        ],
        "validation": [
            "NDCG@5, unclipped Recall@5, MRR and selection F1 on the 100 held-out ToolBench G1 "
            "queries and on the derived ToolBench-Sessions turns.",
            "Ablation against the stateless configuration (all session weights zero) inside the "
            "same orchestrator, so the delta isolates session conditioning.",
            "Leakage test: the co-occurrence graph is asserted to contain no test-query gold set.",
            "Falsifier: if the session-aware configuration does not beat the stateless one on "
            "multi-turn sessions, the objective's central claim fails.",
        ],
    },
    {
        "id": "O2",
        "code": "APRR",
        "repo": "aprr-multi-agent-routing",
        "module": "paim.o2_aprr",
        "title": "Adaptive Priority and Risk-aware Routing across a heterogeneous agent pool",
        "short": "Route each sub-goal by risk-adjusted utility, and escalate rather than guess.",
        "statement": (
            "To design and analyse an online routing policy that assigns each sub-goal of an "
            "agentic session to the agent maximising a risk-adjusted utility over expected "
            "quality, monetary cost, latency and hallucination exposure, subject to a "
            "hierarchical escalation guarantee that high-stakes or low-confidence sub-goals are "
            "never served by the cheapest tier, and to characterise its realised regret against "
            "an oracle policy."
        ),
        "questions": ["RQ2"],
        "sota": [
            {
                "work": "AutoGen, CAMEL, MetaGPT",
                "does": "Compose agents in a topology fixed by the developer.",
                "limitation": "Which agent handles which step is a design-time constant; the "
                "system cannot reallocate when an agent underperforms.",
            },
            {
                "work": "RouteLLM, FrugalGPT",
                "does": "Learn a router that trades cost against quality.",
                "limitation": "Decisions are query-level and the objective is a scalar "
                "trade-off; no session state, no per-decision uncertainty, no risk posture.",
            },
            {
                "work": "Mixture-of-Agents",
                "does": "Queries every agent at every layer and aggregates.",
                "limitation": "Maximal cost by construction - there is no routing decision at all.",
            },
            {
                "work": "LLM-as-a-router prompting",
                "does": "Asks a model which agent should handle the request.",
                "limitation": "Uncalibrated, non-reproducible, and admits no regret analysis.",
            },
        ],
        "novelty": [
            "Routing at sub-goal granularity rather than query granularity, so a compound "
            "request can be split across tiers.",
            "A signed uncertainty coefficient (alpha - gamma): the same router explores when the "
            "session is low-stakes and turns pessimistic once it starts failing or the requester "
            "is risk-averse. LinUCB is the gamma = 0 special case.",
            "Hallucination exposure as an explicit term in the routing objective, not a "
            "post-hoc filter.",
            "Hierarchical escalation with an admission threshold, so the fallback for a "
            "low-confidence decision is a stronger agent rather than a weak answer.",
            "Closed-loop rewards: the reward signal is O3's consensus certificate, so the router "
            "learns from certifiability at cost, never from a supervised label.",
            "Realised regret measured against a simulator oracle instead of an asserted bound.",
        ],
        "method": [
            "Thirteen-dimensional routing context per candidate agent: capability match, tier "
            "one-hot, normalised cost and latency, live session risk, remaining token budget, "
            "in-session failure pressure, sub-goal complexity, candidate-set size.",
            "LinUCB posterior per agent, maintained with Sherman-Morrison rank-one inverse "
            "updates on plain lists so the policy is portable and dependency-free.",
            "Risk-adjusted score with cost, latency and hallucination penalties, then an "
            "escalation loop bounded by max_escalations.",
            "Weak reliability prior (one pseudo-observation) so cold-start decisions are sane.",
        ],
        "validation": [
            "Realised cumulative regret and oracle agreement over sessions; the claim is a "
            "sub-linear regret curve, plotted rather than asserted.",
            "Escalation audit: every high-stakes sub-goal must show a non-light tier, checked as "
            "a hard invariant in the test suite.",
            "Cost and latency accounting per session against the static-heavy and round-robin "
            "baselines.",
            "Falsifier: if realised regret grows linearly, or a high-stakes sub-goal is ever "
            "served by the light tier, the objective fails.",
        ],
    },
    {
        "id": "O3",
        "code": "MNCD",
        "repo": "mncd-mesh-agents",
        "module": "paim.o3_mncd",
        "title": "Mesh-Networked Consensus and Deliberation among heterogeneous agents",
        "short": "Weigh evidence, not votes - and allow the system to abstain.",
        "statement": (
            "To replace hub-and-spoke and all-to-all multi-agent coordination with a "
            "degree-bounded peer mesh in which concrete tool calls are allocated by contract-net "
            "bidding and answers are certified by an evidence-weighted Byzantine quorum that may "
            "abstain, and to quantify the resulting robustness to confidently wrong agents at a "
            "bounded message and cost budget."
        ),
        "questions": ["RQ3"],
        "sota": [
            {
                "work": "Self-consistency (Wang et al. 2023)",
                "does": "Samples one model repeatedly and takes the majority answer.",
                "limitation": "Unweighted majority. Correlated errors are amplified, not "
                "detected.",
            },
            {
                "work": "Multi-agent debate (Du et al. 2023)",
                "does": "Rounds of all-to-all critique, then a majority vote.",
                "limitation": "O(n^2) messages per round, and agents converge on a confident "
                "wrong answer as readily as on a right one.",
            },
            {
                "work": "LLM-as-a-judge",
                "does": "Delegates arbitration to a single strong model.",
                "limitation": "Single point of failure, and demonstrably biased towards "
                "confident-sounding text.",
            },
            {
                "work": "Chain-of-Agents, hub planners",
                "does": "Sequential hand-offs coordinated by an orchestrator.",
                "limitation": "No quorum, so one bad link is unrecoverable, and the hub is a "
                "bottleneck and a single point of failure.",
            },
        ],
        "novelty": [
            "Degree-bounded connected mesh (Hamiltonian ring plus logarithmic chords) with "
            "O(n x degree) messages per round instead of O(n^2), and an exactly computed diameter "
            "bounding the gossip rounds needed.",
            "Contract-net bids that include the bidder's live in-session track record on that "
            "specific endpoint, so the mesh self-heals without a central scheduler.",
            "Evidence-weighted voting in which a report's weight collapses to zero when its "
            "citations do not match observations a tool actually returned - so k agreeing "
            "fabricators cannot outvote one grounded peer.",
            "A corroboration gate that prevents any single peer from certifying a claim alone.",
            "Abstention as a first-class outcome that emits an escalation request instead of a "
            "guess; majority voting cannot express this.",
            "Explicit Byzantine quorum min(n, 2f+1) with f capped by the classical n >= 3f+1 "
            "bound, reported in every certificate.",
        ],
        "method": [
            "Five-phase round per sub-goal: recruit from the mesh neighbourhood, allocate calls "
            "by contract net, execute and form claims, gossip claim distributions, certify.",
            "Claim identity is a content digest of the observations the peer actually received, "
            "so agreement is verifiable rather than textual.",
            "Certificate records supporting and dissenting peers, weighted share, quorum "
            "observed versus required, tolerated fault count, and the reason for any abstention.",
            "Gossip beliefs are advisory and do not enter the certificate, keeping the safety "
            "argument independent of the mixing assumption.",
        ],
        "validation": [
            "Consensus accuracy as the number of hallucinating peers increases, against majority "
            "vote and single-judge baselines.",
            "Message count and cost per certified answer against star and complete topologies.",
            "Invariant tests: the consensus mechanism never reads the ground-truth "
            "hallucination flag; a claim with no matching evidence can never be certified.",
            "Falsifier: if evidence weighting does not beat majority voting as the fabricator "
            "count rises, the objective fails.",
        ],
    },
    {
        "id": "O4",
        "code": "FCNP",
        "repo": "fcnp-context-pruning",
        "module": "paim.o4_fcnp",
        "title": "Fidelity-Controlled Neural Context Pruning for agentic context windows",
        "short": "Compress as far as a fidelity floor allows, never far enough to break a call.",
        "statement": (
            "To formulate agent context management as constrained online optimisation - minimise "
            "retained tokens subject to a task-fidelity floor - and to realise it as a "
            "closed-loop controller over typed context blocks whose fidelity measure gates "
            "multiplicatively on tool-schema integrity, so that compression can never silently "
            "break a callable tool signature."
        ),
        "questions": ["RQ4"],
        "sota": [
            {
                "work": "LLMLingua, LongLLMLingua",
                "does": "Perplexity-driven token compression of free prose.",
                "limitation": "Compresses to a ratio the engineer picks, and is unaware of which "
                "spans are syntactically load-bearing for a tool call.",
            },
            {
                "work": "Selective-Context",
                "does": "Filters tokens by self-information.",
                "limitation": "Same family; a required parameter name that appears once is "
                "exactly what an information criterion discards.",
            },
            {
                "work": "H2O, SnapKV, StreamingLLM",
                "does": "Evicts KV-cache entries inside a forward pass.",
                "limitation": "Operates inside the model, so it cannot govern the context an "
                "orchestrator assembles from retrieval and observations.",
            },
            {
                "work": "RAG top-k truncation",
                "does": "Keeps the k highest-scoring documents whole.",
                "limitation": "Document-granular, so one verbose tool schema can evict the "
                "entire session history.",
            },
        ],
        "novelty": [
            "The compression target is derived from a fidelity constraint rather than chosen by "
            "hand - the controller answers 'how much can I compress this turn?'.",
            "Pruning operates over typed blocks (signature, required parameter, observation, "
            "session memory) rather than an undifferentiated token stream.",
            "Schema integrity enters fidelity multiplicatively, so breaking a required parameter "
            "drives fidelity towards zero and forces the controller to buy the tokens back.",
            "PI control with anti-windup and a dead band, whose fixed point is the smallest "
            "budget satisfying the floor.",
            "Runs inside the session loop: surviving observations become the next turn's context "
            "and feed O1's session memory, so compression decisions have downstream consequences.",
            "The monotonicity premise underlying the fixed-point argument is verified "
            "empirically rather than assumed.",
        ],
        "method": [
            "Fidelity = (0.55 x sub-goal term coverage + 0.45 x retained criticality mass) x "
            "schema integrity, all computable from the retained blocks alone.",
            "Value-density knapsack admission with a hard-keep set covering the query, the open "
            "sub-goal, every committed tool signature and every required parameter.",
            "PI controller on the keep-fraction with a squeeze term active only when fidelity "
            "exceeds the floor by more than the margin.",
        ],
        "validation": [
            "Compression achieved at equal fidelity, and floor-violation count, against "
            "fixed-ratio, tail-truncation and no-pruning baselines.",
            "Broken-signature count: the proposed controller must reach zero; fixed-ratio "
            "compression is expected not to.",
            "Empirical monotonicity check of fidelity against token budget.",
            "Falsifier: if the controller violates the fidelity floor more often than the fixed-"
            "ratio baseline at equal compression, the objective fails.",
        ],
    },
    {
        "id": "O5",
        "code": "INTEGRATION",
        "repo": "phd-agentic-ai-master",
        "module": "paim.pipeline",
        "title": "End-to-end integration, instrumentation and validation of the framework",
        "short": "Prove the four objectives compose into one loop, not four stages.",
        "statement": (
            "To integrate O1-O4 into a single orchestrator over a shared session state, to "
            "instrument the six feedback edges between them so that each can be shown to carry "
            "signal, and to validate that the resulting pipeline executes end to end on both "
            "the 10,439-endpoint ToolBench corpus and live data.gov.in Agriculture resources "
            "under one configuration surface."
        ),
        "questions": ["RQ5"],
        "sota": [
            {
                "work": "Agent frameworks (LangChain, AutoGen, LlamaIndex)",
                "does": "Provide composable components and a plumbing layer.",
                "limitation": "Composition is a cascade: components pass messages but share no "
                "state, so no component can adapt to another's outcome.",
            },
            {
                "work": "Published agent papers",
                "does": "Contribute one mechanism, evaluated in isolation.",
                "limitation": "Interaction effects between mechanisms are untested, and the "
                "reported gains do not compose.",
            },
        ],
        "novelty": [
            "A single shared session state that all four objectives read and write, making the "
            "framework a closed loop with six instrumented feedback edges rather than a pipeline.",
            "Every objective can be ablated to its SOTA analogue inside the same orchestrator, "
            "so baseline and proposed system are the same program under different flags.",
            "Two execution back-ends behind one tool interface: ToolBench schemas for ranking, "
            "and a live data.gov.in client for Agriculture answers. The lab never substitutes "
            "dummy mandi rows when the live call fails.",
            "A full per-stage trace (inputs, outputs, metrics, human-readable explanation, "
            "timing) emitted for every turn, which is what makes the pipeline auditable.",
            "ToolBench-Sessions: a label-preserving multi-turn benchmark derived from released "
            "qrels with no new annotation.",
        ],
        "method": [
            "Six-stage turn loop: session-state update, O1 retrieval and reranking, O4 context "
            "governance, O2 routing, O3 deliberation, credit assignment.",
            "Feedback edges: certificate to bandit reward; tool failure to endpoint penalty and "
            "session risk; abstention to forced escalation; surviving observations to session "
            "memory; committed tools to the pruner's hard-keep set; reranked tools to routing "
            "context.",
            "One configuration dataclass covering all four objectives plus the ablation switches.",
        ],
        "validation": [
            "Integration tests asserting each feedback edge changes a downstream decision: an "
            "injected endpoint failure must lower that endpoint's later rank, and an abstention "
            "must raise the tier of the subsequent routing decision.",
            "End-to-end runs on 100 ToolBench sessions and on the live Agriculture scenarios, "
            "with a complete stage trace for each turn.",
            "Cross-language parity between the Python reference implementation and the "
            "TypeScript port that serves the web demo, on committed golden vectors.",
            "Falsifier: if an edge can be removed with no change to any downstream decision, the "
            "integration claim is not supported for that edge.",
        ],
    },
]

OBJECTIVES_BY_ID = {o["id"]: o for o in OBJECTIVES}


def objective(objective_id: str) -> dict[str, Any]:
    return OBJECTIVES_BY_ID[objective_id.upper()]


def as_bundle() -> dict[str, Any]:
    """Serialisable snapshot consumed by the deck builder and the web app."""
    from paim.pipeline import FEEDBACK_EDGES, STAGES

    return {
        "title": TITLE,
        "overall": OVERALL_OBJECTIVE,
        "researchQuestions": RESEARCH_QUESTIONS,
        "objectives": OBJECTIVES,
        "stages": list(STAGES),
        "feedbackEdges": list(FEEDBACK_EDGES),
    }


__all__ = [
    "OBJECTIVES",
    "OBJECTIVES_BY_ID",
    "OVERALL_OBJECTIVE",
    "RESEARCH_QUESTIONS",
    "TITLE",
    "as_bundle",
    "objective",
]
