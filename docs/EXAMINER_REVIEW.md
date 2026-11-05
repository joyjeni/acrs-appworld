# Mock examiner report (IISc-style review), 5 November 2026

*Written as a demanding external examiner would. The purpose is to find every weakness before the real
examiners do.*

**Thesis:** Adaptive Context Reasoning System (ACRS): Session-Adaptive Tool Scoring, Risk-Weighted Policy
Routing, Byzantine-Tolerant Mesh Coordination, and PID-Controlled Provenance Pruning in Multi-Agent LLM
Pipelines. Jenisha T, MSRUAS.

## Recommendation today

**Major revision; do not submit yet.** The architecture is coherent and the experimental discipline is
better than typical: the leakage audit, held-out thresholds, a power analysis, and negative results that
are reported. But most of the evidence still comes from simulated agents. The AppWorld layer must
produce results with real open-weight models before the thesis can be defended. Once it does, and the
items below are addressed, the work is defendable for award.

## Contribution by contribution

| # | Contribution | Novelty | Evidence now | Examiner's main objection | What makes it defendable |
|---|---|---|---|---|---|
| O1 | Session-aware, training-free tool retrieval | **Moderate.** Session context in retrieval exists (e.g. COLT, ToolRet line); training-free plus leakage-safe is the angle | +12.4 % NDCG@5 on ToolBench (CV, CI excludes 0); **hurts on KCC** (0.412 → 0.397) | "An incremental reranker that fails on your own public dataset." | Frame it as *when* session signals help: report the KCC failure as a scope result; show gains on AppWorld multi-step episodes; compare with a ToolRet-trained dense retriever |
| O2 | Risk-signed bandit routing between agents or model tiers | **Low–moderate.** LinUCB/Thompson with a risk term; the regret bound is inherited | Lowest regret of six routers; −74 % cost alone, but errors rise | "The theory is LinUCB's; the router alone makes answers worse." | Show cost–quality Pareto gains with *real* model tiers on AppWorld (small vs large Qwen3), and compare with a RouteLLM-style router; claim the coupling with O3 (cheap routing made safe), not the bandit |
| O3 | Evidence-weighted Byzantine quorum → **certified execution** (outcome consensus, Learn-then-Test threshold, policy-as-code) | **Highest.** Certifying *actions* by executing candidate programs and voting on outcomes, with a finite-sample bound on executed error, is not established for tool agents | Accepted errors 24 % → 0.6 % (ToolBench); injection 29.3 % → 5.7 % (ToolBench) and 27.8 % → 0.86 % (KCC); α = 5 % certified with held-out risk ≤ α | "Save/restore of world state is unrealistic outside a simulator"; "the LTT guarantee is an application, not new theory"; "abstention is 21 %" | State the deployment analogue (sandbox or dry-run, or approve-before-execute); show reduced **collateral damage** on AppWorld with McNemar significance; report the review/abstention cost honestly; position LTT as the first use for agent action certification |
| O4 | Typed-block pruning with PI fidelity control | **Moderate.** The controller is simple; fidelity is self-defined | Floor met on 99.7 % of turns at 47 % compression | "The fidelity metric is yours; does the *task* still succeed?" | On AppWorld, report task success (TGC) versus tokens against recency and LLMLingua-style baselines |
| Int. | The closed loop + KCC public problem + governance | **Good.** The integration contract, governance invariants and a public benchmark built without inventing labels | KCC: executed errors 11.3 % → 0.51 % | "Simulated agents throughout"; "pseudo-sessions" | AppWorld results with real agents; state the pseudo-session limitation |

## Issues the examiners will raise regardless

1. **Title accuracy.** The title says *PID-controlled*, but the implementation is PI. It says *mesh
   coordination*, but gossip is advisory and the certificate does not depend on it. Either change the
   title wording (needs a committee request) or justify the derivative term and the mesh's role
   explicitly in Chapter 6.
2. **Real agents.** Every simulated-agent number needs a real-agent counterpart, even on a subset. This
   is the single most important gap.
3. **Baselines on the same model.** On AppWorld, compare against the reference ReAct agent on the *same*
   Qwen3 model. Leaderboard frontier numbers (91–94 %) are context, not the comparison.
4. **Statistics.** Use ≥3 seeds for final runs; McNemar and bootstrap on per-task success; one primary
   comparison per objective, declared before running the test splits.
5. **Publications.** MSRUAS D.8.4 requires two refereed journal papers and two conference presentations
   before submission; none are submitted yet. The first submission should be O3 (certified execution)
   to *Autonomous Agents and Multi-Agent Systems*.
6. **Reproducibility and licence.** AppWorld content must never be committed unencrypted. This is
   already enforced in `.gitignore`; say so in the thesis.
7. **Similarity check.** Keep the thesis similarity index under 10 % (MSRUAS D.8.4). Reusing text from
   your own papers counts.

## Bottom line

The defendable core of the thesis is **certified, governed execution for tool agents (O3 + G1),
enabled by the other three objectives and shown on a public problem**. O1, O2 and O4 are defendable as
enabling mechanisms with measured scope, not as stand-alone breakthroughs. Once the AppWorld experiments
are complete, with real open models, same-model baselines, collateral damage and significance tests, an
examiner can recommend the award.
