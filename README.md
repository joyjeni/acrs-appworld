# ACRS-AppWorld — certified, cost-aware tool agents on open models

**Ph.D. research (Computer Science and Engineering), M S Ramaiah University of Applied Sciences**  
Scholar: **Jenisha T** (24ETRP720001) · Supervisor: **Dr. Padma Priya Dharishini P**

ACRS (Adaptive Context Reasoning System) integrates four objectives and a governance stage in one agent loop. It is evaluated on **[AppWorld](https://github.com/StonyBrookNLP/appworld)** (Trivedi et al., ACL 2024 Best Resource Paper): 9 apps, 457 APIs, 750 tasks graded by state-based unit tests.

## Research gap

AppWorld's leaderboard top three reach 91–94 % task goal completion (TGC) on `test_normal` with frontier models. The best entry with a ≤14B open model reaches 86.9 %. Entries report accuracy only, which leaves these gaps open:

| Gap | Not yet studied on AppWorld | Objective |
|---|---|---|
| G1 | session-aware API retrieval from 457 APIs under a fixed context budget, for small models | **O1 SATR** |
| G2 | per-step routing between model tiers by risk, with cost reported | **O2 APRR** |
| G3 | guarantees on wrong or damaging state changes: outcome-level consensus (ARC-style execute-and-vote), certified thresholds, collateral damage, injection | **O3 MNCD + G1** |
| G4 | fidelity-controlled pruning of 30–60-step trajectories | **O4 FCNP** |
| G5 | all of the above in one loop on free or Colab-Pro GPUs | integration |

## Objectives (registered titles kept; O3 widened)

- **O1 Session-adaptive tool scoring:** `SessionApiRetriever`, using BM25 plus decayed app memory, auth completion and committed-API retention.
- **O2 Risk-weighted policy routing:** `TierRouter`, a risk-signed contextual Thompson router between a small and a large open model.
- **O3 Byzantine-tolerant mesh coordination → certified execution:** `OutcomeConsensus`. On state-changing steps it samples candidate programs, executes each from a saved world state, votes on the state changes they produce, and commits only a certified consensus; otherwise it abstains. It works with `AppWorldPolicy` (G1, policy as code: no unrequested deletion, scope) and the Learn-then-Test threshold in `paim.governance`.
- **O4 PID-controlled provenance pruning:** `HistoryGovernor`, a PI-controlled compression of the episode history with a fidelity floor on tokens, ids and errors.

Every objective has an `enabled` switch, and its disabled form is the SOTA analogue. One agent therefore produces the baseline, the full system and every leave-one-out ablation.

## Run

```bash
python -m pytest -q tests          # 40 offline tests (fake AppWorld world; no GPU)
```

On a GPU, open [`experiments/ACRS_appworld.ipynb`](experiments/ACRS_appworld.ipynb) in Colab (Pro) or Kaggle and choose Run all. It picks Qwen3-32B/14B/8B-AWQ by GPU memory, serves the models with vLLM, runs the `dev` pilot and ablations, and reports official TGC/SGC, collateral damage, cost, and McNemar/bootstrap tests.

## Protocol

- Tune on `train`/`dev` only. Report `test_normal` and `test_challenge` once per final configuration.
- Use the same backbone LLM for every configuration; ≥3 seeds for final runs.
- Metrics:
  - official TGC/SGC;
  - **collateral-damage rate** (tasks failing an AppWorld `no_op_pass` test);
  - tokens and cost;
  - router mix;
  - consensus certified/abstained counts;
  - policy denials;
  - history fidelity.
- Candidate exploration uses AppWorld's `save_state`/`load_state`. Executions count toward the interaction budget and are disclosed in every result.

## Licence and data

The code is MIT. AppWorld is Apache 2.0, and its task and API content may only be redistributed **encrypted**. This repository therefore never commits AppWorld data, episode logs or per-task evaluation reports (see `.gitignore`); only aggregate summaries are committed.

## Status

The ToolBench and KCC results for O1–O4 + G1 are in the companion repository [`phd-agentic-ai-master`](https://github.com/joyjeni/phd-agentic-ai-master) (branch `acrs-v3-governed`). No AppWorld numbers are claimed here until the notebook has run.
