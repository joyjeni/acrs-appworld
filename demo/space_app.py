"""ACRS public demo (Hugging Face Space).

Three tabs:
  1. Results   - the measured numbers from experiments (ToolBench, KCC); AppWorld marked pending.
  2. Live run  - the integrated O1-O4 + G1 loop on a real sessionised ToolBench query, on CPU,
                 with each objective switchable and an optional prompt-injection attack.
  3. Scope     - what is and is not claimed.

Agents in the live run are simulated (seeded, schema-faithful); the demo shows the orchestration
mechanisms, not LLM quality. No AppWorld task content is used or shown (licence).
"""
from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from paim.data.toolbench import build_sessions, load_bundle  # noqa: E402
from paim.governance import InjectingExecutor  # noqa: E402
from paim.o3_mncd.execution import SimulatedToolExecutor  # noqa: E402
from paim.pipeline.orchestrator import Ablations, IntegratedPipeline, PipelineConfig  # noqa: E402

# --------------------------------------------------------------------------------------------
# Measured results (copied from experiments/RESULTS.md of joyjeni/phd-agentic-ai-master; 3 seeds)
# --------------------------------------------------------------------------------------------
TOOLBENCH_ABLATION = pd.DataFrame(
    [
        ["SOTA baseline", 0.519, 24.0, 0.5, 0.0309, 11],
        ["+ O1 SATR", 0.584, 24.6, 0.3, 0.0308, 13],
        ["+ O2 APRR", 0.519, 30.5, 0.1, 0.0080, 11],
        ["+ O3 MNCD", 0.519, 0.6, 23.1, 0.0321, 11],
        ["+ O4 FCNP", 0.519, 21.6, 0.0, 0.0306, 100],
        ["Full ACRS (O1-O4)", 0.587, 2.0, 21.3, 0.0141, 100],
    ],
    columns=["Configuration", "NDCG@5", "Accepted errors %", "Abstention %", "Cost / session", "Fidelity floor met %"],
)

KCC_ABLATION = pd.DataFrame(
    [
        ["SOTA baseline", 0.412, 99.7, 11.3],
        ["+ O1 SATR", 0.397, 99.3, 11.7],
        ["+ O2 APRR", 0.412, 99.7, 26.4],
        ["+ O3 MNCD", 0.412, 89.5, 0.34],
        ["+ O4 FCNP", 0.412, 99.8, 6.2],
        ["Full ACRS (O1-O4)", 0.399, 85.9, 0.86],
        ["Full ACRS + G1", 0.399, 53.3, 0.51],
    ],
    columns=["Configuration", "NDCG@5", "Auto-executed %", "Executed errors %"],
)

INJECTION = pd.DataFrame(
    [
        ["ToolBench, 15 % endpoints poisoned", 29.3, 5.7],
        ["ToolBench, 30 % endpoints poisoned", 45.4, 6.3],
        ["KCC farmer queries, 15 % poisoned", 27.8, 0.86],
    ],
    columns=["Setting", "Attack success without G1 %", "Attack success with G1 %"],
)

HEADLINES = """
| Objective | What it does | Headline result (held-out, 3 seeds) |
|---|---|---|
| **O1 SATR**: session-adaptive tool scoring | ranks tools using what the session already used, training-free | NDCG@5 +12.4 % over BM25 on ToolBench (95 % CI excludes 0); does **not** transfer to KCC pseudo-sessions (scope limit) |
| **O2 APRR**: risk-weighted policy routing | contextual bandit choosing cheap vs strong agents, exploration signed by risk | cost -74 %, lowest regret of six routers; alone it raises errors, so it is paired with O3 |
| **O3 MNCD**: Byzantine-tolerant consensus to certified execution | certifies an answer/action only when independent agents agree on grounded evidence; threshold from Learn-then-Test | accepted errors 24 % to 0.6 %; certified at alpha = 5 % with held-out risk 3.0 % or less |
| **O4 FCNP**: PI-controlled provenance pruning | compresses context under a budget while holding a fidelity floor | floor met on 99.7 % of turns while removing 47 % of tokens |
| **G1** governance | deny-by-default policy-as-code, injection guard, tamper-evident decision log | injection success 29.3 % to 5.7 % (ToolBench), 27.8 % to 0.86 % (KCC) |
"""

APPWORLD_NOTE = """
### AppWorld with real open-weight agents: running
ACRS is being evaluated on **AppWorld** (ACL 2024; 9 apps, 457 APIs) with **Qwen3-8B and Qwen3-4B** served by vLLM on a free
Kaggle 2xT4 GPU, against a reference agent on the **same models**, scored by AppWorld's official evaluator
(task/scenario goal completion, collateral damage) with McNemar tests. Numbers will appear here when the runs finish;
none are shown before then.
"""

SCOPE = """
### What this demo shows
* The **orchestration mechanisms** of ACRS (O1-O4 + G1) running end to end on a real ToolBench query and catalogue (10,439 APIs).
* How each objective changes the outcome: switch any of them off and compare.
* What happens under a **prompt-injection attack** on tool outputs.

### What it does not claim
* **Agents in the live run are simulated** (seeded and schema-faithful), so the numbers reflect the mechanisms, not LLM quality.
  Real-model results come from the AppWorld experiments.
* KCC sessions are pseudo-sessions built from real Kisan Call Centre queries; most KCC services are simulated.
* The injection harness is in-house; AgentDojo is the next external test.
* Frontier systems lead the AppWorld leaderboard; ACRS is compared with a reference agent on the same open model.

### Links
* Code: [github.com/joyjeni/acrs-appworld](https://github.com/joyjeni/acrs-appworld) and
  [github.com/joyjeni/phd-agentic-ai-master](https://github.com/joyjeni/phd-agentic-ai-master)
* Author: Jenisha T, PhD scholar, M S Ramaiah University of Applied Sciences (24ETRP720001)
"""

# --------------------------------------------------------------------------------------------
# Live run
# --------------------------------------------------------------------------------------------
STAGE_NAMES = {
    "S0": "Session state", "S1": "O1 tool retrieval + session rerank", "S2": "O2 routing",
    "S3": "O3 deliberation + consensus", "S4": "O4 context pruning", "G1": "G1 governance", "S5": "Answer",
}


@lru_cache(maxsize=1)
def _bundle():
    b = load_bundle(ROOT / "data")
    gold = {frozenset(q["gold_doc_ids"]) for q in b.queries}
    traj = [t for t in b.train_trajectories if frozenset(t) not in gold]  # leak-filtered, as in the thesis
    return b, traj


def session_choices() -> list[str]:
    b, _ = _bundle()
    out = []
    for i, s in enumerate(build_sessions(b)):
        q = " ".join(s.query.split())
        out.append(f"{i:03d} | {len(s.sub_goals)} step(s) | {q[:110]}")
    return out


def _tool_label(b, doc_id: str) -> str:
    t = b.tools.get(doc_id)
    return f"{t.tool_name} / {t.api_name}" if t else doc_id


def run_live(choice: str, o1: bool, o2: bool, o3: bool, o4: bool, g1: bool, attack_pct: float, seed: int = 20260905):
    """Run one session; return (summary markdown, per-step table, decision-log json)."""
    b, traj = _bundle()
    idx = int(choice.split("|", 1)[0]) if choice else 0
    session = build_sessions(b)[idx]
    abl = Ablations(o1, o2, o3, o4, g1)
    cfg = PipelineConfig(seed=int(seed), ablations=abl)
    ex = None
    if attack_pct > 0:
        ex = InjectingExecutor(
            SimulatedToolExecutor(seed=cfg.seed, failure_rate=cfg.tool_failure_rate), rate=attack_pct / 100.0, seed=cfg.seed
        )
    pipe = IntegratedPipeline(b.tools, traj, cfg, executor=ex)
    trace = pipe.run(session)
    m = trace.metrics
    rows, log = [], []
    prev = "0" * 16
    for r, sg in zip(pipe.last_records, session.sub_goals):
        gold = set(r["gold"])
        hit = any(d in gold for d in r["selected"])
        outcome = (
            "executed, correct" if r["executedCorrect"] else
            "executed, WRONG" if r["executedWrong"] else
            "held for human review" if r["disposition"] == "REVIEW" else
            "blocked" if r["disposition"] == "BLOCK" else "not executed"
        )
        if r.get("attackSucceeded"):
            outcome += " (ATTACK SUCCEEDED)"
        rows.append({
            "Step": r["turn"],
            "Sub-goal": " ".join(sg.text.split())[:90],
            "O1 top tools": "; ".join(_tool_label(b, d) for d in r["selected"][:3]),
            "Gold tool in top 3": "yes" if hit else "no",
            "O2 agent tier": r["tier"],
            "O3 agreement": round(float(r.get("weightedShare") or 0.0), 2),
            "O3 certified": "yes" if r["certified"] else "abstained" if r["abstained"] else "no",
            "O4 fidelity": round(float(r["fidelity"]), 2),
            "O4 tokens kept": f'{r["keptTokens"]}/{r["originalTokens"]}',
            "Poisoned peers": r.get("hijackedPeers", 0),
            "G1 decision": r["disposition"],
            "Outcome": outcome,
        })
        entry = {"step": r["turn"], "decision": r["disposition"], "certified": r["certified"],
                 "executed": r["executed"], "prev": prev}
        prev = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
        log.append({**entry, "hash": prev})

    def pct(x):
        return f"{100 * float(x):.1f} %" if x is not None else "n/a"

    summary = f"""
**Query:** {" ".join(session.query.split())}

**Configuration:** {abl.label()}; poisoned tool outputs: {attack_pct:.0f} %

| Retrieval NDCG@5 | Auto-executed | Executed wrong | Attack success | Cost | Context tokens removed | Fidelity floor met |
|---|---|---|---|---|---|---|
| {m.get("ndcg@5", float("nan")):.3f} | {pct(m.get("executeRate"))} | {pct(m.get("executedErrorRate"))} | {pct(m.get("attackSuccessRate"))} | {m.get("totalCost", 0):.4f} | {pct(m.get("compression"))} | {pct(m.get("floorSatisfactionRate"))} |

*Simulated agents: one session is a single draw. The Results tab gives the averages over all sessions and seeds.*
"""
    return summary, pd.DataFrame(rows), log


def compare_all(n_sessions: float, attack_pct: float):
    """Average over the first n sessions: SOTA baseline, each objective alone, full ACRS, full ACRS + G1."""
    configs = [
        ("SOTA baseline", (False, False, False, False, False)),
        ("+ O1 only", (True, False, False, False, False)),
        ("+ O2 only", (False, True, False, False, False)),
        ("+ O3 only", (False, False, True, False, False)),
        ("+ O4 only", (False, False, False, True, False)),
        ("Full ACRS (O1-O4)", (True, True, True, True, False)),
        ("Full ACRS + G1", (True, True, True, True, True)),
    ]
    b, traj = _bundle()
    n = max(1, min(100, int(n_sessions)))
    out = []
    for name, flags in configs:
        cfg = PipelineConfig(ablations=Ablations(*flags))
        ex = InjectingExecutor(SimulatedToolExecutor(seed=cfg.seed, failure_rate=cfg.tool_failure_rate),
                               rate=attack_pct / 100.0, seed=cfg.seed) if attack_pct > 0 else None
        pipe = IntegratedPipeline(b.tools, traj, cfg, executor=ex)
        recs, ms = [], []
        for s in build_sessions(b)[:n]:
            ms.append(pipe.run(s).metrics)
            recs.extend(pipe.last_records)
        k = len(recs)
        ex_n = sum(r["executed"] for r in recs)
        nd = [m["ndcg@5"] for m in ms if "ndcg@5" in m]
        out.append({
            "Configuration": name,
            "NDCG@5": round(sum(nd) / len(nd), 3) if nd else None,
            "Auto-executed %": round(100 * ex_n / k, 1),
            "Executed wrong % (of steps)": round(100 * sum(r["executedWrong"] for r in recs) / k, 1),
            "Attack success % (of steps)": round(100 * sum(r["attackSucceeded"] for r in recs) / k, 1),
            "Held for review %": round(100 * sum(r["disposition"] == "REVIEW" for r in recs) / k, 1),
            "Cost / session": round(sum(m.get("totalCost", 0) for m in ms) / len(ms), 4),
            "Tokens removed %": round(100 * sum(m.get("compression", 0) for m in ms) / len(ms), 1),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    choices = session_choices()
    with gr.Blocks(title="ACRS: certified, governed tool agents") as demo:
        gr.Markdown(
            "# ACRS: Adaptive Context Reasoning System\n"
            "Session-adaptive tool scoring (O1), risk-weighted routing (O2), Byzantine-tolerant consensus with "
            "certified execution (O3), PI-controlled context pruning (O4) and policy-as-code governance (G1) "
            "in one closed loop. Jenisha T, M S Ramaiah University of Applied Sciences."
        )
        with gr.Tab("Results"):
            gr.Markdown(HEADLINES)
            gr.Markdown("### ToolBench: one objective at a time vs the full loop (100 sessions, 3 seeds, leak-filtered)")
            gr.Dataframe(TOOLBENCH_ABLATION, interactive=False)
            gr.BarPlot(TOOLBENCH_ABLATION, x="Configuration", y="Accepted errors %",
                       title="Accepted errors (lower is better)")
            gr.Markdown("### KCC-Sessions: real Kisan Call Centre farmer queries (586 queries, 15 states)")
            gr.Dataframe(KCC_ABLATION, interactive=False)
            gr.BarPlot(KCC_ABLATION, x="Configuration", y="Executed errors %",
                       title="Executed errors (lower is better)")
            gr.Markdown("### Prompt injection through tool outputs")
            gr.Dataframe(INJECTION, interactive=False)
            gr.Markdown(APPWORLD_NOTE)

        with gr.Tab("Live run"):
            gr.Markdown("Pick a real ToolBench request, switch objectives on or off, optionally poison tool outputs, and run. "
                        "Try the default session with **G1 off**, then **on**: the poisoned step is executed, then held for review.")
            sess = gr.Dropdown(choices, value=choices[25] if len(choices) > 25 else choices[0], label="Session")
            with gr.Row():
                o1 = gr.Checkbox(True, label="O1 session-adaptive retrieval")
                o2 = gr.Checkbox(True, label="O2 risk-weighted routing")
                o3 = gr.Checkbox(True, label="O3 consensus / certified execution")
                o4 = gr.Checkbox(True, label="O4 fidelity-controlled pruning")
                g1 = gr.Checkbox(True, label="G1 governance")
            attack = gr.Slider(0, 50, value=15, step=5, label="Poisoned tool outputs (%)")
            go = gr.Button("Run", variant="primary")
            summary = gr.Markdown()
            steps = gr.Dataframe(label="Per step: what each objective decided", interactive=False, wrap=True)
            log = gr.JSON(label="G1 decision log (hash-chained)")
            go.click(run_live, [sess, o1, o2, o3, o4, g1, attack], [summary, steps, log])
            gr.Markdown("### Every configuration, averaged over many sessions (about 10 s for 30 sessions)")
            n_sess = gr.Slider(5, 100, value=30, step=5, label="Number of sessions")
            cmp_btn = gr.Button("Compare all configurations")
            cmp_out = gr.Dataframe(interactive=False, wrap=True)
            cmp_btn.click(compare_all, [n_sess, attack], cmp_out)

        with gr.Tab("Scope and limits"):
            gr.Markdown(SCOPE)
    return demo


if __name__ == "__main__":
    build_ui().launch()
