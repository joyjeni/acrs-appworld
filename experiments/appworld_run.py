"""Run the ACRS agent (or an ablation) on an AppWorld split, then score it officially.

Example (after starting the vLLM servers, see experiments/ACRS_appworld.ipynb):

    python experiments/appworld_run.py --split dev --config full --workers 8 \
        --large Qwen/Qwen3-14B-AWQ@http://localhost:8001/v1 \
        --small Qwen/Qwen3-4B-AWQ@http://localhost:8002/v1

Configs: baseline (ReAct-style, docs discovered by the agent), full, and the
leave-one-out ablations no_o1 / no_o2 / no_o3 / no_o4 / no_g1.
Official TGC/SGC come from appworld.evaluator.evaluate_dataset; the per-episode
log adds cost, router decisions, consensus counts, policy denials and fidelity.
Rules followed: tune on train/dev only; test splits are run once per final config.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paim.appworld import ACRSAppWorldAgent, AgentConfig, ChatClient, ModelTier  # noqa: E402

CONFIGS = {
    "baseline": replace(AgentConfig.baseline(), k_apis=0),
    "full": AgentConfig(),
    "no_o1": AgentConfig(o1_retrieval=False),
    "no_o2": AgentConfig(o2_routing=False),
    "no_o3": AgentConfig(o3_consensus=False),
    "no_o4": AgentConfig(o4_history=False),
    "no_g1": AgentConfig(g1_policy=False),
}


def tier(spec: str, name: str) -> ModelTier:
    model, url = spec.split("@", 1)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in model.lower() else {}
    return ModelTier(name=model, base_url=url, tier=name, extra_body=extra)


def worker(args_tuple):
    task_ids, cfg_name, exp, small_spec, large_spec, seed, out_path = args_tuple
    from appworld import AppWorld

    cfg = CONFIGS[cfg_name]
    small = ChatClient(tier(small_spec, "small"))
    large = ChatClient(tier(large_spec, "large"))
    rows = []
    for tid in task_ids:
        small.usage.__init__()
        large.usage.__init__()
        t0 = time.time()
        try:
            with AppWorld(task_id=tid, experiment_name=exp, max_interactions=cfg.max_steps * (cfg.consensus_n + 1) + 10) as world:
                log = ACRSAppWorldAgent(world, small, large, cfg, seed=seed).run()
                row = log.to_dict()
        except Exception as e:  # fail closed: record, never crash the batch
            row = {"task_id": tid, "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}
        rows.append(row)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test_normal", "test_challenge"])
    ap.add_argument("--config", default="full", choices=list(CONFIGS))
    ap.add_argument("--small", required=True, help="model@base_url")
    ap.add_argument("--large", required=True, help="model@base_url")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from appworld import load_task_ids

    ids = load_task_ids(args.split)
    if args.limit:
        ids = ids[: args.limit]
    exp = f"acrs_{args.config}{('_' + args.tag) if args.tag else ''}_s{args.seed}_{args.split}"
    out_dir = ROOT / "experiments/results/appworld"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{exp}.episodes.jsonl"
    if log_path.exists():
        log_path.unlink()
    chunks = [ids[i:: args.workers] for i in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        pool.map(worker, [(c, args.config, exp, args.small, args.large, args.seed, str(log_path)) for c in chunks if c])
    wall = time.time() - t0

    from appworld.evaluator import evaluate_tasks

    # Official scorer. On a full split this equals `appworld evaluate <exp> <split>`;
    # per-task details are kept for paired tests (McNemar / bootstrap) between configs.
    report = evaluate_tasks(ids, experiment_name=exp, include_details=True)
    (out_dir / f"{exp}.evaluation.json").write_text(json.dumps(report, default=str))
    # Collateral damage: a failed test labelled "no_op_pass" passes when the agent
    # does nothing, so failing it means the agent changed something it should not have.
    indiv = report.get("individual", {})
    labelled = [f for t in indiv.values() for f in t.get("failures", []) if "label" in f]
    collateral = (
        sum(any(f.get("label") == "no_op_pass" for f in t.get("failures", [])) for t in indiv.values()) / max(1, len(indiv))
        if labelled or not indiv else None
    )
    eps = [json.loads(l) for l in open(log_path)]
    ok = [e for e in eps if "error" not in e]
    summary = {
        "experiment": exp, "split": args.split, "config": args.config, "tasks": len(ids),
        "official": report.get("aggregate", {}),
        "episodesFailed": len(eps) - len(ok),
        "collateralDamageRate": collateral,
        "meanSteps": sum(e["steps"] for e in ok) / max(1, len(ok)),
        "policyDenials": sum(e["policy_denials"] for e in ok),
        "consensus": {k: sum(e["consensus"].get(k, 0) for e in ok) for k in
                      ("consensus_steps", "certified", "abstain_disagree", "abstain_all_failed", "candidate_executions")},
        "router": {k: sum(e["router"].get(k, 0) for e in ok) for k in ("small", "large")},
        "tokens": sum(e["usage"]["prompt_tokens"] + e["usage"]["completion_tokens"] for e in ok),
        "meanFidelity": sum(e["fidelity_mean"] for e in ok) / max(1, len(ok)),
        "wallSeconds": wall,
        "models": {"small": args.small.split("@")[0], "large": args.large.split("@")[0]},
    }
    (out_dir / f"{exp}.summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
