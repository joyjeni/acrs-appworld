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

Kaggle sessions stop at 12 hours: ACRS_DEADLINE (Unix seconds) stops workers between tasks, and --resume keeps
the episode log and skips tasks already in it. Each train/dev episode records AppWorld's own per-task test result,
so a resumed run can be scored even when an earlier session's app databases are gone.

--few-shot K puts K worked examples from the train split in every config's prompt (paim.appworld.demos); the
experiment name records K, so runs with and without examples never share an episode log.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
from collections import Counter
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# AppWorld resolves data/ and experiments/outputs/ against the working directory; keep them out of
# the repo (whose data/ holds the ToolBench bundle) when APPWORLD_ROOT is set.
if os.environ.get("APPWORLD_ROOT"):
    os.chdir(os.environ["APPWORLD_ROOT"])

from paim.appworld import ACRSAppWorldAgent, AgentConfig, ChatClient, ModelTier  # noqa: E402
from paim.appworld.demos import data_dir, render_demos, select_demos  # noqa: E402
from paim.common import clock  # noqa: E402  (imported before AppWorld freezes time)

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


def experiment_name(config: str, tag: str, seed: int, split: str, few_shot: int = 0) -> str:
    """AppWorld experiment name; includes the number of worked examples when there are any."""
    return f"acrs_{config}{('_' + tag) if tag else ''}{f'_fs{few_shot}' if few_shot else ''}_s{seed}_{split}"


def deadline_reached() -> bool:
    """True once the real wall clock passes ACRS_DEADLINE (Unix seconds); unset means no deadline."""
    raw = os.environ.get("ACRS_DEADLINE", "")
    return bool(raw) and clock.wall_time() >= float(raw)


def task_result(world) -> dict:
    """AppWorld's own tests for this task (train/dev only). Empty if the evaluator is unavailable."""
    try:
        tracker = world.evaluate()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for name in ("success", "pass_count", "fail_count", "num_tests"):
        val = getattr(tracker, name, None)
        if isinstance(val, (bool, int, float)):
            out[name] = val
    if "success" not in out and hasattr(tracker, "to_dict"):
        d = tracker.to_dict()
        if isinstance(d, dict) and isinstance(d.get("success"), bool):
            out["success"] = d["success"]
    return out


def worker(args_tuple):
    task_ids, cfg_name, exp, small_spec, large_spec, seed, out_path, context_limit, record, demos = args_tuple
    from appworld import AppWorld

    cfg = replace(CONFIGS[cfg_name], context_limit_tokens=context_limit)
    small = ChatClient(tier(small_spec, "small"))
    large = ChatClient(tier(large_spec, "large"))
    rows = []
    for tid in task_ids:
        if deadline_reached():
            break
        small.usage.__init__()
        large.usage.__init__()
        t0 = clock.wall_time()
        try:
            with AppWorld(task_id=tid, experiment_name=exp, max_interactions=cfg.max_steps * (cfg.consensus_n + 1) + 10) as world:
                try:
                    log = ACRSAppWorldAgent(world, small, large, cfg, seed=seed, demos=demos).run()
                    row = log.to_dict()
                finally:
                    # Always persist the end state, so the official evaluator can score the episode.
                    if hasattr(world, "save"):
                        world.save()
                if record:
                    row.update(task_result(world))
        except Exception as e:  # fail closed: record, never crash the batch
            row = {"task_id": tid, "error": f"{type(e).__name__}: {e}", "seconds": clock.wall_time() - t0}
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
    ap.add_argument("--context-limit", type=int, default=0,
                    help="prompt limit in tokens for every config (server max model length); 0 = off")
    ap.add_argument("--resume", action="store_true", help="keep the episode log and skip tasks already in it")
    ap.add_argument("--few-shot", type=int, default=0,
                    help="worked examples from the train split in every prompt (same for every config); 0 = none")
    args = ap.parse_args()

    from appworld import load_task_ids

    ids = load_task_ids(args.split)
    if args.limit:
        ids = ids[: args.limit]
    demos = select_demos(data_dir(), args.few_shot) if args.few_shot else []
    if args.few_shot and len(demos) < args.few_shot:
        sys.exit(f"--few-shot {args.few_shot}: only {len(demos)} usable train examples under {data_dir().resolve()} "
                 "(see experiments/check_demos.py)")
    demo_text = render_demos(demos)
    exp = experiment_name(args.config, args.tag, args.seed, args.split, args.few_shot)
    out_dir = ROOT / "experiments/results/appworld"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{exp}.episodes.jsonl"
    if log_path.exists() and not args.resume:
        log_path.unlink()
    done = {json.loads(l)["task_id"] for l in open(log_path)} if log_path.exists() else set()
    todo = [t for t in ids if t not in done]
    record = args.split in ("train", "dev")
    chunks = [todo[i:: args.workers] for i in range(args.workers)]
    t0 = clock.wall_time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        pool.map(worker, [(c, args.config, exp, args.small, args.large, args.seed, str(log_path), args.context_limit,
                           record, demo_text) for c in chunks if c])
    wall = clock.wall_time() - t0
    eps = [json.loads(l) for l in open(log_path)] if log_path.exists() else []
    attempted = [t for t in ids if t in {e["task_id"] for e in eps}]   # a deadline can leave tasks unattempted
    recorded = {e["task_id"]: bool(e["success"]) for e in eps if isinstance(e.get("success"), bool)}

    from appworld.evaluator import evaluate_tasks

    # Official scorer. On a full split this equals `appworld evaluate <exp> <split>`;
    # per-task details are kept for paired tests (McNemar / bootstrap) between configs.
    # Fail closed: a task whose end state was not saved (crashed worker) is scored as failed, not dropped.
    out_root = Path(os.environ.get("APPWORLD_ROOT", ".")) / "experiments/outputs" / exp / "tasks"
    scorable = [t for t in attempted if (out_root / t / "dbs").exists()]
    missing = [t for t in attempted if t not in scorable]
    report = evaluate_tasks(scorable, experiment_name=exp, include_details=True) if scorable else {"aggregate": {}, "individual": {}}
    for t in missing:
        # No saved end state in this session: use the result recorded when the episode ran (a resumed run),
        # otherwise fail closed.
        report.setdefault("individual", {})[t] = {"success": recorded.get(t, False), "missing_output": True,
                                                  "source": "episode_log" if t in recorded else "none", "failures": []}
    if missing:
        # Official aggregate covers the scorable tasks only; recompute both rates over every attempted task.
        indiv = report["individual"]
        scen: dict = {}
        for t in attempted:
            scen.setdefault(t.split("_")[0], []).append(bool(indiv.get(t, {}).get("success")))
        report["aggregate_official_subset"] = report.get("aggregate", {})
        report["aggregate"] = {        # percent, like AppWorld's own aggregate
            "task_goal_completion": 100.0 * sum(bool(indiv.get(t, {}).get("success")) for t in attempted) / max(1, len(attempted)),
            "scenario_goal_completion": 100.0 * sum(all(v) for v in scen.values()) / max(1, len(scen)),
        }
    (out_dir / f"{exp}.evaluation.json").write_text(json.dumps(report, default=str))
    # Collateral damage: a failed test labelled "no_op_pass" passes when the agent
    # does nothing, so failing it means the agent changed something it should not have.
    indiv = report.get("individual", {})
    labelled = [f for t in indiv.values() for f in t.get("failures", []) if "label" in f]
    collateral = (
        sum(any(f.get("label") == "no_op_pass" for f in t.get("failures", [])) for t in indiv.values()) / max(1, len(indiv))
        if labelled or not indiv else None
    )
    ok = [e for e in eps if "error" not in e]
    summary = {
        "experiment": exp, "split": args.split, "config": args.config, "tasks": len(ids),
        "tasksAttempted": len(attempted), "stoppedByDeadline": len(attempted) < len(ids),
        "contextLimit": args.context_limit, "windowDropped": sum(e.get("window_dropped", 0) for e in ok),
        "fewShot": args.few_shot, "demoTasks": [d.task_id for d in demos],
        "official": report.get("aggregate", {}),
        "episodesFailed": len(eps) - len(ok),
        "missingOutputs": len(missing),
        "collateralDamageRate": collateral,
        "meanSteps": sum(e["steps"] for e in ok) / max(1, len(ok)),
        # Diagnostics: why episodes ended, and how often an executed step raised an error.
        "stopReasons": {r: sum(e.get("stop_reason") == r for e in ok) for r in sorted({e.get("stop_reason", "") for e in ok})},
        "errorStepRate": sum(e.get("errors", 0) for e in ok) / max(1, sum(e["steps"] for e in ok)),
        "episodeErrors": sorted({e["error"][:160] for e in eps if "error" in e})[:5],
        # Most frequent error messages in executed steps (digits masked so similar errors group).
        "topStepErrors": Counter(re.sub(r"\d+", "#", m) for e in ok for m in e.get("error_samples", [])).most_common(10),
        "policyDenials": sum(e["policy_denials"] for e in ok),
        "consensus": {k: sum(e["consensus"].get(k, 0) for e in ok) for k in
                      ("consensus_steps", "certified", "abstain_disagree", "abstain_all_failed", "candidate_executions",
                       "exempt_steps")},
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
