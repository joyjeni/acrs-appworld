"""Offline replay of recorded AppWorld trajectories: the history O4 would have shown (CPU only, no model).

    APPWORLD_ROOT=/path/to/appworld_root python experiments/replay_o4.py \\
        --tasks-dir /path/to/experiments/outputs/acrs_baseline_s0_dev/tasks [--limit 20] [--out replay.json]

For each task of a recorded run, the executed code blocks are read from AppWorld's own log
(``tasks/<id>/logs/environment_io.md``) and executed again in a fresh world. That rebuilds the interpreter state
step by step, as the agent had it. After every step the script measures the history each policy would put in the
next prompt for the same actions:

- ``window``: the baseline's last six turns, verbatim, each output cut at 6,000 characters;
- ``fidelity``: the first O4 (``HistoryGovernor``);
- ``provenance``: the current O4 (``ProvenanceGovernor`` plus the variable ledger).

This measures compression on real AppWorld observations for fixed actions. It does not measure how the agent would
act with the smaller context; that needs the model (the GPU run). Use a run without O3 (for example ``baseline``):
O3's trial executions are in the same log and would be replayed as if committed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if os.environ.get("APPWORLD_ROOT"):
    os.chdir(os.environ["APPWORLD_ROOT"])

from paim.appworld.catalog import is_error  # noqa: E402
from paim.appworld.objectives import HistoryGovernor  # noqa: E402
from paim.appworld.provenance import InterpreterState, ProvenanceGovernor, ProvenanceLedger  # noqa: E402


def size(history: list[tuple[str, str]], extra: str = "") -> int:
    """Estimated tokens of the history messages, as the agent builds them (characters / 4)."""
    chars = sum(len(f"```python\n{code}\n```") + len(f"Output:\n{out[:6000]}") for code, out in history)
    return (chars + (len(extra) + 2 if extra else 0)) // 4


def replay_task(task_dir: Path, experiment: str) -> dict:
    from appworld import AppWorld

    tid = task_dir.name
    io = AppWorld.parse_environment_io_log(file_path=str(task_dir / "logs" / "environment_io.md"))
    window, fidelity, provenance, phis, window_phis = [], [], [], [], []
    same = 0
    with AppWorld(task_id=tid, experiment_name=experiment) as world:
        state = InterpreterState(world)
        ledger = ProvenanceLedger(state, world.task.instruction)
        legacy = HistoryGovernor()
        governor = ProvenanceGovernor()
        turns, metas = [], []
        for step, entry in enumerate(io):
            code = entry["input"]
            output = world.execute(code)
            same += int(output.strip() == (entry.get("output") or "").strip())
            err = is_error(output)
            turns.append((code, output))
            metas.append(ledger.record(step, code, output, err))
            window.append(size(turns[-6:]))
            fidelity.append(size(legacy.compress(turns)[0]))
            history, phi, _, summary = governor.compress(turns, metas, ledger)
            provenance.append(size(history, ledger.render() + summary))
            phis.append(phi)
            # The window shows the last six turns and nothing about what variables hold.
            big = [len(o.strip()) > governor.inline_chars for _, o in turns]
            window_phis.append(sum(big[-6:]) / sum(big) if any(big) else 1.0)
    return {
        "task_id": tid, "steps": len(io), "replayed_identically": same,
        "window": sum(window), "fidelity": sum(fidelity), "provenance": sum(provenance),
        "max_window": max(window, default=0), "max_provenance": max(provenance, default=0),
        "recoverability": statistics.mean(phis) if phis else 1.0,
        "window_recoverability": statistics.mean(window_phis) if window_phis else 1.0, "o4": dict(governor.stats),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True, help="experiments/outputs/<experiment>/tasks of a recorded run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dirs = sorted(d for d in Path(args.tasks_dir).iterdir() if (d / "logs" / "environment_io.md").exists())
    if args.limit:
        dirs = dirs[: args.limit]
    rows = []
    for d in dirs:
        try:
            rows.append(replay_task(d, "o4_replay"))
        except Exception as e:  # noqa: BLE001  (one bad log must not stop the replay)
            rows.append({"task_id": d.name, "error": f"{type(e).__name__}: {e}"[:300]})
    ok = [r for r in rows if "error" not in r]
    steps = sum(r["steps"] for r in ok)
    summary = {"tasks": len(rows), "replayed": len(ok), "steps": steps,
               "replayedIdentically": sum(r["replayed_identically"] for r in ok) / max(1, steps)}
    for policy in ("window", "fidelity", "provenance"):
        summary[f"{policy}TokensPerStep"] = sum(r[policy] for r in ok) / max(1, steps)
    if ok:
        summary["provenanceVsWindow"] = 1 - summary["provenanceTokensPerStep"] / max(1e-9, summary["windowTokensPerStep"])
        summary["meanRecoverability"] = statistics.mean(r["recoverability"] for r in ok)
        summary["meanWindowRecoverability"] = statistics.mean(r["window_recoverability"] for r in ok)
        summary["maxWindowTokens"] = max(r["max_window"] for r in ok)
        summary["maxProvenanceTokens"] = max(r["max_provenance"] for r in ok)
        summary["o4"] = {k: sum(r["o4"].get(k, 0) for r in ok) for k in ("held", "errors", "short", "kept", "cut")}
        a, b = [r["provenance"] for r in ok], [r["window"] for r in ok]
        ra, rb = [r["recoverability"] for r in ok], [r["window_recoverability"] for r in ok]
        summary["tasksWithFewerTokens"] = sum(x < y for x, y in zip(a, b))
        summary["tasksWithHigherRecoverability"] = sum(x > y for x, y in zip(ra, rb))
        try:   # paired over tasks, two-sided
            from scipy.stats import wilcoxon
            summary["wilcoxonTokens"] = float(wilcoxon(a, b).pvalue) if a != b else 1.0
            summary["wilcoxonRecoverability"] = float(wilcoxon(ra, rb).pvalue) if ra != rb else 1.0
        except ImportError:
            pass
    summary["errors"] = [r for r in rows if "error" in r][:5]
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "tasks": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
