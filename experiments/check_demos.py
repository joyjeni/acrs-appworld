"""Check the worked examples that --few-shot K would put in the prompt, before spending GPU time.

    APPWORLD_ROOT=/path/to/appworld_root python experiments/check_demos.py --k 3 [--show]

Prints the selected train task ids, their apps and sizes, why other train tasks were not usable, and the
prompt size the examples add. Exits with status 1 if fewer than K usable examples are found.
Needs only the downloaded AppWorld data (no appworld package, no GPU). Task text and solutions are printed
only with --show: AppWorld asks that its task content is not published, and notebook outputs can be.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paim.appworld.demos import (APPWORLD_DEMO_TASKS, data_dir, load_demo, missing_names,  # noqa: E402
                                 rejection, render_demos, select_demos, split_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--root", default=None, help="AppWorld root (default: APPWORLD_ROOT or the working directory)")
    ap.add_argument("--show", action="store_true", help="print the rendered examples (local use only)")
    args = ap.parse_args()

    data = data_dir(args.root)
    ids = split_ids(data, "train")
    reasons, missing = Counter(), Counter()
    for tid in ids:
        demo = load_demo(data, tid)
        reasons["no compiled solution or instruction" if demo is None else (rejection(demo) or "usable")] += 1
        if demo is not None:
            try:
                missing.update(missing_names(demo.code))
            except SyntaxError:
                pass
    print(f"train tasks: {len(ids)} under {data.resolve()}")
    for reason, n in reasons.most_common():
        print(f"  {n:4d}  {reason}")
    if missing:   # helper names only, no task content
        print("  names the agent's REPL lacks (tasks):", ", ".join(f"{k} ({n})" for k, n in missing.most_common(10)))
    for tid in APPWORLD_DEMO_TASKS:   # the AppWorld authors' demonstration tasks, preferred when usable
        demo = load_demo(data, tid) if tid in ids else None
        why = "not in train" if tid not in ids else ("no solution" if demo is None else (rejection(demo) or "usable"))
        print(f"  authors' demo {tid}: {why}")

    demos = select_demos(data, args.k)
    for d in demos:
        print(f"selected {d.task_id}: apps={list(d.apps)} solution chars={len(d.code)}")
    text = render_demos(demos)
    print(f"examples add {len(text)} characters (about {len(text) // 3} tokens) to the system prompt")
    if args.show:
        print(text)
    if len(demos) < args.k:
        print(f"ERROR: only {len(demos)} usable examples, {args.k} requested", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
