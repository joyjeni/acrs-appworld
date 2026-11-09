"""Worked examples (few-shot demonstrations) from AppWorld's train split.

The AppWorld paper's full-code baseline puts three solved train tasks in the prompt (StonyBrookNLP/appworld
at tag v0.1.3.post1, the version this repository runs: experiments/configs/full_code_refl_*.jsonnet set
``demo_task_ids`` 82e2fac_1, 29caf6f_1, d0b1f43_1 from ``train_dataset: train`` with
``remove_code_demo_comments``, and experiments/code/plain/agents/full_code_reflexion_agent.py shows each as
``ground_truth.compiled_solution_code_body``). This module does the same from the data
the user downloaded with ``appworld download data``: ``tasks/<id>/specs.json`` gives the
instruction and ``tasks/<id>/ground_truth/compiled_solution.py`` the solution, whose ``solution``
function body becomes the example. No AppWorld data is committed to this repository.

Leakage rules: examples come from the train split only. AppWorld splits tasks by scenario, so no
train scenario appears in dev or test, and a task listed in any other split is never used.
The selection is deterministic and the same for every config, so it is part of the shared
harness, not of any objective.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import random
import re
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Demonstration tasks of the AppWorld paper's full-code baseline (experiments/configs/full_code_refl_*.jsonnet).
APPWORLD_DEMO_TASKS = ("82e2fac_1", "29caf6f_1", "d0b1f43_1")
HELD_OUT_SPLITS = ("dev", "test_normal", "test_challenge")

# What AppWorld's REPL gives the agent (appworld/environment.py at v0.1.3.post1: TRUE_AVAILABLE_IMPORTS, plus
# `apis`). Ground-truth code can also use generator helpers (find_all_from_pages, main_user, public_data, ...)
# and the low-level `requester`, which the agent's prompt does not offer: such solutions are not shown.
REPL_NAMES = frozenset({
    "apis", "builtins", "calendar", "datetime", "itertools", "json", "math", "random", "re",
    "Counter", "defaultdict", "Iterator", "deepcopy", "reduce", "pendulum", "Date", "DateTime", "Time",
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
})
_HELPER_ATTRS = re.compile(r"\.access_token_from\s*\(")   # a generator helper on apis.<app>, not a real API
_API_APP = re.compile(r"\bapis\.([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*\s*\(")

HEADER = (
    "Worked examples: complete solutions to other tasks, from AppWorld's training split. They show how the "
    "APIs fit together (credentials, login, access tokens, pagination, complete_task). Your task is different: "
    "look up every value you need, and work one step at a time, checking each output before the next step."
)


@dataclass(frozen=True)
class Demo:
    task_id: str
    instruction: str
    code: str

    @property
    def apps(self) -> tuple[str, ...]:
        """Apps the solution calls, in order of first use (supervisor and api_docs left out)."""
        seen: list[str] = []
        for app in _API_APP.findall(self.code):
            if app not in seen and app not in ("supervisor", "api_docs"):
                seen.append(app)
        return tuple(seen)

    def render(self, number: int) -> str:
        return f"Example {number}. Task: {self.instruction}\n```python\n{self.code}\n```"


def data_dir(root: str | os.PathLike | None = None) -> Path:
    """AppWorld's data directory: <root>/data, where root defaults to APPWORLD_ROOT or the working directory."""
    return Path(root or os.environ.get("APPWORLD_ROOT") or ".") / "data"


def split_ids(data: Path, split: str, missing_ok: bool = False) -> list[str]:
    """Task ids listed in data/datasets/<split>.txt (tags after ':' removed)."""
    path = Path(data) / "datasets" / f"{split}.txt"
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"AppWorld split file not found: {path} (run `appworld download data`)")
    return [line.split(":")[0].strip() for line in path.read_text().splitlines() if line.strip()]


def _header_end(lines: list[str], fn: ast.FunctionDef) -> int:
    """1-based line of the ':' that closes the def header (the signature may span lines)."""
    depth = 0
    for tok in tokenize.generate_tokens(io.StringIO("\n".join(lines[fn.lineno - 1:])).readline):
        if tok.type == tokenize.OP:
            if tok.string in "([{":
                depth += 1
            elif tok.string in ")]}":
                depth -= 1
            elif tok.string == ":" and depth == 0:
                return fn.lineno + tok.start[0] - 1
    return fn.lineno


def solution_body(source: str) -> str | None:
    """Body of the top-level ``solution`` function, dedented, with its comments (docstring and canary dropped)."""
    source = "\n".join(line for line in source.split("\n") if "Canary String" not in line)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solution"), None)
    if fn is None:
        return None
    lines = source.split("\n")
    body = list(fn.body)
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        start = body[0].end_lineno          # 0-based index of the line after the docstring
        body = body[1:]
    else:
        start = _header_end(lines, fn)      # 0-based index of the line after the header
    if not body or body[0].lineno <= start:
        return None                         # nothing after the docstring, or code on the header line
    indent = body[0].col_offset
    out = []
    for line in lines[start: fn.end_lineno]:
        if not line.strip():
            out.append("")
        elif line[:indent].strip() == "":
            out.append(line[indent:])
        elif line.lstrip().startswith("#"):
            out.append(line.lstrip())       # a comment indented less than the code
        else:
            out.append(line)                # e.g. inside a multi-line string: leave as written
    return "\n".join(out).strip("\n") or None


def strip_comments(code: str) -> str:
    """Remove comments (not '#' inside strings) and comment-only lines; collapse blank runs."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    cut = {t.start[0]: t.start[1] for t in tokens if t.type == tokenize.COMMENT}
    out = []
    for row, line in enumerate(code.split("\n"), 1):
        if row in cut:
            line = line[: cut[row]].rstrip()
            if not line.strip():
                continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")


def free_names(code: str) -> set[str]:
    """Names the code reads but never binds, Python builtins excluded (raises SyntaxError on invalid code)."""
    loaded: set[str] = set()
    bound: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return loaded - bound - set(dir(builtins))


def missing_names(code: str) -> set[str]:
    """Names the code needs that AppWorld's REPL does not give the agent."""
    return free_names(code) - REPL_NAMES


def load_demo(data: Path, task_id: str, keep_comments: bool = False) -> Demo | None:
    """The task's instruction and compiled ground-truth solution, or None if either is unavailable."""
    task_dir = Path(data) / "tasks" / task_id
    try:
        instruction = str(json.loads((task_dir / "specs.json").read_text())["instruction"]).strip()
        source = (task_dir / "ground_truth" / "compiled_solution.py").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    body = solution_body(source)
    if not body or not instruction:
        return None
    return Demo(task_id, instruction, body if keep_comments else strip_comments(body))


def rejection(demo: Demo, min_chars: int = 200, max_chars: int = 3000) -> str:
    """Why a demo is unusable ('' if usable)."""
    try:
        compile(demo.code, f"<demo {demo.task_id}>", "exec")
    except SyntaxError:
        return "not valid as top-level code"
    if missing_names(demo.code) or _HELPER_ATTRS.search(demo.code):
        return "uses names the agent does not have"
    if "apis.supervisor.complete_task" not in demo.code:
        return "does not call complete_task"
    if len(demo.code) < min_chars:
        return f"shorter than {min_chars} characters"
    if len(demo.code) > max_chars:
        return f"longer than {max_chars} characters"
    return ""


def select_demos(
    data: Path,
    k: int,
    split: str = "train",
    preferred: Iterable[str] = APPWORLD_DEMO_TASKS,
    min_chars: int = 200,
    max_chars: int = 3000,
    seed: int = 0,
) -> list[Demo]:
    """Up to ``k`` usable demos from ``split``: the preferred ids first, then one per scenario favouring new apps.

    Deterministic for a given data directory and seed. Tasks listed in any held-out split are never used.
    """
    if k <= 0:
        return []
    ids = split_ids(data, split)
    held_out = {t for s in HELD_OUT_SPLITS if s != split for t in split_ids(data, s, missing_ok=True)}
    allowed = [t for t in ids if t not in held_out]
    preferred = [t for t in preferred if t in allowed]
    chosen: list[Demo] = []
    apps: set[str] = set()
    scenarios: set[str] = set()

    def usable(tid: str) -> Demo | None:
        demo = load_demo(data, tid)
        return demo if demo is not None and not rejection(demo, min_chars, max_chars) else None

    def take(demo: Demo) -> None:
        chosen.append(demo)
        apps.update(demo.apps)
        scenarios.add(demo.task_id.split("_")[0])

    for tid in preferred:
        if len(chosen) < k and (demo := usable(tid)) is not None:
            take(demo)
    rest = [t for t in allowed if t not in preferred]
    random.Random(seed).shuffle(rest)
    for new_app_only in (True, False):
        for tid in rest:
            if len(chosen) >= k:
                return chosen
            if tid.split("_")[0] in scenarios:
                continue
            demo = usable(tid)
            if demo is None or (new_app_only and (not demo.apps or demo.apps[0] in apps)):
                continue
            take(demo)
    return chosen


def render_demos(demos: Sequence[Demo]) -> str:
    """Text for the system prompt ('' when there are no demos)."""
    if not demos:
        return ""
    return "\n\n".join([HEADER] + [d.render(i) for i, d in enumerate(demos, 1)])


__all__ = [
    "APPWORLD_DEMO_TASKS", "REPL_NAMES", "Demo", "data_dir", "free_names", "load_demo", "missing_names", "rejection",
    "render_demos", "select_demos", "solution_body", "split_ids", "strip_comments",
]
