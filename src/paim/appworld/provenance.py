"""O4 for AppWorld: PID-controlled provenance pruning.

A code agent's observations are mostly printed values of Python variables that stay alive: AppWorld runs a whole
episode in one IPython shell. Dropping such an observation loses nothing if the agent is told where the value now
lives, its where-provenance (Buneman, Khanna and Tan, ICDT 2001): the variable, the API call and step that produced
it, and its structure. An observation that was printed but never stored is lost when it is dropped. O4 therefore

1. keeps a ledger of the agent's variables, read from the REPL namespace (type, size, dict keys; never the text of a
   string), with the API calls and the step that produced each, taken from the step's code;
2. keeps the last turns verbatim and collapses every older turn into one line of a step summary: what it called and
   either the variables that still hold its output (nothing is lost), its error message, a short output in full, or
   an output no variable holds, shortened;
3. sets that shortening with a PID controller so that the history tracks a token budget, and measures
   recoverability: the share of past observations still visible or held by a live variable;
4. shares the ledger with the other objectives: O1 (apps already logged in need no login docs), O2 (a context that
   lost information counts as a risky step), O3 (trial runs leave no variables behind, see
   ``InterpreterState.restore``) and G1 (an id or email written by a state-changing call must have been observed,
   see ``observed_literals``).

The disabled form of O4 is the baseline's window of the last six turns, verbatim (SWE-agent-style observation
truncation). The earlier O4 (``HistoryGovernor``, a fidelity floor on regex-matched lines) is kept as ``o4_mode =
"fidelity"`` for comparison.
"""

from __future__ import annotations

import ast
import builtins
import re
import types
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

_API_CALL = re.compile(r"\bapis\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\s*\(")
_ERROR_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)\b.*$", re.MULTILINE)
_BUILTINS = frozenset(dir(builtins))
_IPYTHON = frozenset({"In", "Out", "exit", "quit", "get_ipython"})
_ID_PARAM = re.compile(r"(?:^|_)ids?$|email$|phone_number$")


# ------------------------------------------------------------------ interpreter state


class InterpreterState:
    """The agent's own variables in AppWorld's REPL.

    Only in-process worlds expose the namespace (``world.shell.user_ns``); for a remote world every method degrades
    to "no variables", and O4 falls back to what the code alone says. Names present when the episode starts (imports,
    ``apis``, AppWorld's preamble) are not the agent's and are never reported or touched.
    """

    def __init__(self, world: Any):
        self._world = world
        ns = self._ns()
        self.base = frozenset(ns) if ns is not None else frozenset()

    def _ns(self) -> dict | None:
        ns = getattr(getattr(self._world, "shell", None), "user_ns", None)
        return ns if isinstance(ns, dict) else None

    @property
    def available(self) -> bool:
        return self._ns() is not None

    def names(self) -> list[str]:
        ns = self._ns()
        if ns is None:
            return []
        return [k for k, v in ns.items() if k not in self.base and not k.startswith("_") and k not in _IPYTHON
                and not isinstance(v, types.ModuleType)]

    def get(self, name: str) -> Any:
        ns = self._ns()
        return None if ns is None else ns.get(name)

    def ids(self) -> dict[str, int]:
        ns = self._ns()
        return {} if ns is None else {k: id(ns[k]) for k in self.names()}

    def snapshot(self) -> dict[str, Any] | None:
        ns = self._ns()
        return None if ns is None else {k: ns[k] for k in self.names()}

    def restore(self, snap: dict[str, Any] | None) -> None:
        """Undo a trial run's bindings: drop the variables it created and rebind the ones it rebound.

        AppWorld's ``load_state`` restores the app databases but not the Python namespace, so without this an O3
        candidate that lost the vote would leave its variables behind. In-place mutation of an existing object (for
        example ``items.append(...)``) is not undone.
        """
        ns = self._ns()
        if ns is None or snap is None:
            return
        for k in self.names():
            if k not in snap:
                del ns[k]
        for k, v in snap.items():
            if ns.get(k) is not v:
                ns[k] = v


def describe(value: Any) -> str:
    """Structure of a value for the ledger: type, size and dict keys. String contents are never shown."""
    try:
        if value is None or isinstance(value, bool):
            return repr(value)
        if isinstance(value, (int, float)):
            return f"{type(value).__name__} {value!r}"[:40]
        if isinstance(value, str):
            return f"str ({len(value)} chars)"
        if isinstance(value, dict):
            keys = [str(k) for k in list(value)[:10]]
            more = ", ..." if len(value) > 10 else ""
            return f"dict, {len(value)} keys: {', '.join(keys)}{more}" if keys else "dict (empty)"
        if isinstance(value, (list, tuple, set, frozenset)):
            first = next(iter(value), None)
            if isinstance(first, dict):
                keys = [str(k) for k in list(first)[:8]]
                return f"{type(value).__name__} of {len(value)} dicts with keys: {', '.join(keys)}"
            inner = f" {type(first).__name__}" if first is not None else ""
            return f"{type(value).__name__} of {len(value)}{inner}"
        if callable(value):
            return "function"
        return type(value).__name__
    except Exception:  # noqa: BLE001  (a value whose len/iter raises)
        return "?"


# ------------------------------------------------------------------ static analysis of a step


@dataclass(frozen=True)
class Binding:
    step: int
    apis: tuple[str, ...]      # API calls in the expression that produced the value ('app.api')
    derived: tuple[str, ...]   # agent variables it was computed from, when no API call is involved
    loop: bool = False         # a for-loop variable: it holds only the last item of what was iterated


def _names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for elt in node.elts for n in _names(elt)]
    if isinstance(node, ast.Starred):
        return _names(node.value)
    return []


def _apis(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    found = []
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Attribute)
                and isinstance(n.func.value.value, ast.Name) and n.func.value.value.id == "apis"):
            key = f"{n.func.value.attr}.{n.func.attr}"
            if key not in found:
                found.append(key)
    return tuple(found)


def _reads(node: ast.AST | None) -> tuple[str, ...]:
    """Free names an expression reads (builtins, ``apis`` and comprehension or lambda variables left out)."""
    if node is None:
        return ()
    local: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.comprehension):
            local.update(_names(n.target))
        elif isinstance(n, ast.Lambda):
            local.update(a.arg for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs)
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in _BUILTINS and n.id != "apis" \
                and n.id not in local and n.id not in out:
            out.append(n.id)
    return tuple(out)


def _module_level(tree: ast.AST) -> Iterator[ast.AST]:
    """Nodes that run in the module namespace, in source order (function and class bodies, lambdas and
    comprehensions are not entered)."""
    stack = list(reversed(list(ast.iter_child_nodes(tree))))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp,
                             ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def bindings(code: str, step: int, prior: dict[str, Binding] | None = None) -> dict[str, Binding]:
    """Variables a step binds in the REPL namespace, with where their values came from.

    ``prior`` holds earlier steps' bindings, so ``x = f(x)`` or ``x += ...`` keeps where ``x`` came from.
    """
    prior = prior or {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    out: dict[str, Binding] = {}

    def bind(targets: list[str], value: ast.AST | None, loop: bool = False, accumulate: bool = False) -> None:
        reads = _reads(value)
        for t in targets:
            apis = _apis(value)
            derived = () if apis else tuple(n for n in reads if n not in targets)
            prev = (out.get(t) or prior.get(t)) if (accumulate or t in reads) else None
            if prev is not None:   # x += ... or x = f(x): the value keeps where x came from
                apis = tuple(dict.fromkeys(prev.apis + apis))
                derived = () if apis else tuple(dict.fromkeys(prev.derived + derived))
            out[t] = Binding(step, apis, derived, loop=loop)

    for node in _module_level(tree):
        if isinstance(node, ast.Assign):
            bind([n for t in node.targets for n in _names(t)], node.value)
        elif isinstance(node, ast.AugAssign):
            bind(_names(node.target), node.value, accumulate=True)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(_names(node.target), node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind(_names(node.target), node.iter, loop=True)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind(_names(node.optional_vars), node.context_expr)
        elif isinstance(node, ast.NamedExpr):
            bind(_names(node.target), node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = Binding(step, (), ())
    return out


def printed(code: str) -> tuple[set[str], bool]:
    """Names read inside print(...) calls, and whether a print argument calls an API directly."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), True
    names: set[str] = set()
    direct = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            for arg in list(node.args) + [k.value for k in node.keywords]:
                direct = direct or bool(_apis(arg))
                names.update(_reads(arg))
    return names, direct


def observed_literals(code: str, writes: set[str]) -> list[tuple[str, str, str]]:
    """(api, parameter, value) for id-like literal arguments of the given state-changing APIs ('app.api')."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "apis"):
            continue
        key = f"{node.func.value.attr}.{node.func.attr}"
        if key not in writes:
            continue
        for kw in node.keywords:
            if not kw.arg or not _ID_PARAM.search(kw.arg):
                continue
            values = kw.value.elts if isinstance(kw.value, (ast.List, ast.Tuple, ast.Set)) else [kw.value]
            for v in values:
                if isinstance(v, ast.Constant) and isinstance(v.value, (str, int)) and not isinstance(v.value, bool):
                    out.append((key, kw.arg, str(v.value)))
    return out


# ------------------------------------------------------------------ ledger


@dataclass
class TurnMeta:
    step: int
    kind: str = "executed"        # executed | denied | abstained
    error: bool = False
    recoverable: bool = False     # every printed value is still held by an unchanged variable
    holders: tuple[str, ...] = ()


class ProvenanceLedger:
    """Where each value in the agent's session came from, and everything the agent has observed."""

    def __init__(self, state: InterpreterState, task_text: str = "", corpus_limit: int = 2_000_000):
        self.state = state
        self.bindings: dict[str, Binding] = {}
        self._ids_after: dict[int, dict[str, int]] = {}
        self._corpus: list[str] = [task_text.lower()]
        self._size = len(task_text)
        self._limit = corpus_limit

    # observations ----------------------------------------------------
    def observe(self, text: str) -> None:
        """Add something the agent was shown to the observed corpus (kept even when the context is pruned)."""
        if text and self._size < self._limit:
            self._corpus.append(text.lower())
            self._size += len(text)

    def seen(self, value: str) -> bool:
        v = value.lower()
        return any(v in chunk for chunk in self._corpus)

    # steps -------------------------------------------------------------
    def record(self, step: int, code: str, output: str, error: bool) -> TurnMeta:
        """Update provenance after a committed step; return what O4 needs to prune the step's observation later."""
        self.observe(output)
        new = bindings(code, step, self.bindings)
        live = set(self.state.names())
        for name, b in new.items():
            if name in live or not self.state.available:
                self.bindings[name] = b
        for name in list(self.bindings):
            if self.state.available and name not in live:
                del self.bindings[name]
        ids = self.state.ids()
        self._ids_after[step] = ids
        holders: tuple[str, ...] = ()
        recoverable = False
        if not error and self.state.available:
            names, direct = printed(code)
            agent: set[str] = set()
            for n in names:
                b = new.get(n)
                if b is not None and b.loop:
                    if b.apis:          # iterated straight over an API result: the items were never stored
                        direct = True
                    agent.update(b.derived)   # the loop variable holds one item; the iterable holds them all
                elif n not in self.state.base:
                    agent.add(n)
            recoverable = bool(agent) and not direct and all(n in ids for n in agent)
            holders = tuple(sorted(agent)) if recoverable else ()
        return TurnMeta(step=step, error=error, recoverable=recoverable, holders=holders)

    def still_held(self, meta: TurnMeta) -> bool:
        """The variables that held a step's printed values are still bound to the same objects."""
        if not meta.recoverable:
            return False
        then = self._ids_after.get(meta.step, {})
        now = self.state.ids()
        return all(n in now and now[n] == then.get(n) for n in meta.holders)

    # what the other objectives read --------------------------------------
    def authenticated_apps(self) -> set[str]:
        """Apps whose login result is held by a live variable (O1 then leaves out their login docs)."""
        live = set(self.state.names()) if self.state.available else set(self.bindings)
        return {a.split(".")[0] for n, b in self.bindings.items() if n in live for a in b.apis if a.endswith(".login")}

    def unobserved(self, code: str, writes: set[str]) -> list[tuple[str, str, str]]:
        """Id-like literals a state-changing call would write that the agent has never been shown (G1)."""
        return [(k, p, v) for k, p, v in observed_literals(code, writes) if len(v) >= 3 and not self.seen(v)]

    def render(self, max_entries: int = 25) -> str:
        names = self.state.names() if self.state.available else list(self.bindings)
        if not names:
            return ""
        names.sort(key=lambda n: -(self.bindings[n].step if n in self.bindings else -1))
        lines = []
        for n in names[:max_entries]:
            desc = describe(self.state.get(n)) if self.state.available else "value"
            b = self.bindings.get(n)
            if b is None:
                where = "earlier"
            elif b.apis:
                where = f"from {', '.join(b.apis[:3])}, step {b.step + 1}"
            elif b.derived:
                where = f"from {', '.join(b.derived[:3])}, step {b.step + 1}"
            else:
                where = f"step {b.step + 1}"
            lines.append(f"- {n}: {desc[:150]} ({where})")
        more = f"\n- ... {len(names) - max_entries} more" if len(names) > max_entries else ""
        return ("Variables in your Python session (reuse them; print what you need instead of calling the API "
                "again):\n" + "\n".join(lines) + more)


# ------------------------------------------------------------------ governor


@dataclass
class ProvenanceGovernor:
    """Collapses old turns into an outline that points to the variables holding their values.

    The last ``keep_last`` turns stay verbatim. Every older turn becomes a line of a step outline: what it called, then
    its result. In order of preference the result is:

    - its error message;
    - a short output in full;
    - the variables that still hold its output (nothing is lost: the ledger says what they contain);
    - an output no variable holds, cut to ``_chars`` characters.

    Consecutive steps with the same line are merged (``[5-9] ... (5 times)``), which also tells the agent that it
    repeated itself. A PID controller sets ``_chars`` so that the history (outline, recent turns and ledger) tracks a
    setpoint: ``budget_tokens``, or with ``match_window``, ``window_ratio`` times the size the baseline's six-turn
    window would have at this step, if that is smaller. O4 then costs less context than the baseline and spends it on
    keeping the past recoverable. Recoverability, the share of past non-trivial observations still visible or held by
    a live variable, is measured at every step.
    """

    budget_tokens: int = 4000
    match_window: bool = True
    window_ratio: float = 0.9
    window_turns: int = 6
    keep_last: int = 3
    kp: float = 0.6
    ki: float = 0.2
    kd: float = 0.1
    inline_chars: int = 120        # outputs this short appear in full in the outline
    min_chars: int = 80
    max_chars: int = 2000
    _chars: int = 400
    _integral: float = 0.0
    _last_err: float = 0.0
    # Old turns by how the latest outline shows them (distinct turns, not cumulative), and the number of steps.
    stats: dict[str, int] = field(default_factory=lambda: {"held": 0, "errors": 0, "short": 0, "cut": 0,
                                                          "kept": 0, "steps": 0})
    log: list[dict[str, float]] = field(default_factory=list)

    @staticmethod
    def tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _error_message(output: str) -> str:
        found = _ERROR_LINE.findall(output or "")
        if found:
            return found[-1][:160]
        lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
        return lines[-1][:160] if lines else ""

    @staticmethod
    def _what(code: str) -> str:
        calls = list(dict.fromkeys(f"{a}.{b}" for a, b in _API_CALL.findall(code or "")))
        if calls:
            return ", ".join(calls[:4]) + (", ..." if len(calls) > 4 else "")
        first = next((ln.strip() for ln in (code or "").splitlines() if ln.strip()), "")
        return first[:70] + ("..." if len(first) > 70 else "")

    @staticmethod
    def _merge(entries: list[tuple[int, str]]) -> list[str]:
        """Outline lines, merging runs of consecutive steps that read the same."""
        lines: list[str] = []
        i = 0
        while i < len(entries):
            j = i
            while j + 1 < len(entries) and entries[j + 1][1] == entries[i][1] and entries[j + 1][0] == entries[j][0] + 1:
                j += 1
            first, text = entries[i]
            last = entries[j][0]
            lines.append(f"[{first}] {text}" if j == i else f"[{first}-{last}] {text} ({j - i + 1} times)")
            i = j + 1
        return lines

    def compress(self, turns: Sequence[tuple[str, str]], metas: Sequence[TurnMeta], ledger: ProvenanceLedger
                 ) -> tuple[list[tuple[str, str]], float, bool, str]:
        """Recent turns, recoverability, whether anything was cut, and the outline of the older turns."""
        if not turns:
            return [], 1.0, False, ""
        first_recent = max(0, len(turns) - self.keep_last)
        entries: list[tuple[int, str]] = []
        counts = {"held": 0, "errors": 0, "short": 0, "cut": 0, "kept": 0}
        total = available = 0
        lossy = False
        for (code, output), meta in zip(turns[:first_recent], metas[:first_recent]):
            text = (output or "").strip()
            nontrivial = len(text) > self.inline_chars
            total += int(nontrivial)
            if meta.error:
                result = self._error_message(text)
                counts["errors"] += 1
                available += int(nontrivial)
            elif not nontrivial:   # short: shown in full (a pointer would not be shorter)
                bound = [n for n, b in ledger.bindings.items() if b.step == meta.step]
                result = " ".join(text.split()) or "ok"
                if bound and (not text or text.startswith("Execution successful")):
                    result = "bound " + ", ".join(bound[:6])
                counts["short"] += 1
            elif ledger.still_held(meta):
                result = f"output held in {', '.join(meta.holders)}"
                counts["held"] += 1
                available += 1
            elif len(text) <= self._chars:
                result = text
                counts["kept"] += 1
                available += 1
            else:
                result = (text[: self._chars] + f" ... [+{len(text) - self._chars} chars; no variable holds this "
                          "output: repeat the call and assign the result if you need it]")
                counts["cut"] += 1
                lossy = True
            entries.append((meta.step + 1, f"{self._what(code)} -> {result}"))
        recent = list(turns[first_recent:])
        for _, output in recent:
            nontrivial = len((output or "").strip()) > self.inline_chars
            total += int(nontrivial)
            available += int(nontrivial)
        lines = self._merge(entries)
        summary = ("Earlier steps (code collapsed; each output is shown, pointed to the variable that holds it, or "
                   "shortened):\n" + "\n".join(lines)) if lines else ""
        used = (self.tokens(summary) if summary else 0) + self.tokens(ledger.render()) + \
            sum(self.tokens(c) + self.tokens(o[:6000]) for c, o in recent)
        setpoint = self.budget_tokens
        if self.match_window:
            window = sum(self.tokens(c) + self.tokens(o[:6000]) for c, o in turns[-self.window_turns:])
            setpoint = max(1, min(setpoint, int(self.window_ratio * window)))
        err = (setpoint - used) / setpoint   # under the setpoint: keep more of the shortened outputs
        derivative = err - self._last_err
        self._last_err = err
        self._integral = max(-2.0, min(2.0, self._integral + err))
        factor = 1 + self.kp * err + self.ki * 0.1 * self._integral + self.kd * derivative
        self._chars = int(max(self.min_chars, min(self.max_chars, self._chars * max(0.5, factor))))
        phi = available / total if total else 1.0
        self.stats.update(counts)
        self.stats["steps"] += 1
        self.log.append({"recoverability": phi, "tokens": float(used), "setpoint": float(setpoint),
                         "chars": float(self._chars)})
        return recent, phi, lossy, summary


__all__ = [
    "Binding", "InterpreterState", "ProvenanceGovernor", "ProvenanceLedger", "TurnMeta", "bindings", "describe",
    "observed_literals", "printed",
]
