"""O1-O4 specialised to AppWorld episodes.

O1  SessionApiRetriever     which API docs the model sees at each step
O2  TierRouter              which model tier writes each step
O3  OutcomeConsensus        ARC-style: execute several candidate programs from a
                            saved world state, vote on their *outcomes*, commit
                            only a certified one
O4  HistoryGovernor         keep the episode history inside a token budget while
                            holding a fidelity floor on the facts the agent needs

Each class has an ``enabled`` switch; disabled means its SOTA analogue
(static retrieval, always-large model, single greedy execution, recency
truncation), so one agent produces every row of the ablation.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from paim.appworld.catalog import ApiCatalog, ApiSpec, call_signatures, is_error, write_calls
from paim.o1_sasr.retriever import BM25Retriever

# ----------------------------------------------------------------------- O1


@dataclass
class SessionApiRetriever:
    """BM25 over API docs + session app memory + auth completion.

    Session terms (all training-free, so they cannot leak test data):
      * app memory: apps the episode already used are boosted, decayed per step;
      * auth completion: when an app enters the session, its login API is added,
        because every protected call needs that app's access token;
      * committed APIs stay visible until the episode stops using them.
    The SOTA analogue retrieves once from the instruction and never updates.
    """

    catalog: ApiCatalog
    k: int = 10
    decay: float = 0.7
    app_weight: float = 0.6
    enabled: bool = True
    _bm25: BM25Retriever | None = None
    _app_mem: dict[str, float] = field(default_factory=dict)
    _committed: list[str] = field(default_factory=list)
    _static: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._bm25 = BM25Retriever().fit(self.catalog.tools())

    def observe(self, code: str) -> None:
        for k in list(self._app_mem):
            self._app_mem[k] *= self.decay
        for call in re.findall(r"\bapis\.([a-z_]+)\.([a-z_0-9]+)\s*\(", code or ""):
            app, api = call
            if app in ("api_docs",):
                continue
            self._app_mem[app] = self._app_mem.get(app, 0.0) + 1.0
            key = f"{app}.{api}"
            if key in self.catalog.specs and key not in self._committed:
                self._committed.append(key)
        self._committed = self._committed[-12:]

    def retrieve(self, instruction: str, recent: str = "") -> list[ApiSpec]:
        assert self._bm25 is not None
        if not self.enabled:
            if not self._static:
                self._static = [c.doc_id for c in self._bm25.search(instruction, self.k)]
            return [self.catalog[d] for d in self._static]
        query = f"{instruction} {recent}"[-2000:]
        cands = self._bm25.search(query, self.k * 4)
        if not cands:
            return []
        top = max(c.score for c in cands) or 1.0
        scored = []
        for c in cands:
            spec = self.catalog[c.doc_id]
            s = c.score / top + self.app_weight * math.tanh(self._app_mem.get(spec.app, 0.0))
            scored.append((s, c.doc_id))
        scored.sort(reverse=True)
        chosen: list[str] = []
        for key in self._committed[-4:]:
            if key not in chosen:
                chosen.append(key)
        for _, d in scored:
            if len(chosen) >= self.k:
                break
            if d not in chosen:
                chosen.append(d)
        # auth completion
        for app in {self.catalog[d].app for d in chosen}:
            login = f"{app}.login"
            if login in self.catalog.specs and login not in chosen:
                chosen.append(login)
        return [self.catalog[d] for d in chosen]


# ----------------------------------------------------------------------- O2


@dataclass
class TierRouter:
    """Risk-signed two-arm Thompson router (small vs large model) per context.

    Context = (recent error?, write-likely step?, late in episode?). Reward is
    1 for an error-free, policy-allowed step, minus a cost penalty for the large
    tier. High-risk contexts (recent error or write-likely) are biased to the
    large tier until the small tier has proven itself there: this is the
    risk-signed part. The SOTA analogue always uses the large model.
    """

    enabled: bool = True
    cost_penalty: float = 0.15
    risk_margin: float = 0.35
    seed: int = 0
    _ab: dict[tuple, list[float]] = field(default_factory=lambda: defaultdict(lambda: [1.0, 1.0, 1.0, 1.0]))
    _rng: random.Random = field(default_factory=random.Random)
    decisions: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self._rng.seed(self.seed)

    @staticmethod
    def context(last_error: bool, write_likely: bool, step: int) -> tuple:
        return (bool(last_error), bool(write_likely), step >= 15)

    def choose(self, ctx: tuple) -> str:
        if not self.enabled:
            self.decisions["large"] += 1
            return "large"
        a_s, b_s, a_l, b_l = self._ab[ctx]
        small = self._rng.betavariate(a_s, b_s)
        large = self._rng.betavariate(a_l, b_l) - self.cost_penalty
        if ctx[0] or ctx[1]:
            # Risk sign: on risky steps the small tier must beat the large one by
            # more than the cost saving before it is trusted.
            small -= self.risk_margin
        arm = "small" if small > large else "large"
        self.decisions[arm] += 1
        return arm

    def update(self, ctx: tuple, arm: str, success: bool) -> None:
        ab = self._ab[ctx]
        i = 0 if arm == "small" else 2
        if success:
            ab[i] += 1.0
        else:
            ab[i + 1] += 1.0


# ----------------------------------------------------------------------- O3


@dataclass
class Candidate:
    code: str
    output: str = ""
    error: bool = False
    signature: tuple[str, ...] = ()
    weight: float = 0.0


@dataclass
class ConsensusResult:
    certified: bool
    winner: Candidate | None
    candidates: list[Candidate]
    share: float
    reason: str




def restore_state(world: Any, state_id: str) -> None:
    """Restore a saved world state.

    AppWorld 0.1.3's ``load_state`` calls ``AppWorld.close_all()``, which stops the task's
    frozen clock but does not restart it; the episode then crashes when the world closes and
    stops the clock again (freezegun: ``fake_names`` / ``pop from empty list``), and every API
    call in between runs on real time. Restart the clock after restoring.
    """
    world.load_state(state_id)
    if getattr(world, "remote_environment_url", None) is None and hasattr(world, "_set_datetime"):
        world._set_datetime()


@dataclass
class OutcomeConsensus:
    """ARC-style execute-and-vote over candidate programs, on state-changing steps.

    ARC solvers sample many candidate programs, run them, and keep the output
    that independent programs agree on. Here the "output" of a candidate is the
    set of state-changing API calls it actually made plus whether it raised.
    Candidates run from the same saved world state and the state is restored
    after each, so exploration has no side effects. A candidate's vote weight
    is 1 if it executed cleanly, 0 otherwise; the signature with the largest
    weighted share wins if it reaches theta with at least ``quorum`` supporters.
    Otherwise MNCD abstains, and the agent is told the candidates disagreed
    (and the router escalates), instead of committing a guess.
    """

    catalog: ApiCatalog
    n: int = 3
    theta: float = 0.5
    quorum: int = 2
    enabled: bool = True
    stats: Counter = field(default_factory=Counter)

    def needed(self, code: str) -> bool:
        return self.enabled and bool(write_calls(code, self.catalog))

    def run(
        self,
        world: Any,
        first: str,
        sample: Callable[[int], list[str]],
        step: int,
    ) -> ConsensusResult:
        codes = [first] + [c for c in sample(self.n - 1) if c.strip()]
        state_id = f"acrs_s{step}"
        world.save_state(state_id)
        cands: list[Candidate] = []
        for code in codes:
            out = world.execute(code)
            err = is_error(out)
            cands.append(
                Candidate(code=code, output=out, error=err,
                          signature=call_signatures(code, self.catalog),
                          weight=0.0 if err else 1.0)
            )
            restore_state(world, state_id)
        self.stats["consensus_steps"] += 1
        self.stats["candidate_executions"] += len(cands)
        groups: dict[tuple, float] = defaultdict(float)
        support: dict[tuple, int] = defaultdict(int)
        for c in cands:
            groups[c.signature] += c.weight
            if c.weight > 0:
                support[c.signature] += 1
        total = sum(c.weight for c in cands)
        if total == 0:
            self.stats["abstain_all_failed"] += 1
            return ConsensusResult(False, None, cands, 0.0, "all candidates raised errors")
        sig, w = max(groups.items(), key=lambda kv: kv[1])
        share = w / len(cands)
        if share >= self.theta and support[sig] >= min(self.quorum, len(cands)):
            winner = next(c for c in cands if c.signature == sig and not c.error)
            self.stats["certified"] += 1
            return ConsensusResult(True, winner, cands, share, f"{support[sig]}/{len(cands)} agree")
        self.stats["abstain_disagree"] += 1
        return ConsensusResult(False, None, cands, share, "candidates disagree on state changes")


# ----------------------------------------------------------------------- O4


_CRITICAL = re.compile(r"access_token|\"id\"|_id\b|password|Traceback|Exception|error", re.IGNORECASE)


@dataclass
class HistoryGovernor:
    """Keeps (code, output) turns within a token budget, with a fidelity floor.

    Hard-keep: the task, the last ``keep_last`` turns, and every turn whose
    output carries critical facts (tokens, ids, errors). Others are compressed
    head+tail. A PI loop adjusts the compression length so measured fidelity
    (share of critical lines retained) stays above ``floor`` while the token
    count stays under budget. The SOTA analogue keeps only the last N turns.
    """

    budget_tokens: int = 6000
    keep_last: int = 3
    floor: float = 0.9
    enabled: bool = True
    kp: float = 0.6
    ki: float = 0.2
    _chars: int = 600
    _integral: float = 0.0
    log: list[dict[str, float]] = field(default_factory=list)

    @staticmethod
    def tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def compress(self, turns: Sequence[tuple[str, str]]) -> tuple[list[tuple[str, str]], float]:
        if not turns:
            return [], 1.0
        if not self.enabled:
            kept = list(turns[-max(self.keep_last * 2, 6):])
            crit_all = sum(len(_CRITICAL.findall(o)) for _, o in turns) or 1
            crit_kept = sum(len(_CRITICAL.findall(o)) for _, o in kept)
            return kept, crit_kept / crit_all
        out: list[tuple[str, str]] = []
        crit_all = 0
        crit_kept = 0
        n = len(turns)
        for i, (code, output) in enumerate(turns):
            c = len(_CRITICAL.findall(output))
            crit_all += c
            if i >= n - self.keep_last or len(output) <= self._chars:
                out.append((code, output))
                crit_kept += c
                continue
            lines = output.splitlines()
            critical = [ln for ln in lines if _CRITICAL.search(ln)]
            head = output[: self._chars // 2]
            tail = output[-self._chars // 2:]
            keep_crit = [ln for ln in critical if ln not in head and ln not in tail][:20]
            compact = head + "\n...[compressed]...\n" + "\n".join(keep_crit) + "\n" + tail
            out.append((code, compact))
            crit_kept += len(_CRITICAL.findall(compact))
        fidelity = min(1.0, crit_kept / crit_all) if crit_all else 1.0
        used = sum(self.tokens(a) + self.tokens(b) for a, b in out)
        # PI control on the compression length: protect fidelity first, then budget.
        err = (fidelity - self.floor) if fidelity < self.floor else (self.budget_tokens - used) / self.budget_tokens
        self._integral = max(-2.0, min(2.0, self._integral + err))
        self._chars = int(max(200, min(4000, self._chars * (1 + self.kp * err + self.ki * self._integral * 0.1))))
        self.log.append({"fidelity": fidelity, "tokens": float(used), "chars": float(self._chars)})
        return out, fidelity


__all__ = [
    "Candidate",
    "ConsensusResult",
    "HistoryGovernor",
    "OutcomeConsensus",
    "SessionApiRetriever",
    "TierRouter",
]
