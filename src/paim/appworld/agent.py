"""ACRS agent for AppWorld: O1-O4 integrated with G1 governance in one ReAct-style loop.

Per step:
  O1  retrieve the API docs for this step (session-aware)
  O4  provenance pruning: old observations whose values a live REPL variable still
      holds are replaced by a pointer to that variable; a ledger of the agent's
      variables (type, keys, source API call) is shown; a PID controller holds the
      history at a token budget. The ledger also feeds O1 (logged-in apps), O2
      (lossy-context risk), O3 (trial runs leave no variables) and G1 (ids and
      emails written must have been observed)
  O2  route the step to the small or large model
      the model writes one Python code block
  G1  policy check on the state-changing calls in that block
  O3  if the block changes state (a login does not count): sample alternatives,
      execute all from a saved state, commit only a certified consensus;
      otherwise execute directly. If every candidate raised, the agent sees the
      error (nothing is committed)
      feedback: router reward, retriever memory, history
The loop stops when the task is marked complete, at the step budget, or at the
token ceiling (bounded agent loop).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from paim.appworld.catalog import (ApiCatalog, is_error, iter_rendered, show_last_expression, with_silence_hint,
                                   write_calls)
from paim.appworld.llm import ChatClient, Usage
from paim.common import clock
from paim.appworld.objectives import (
    HistoryGovernor,
    OutcomeConsensus,
    SessionApiRetriever,
    TierRouter,
)
from paim.appworld.provenance import InterpreterState, ProvenanceGovernor, ProvenanceLedger, TurnMeta

SYSTEM = """You are an autonomous assistant completing a day-to-day task for your supervisor by writing Python code that calls app APIs in a REPL.

Rules:
- Reply with exactly one ```python code block per turn, optionally preceded by one short line of reasoning.
- Call APIs as apis.<app>.<api>(...). Each relevant API is listed below with its parameters ('?' = optional).
- You can always inspect more APIs with print(apis.api_docs.show_api_descriptions(app_name='<app>')) and print(apis.api_docs.show_api_doc(app_name='<app>', api_name='<api>')).
- Most APIs need an access_token: log in with that app's login API using the supervisor's credentials from apis.supervisor.show_account_passwords().
- Results are paginated: loop over page_index until a page is empty when you need all items.
- Do not guess values; look them up. Do only what the task asks: never delete or change anything the task does not require.
- When the task is done, call apis.supervisor.complete_task(answer=<answer>) if a value is asked for, otherwise apis.supervisor.complete_task().
"""

_CODE_RX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _CODE_RX.findall(text or "")
    if m:
        return m[-1].strip()
    return (text or "").strip()


@dataclass(frozen=True)
class AgentConfig:
    o1_retrieval: bool = True
    o2_routing: bool = True
    o3_consensus: bool = True
    o4_history: bool = True
    g1_policy: bool = True
    max_steps: int = 40
    max_total_tokens: int = 400_000
    k_apis: int = 10
    consensus_n: int = 3
    temperature: float = 0.0
    sample_temperature: float = 0.7
    max_tokens_step: int = 768
    budget_tokens: int = 6000
    # O4 variant: "provenance" (current: PID-controlled provenance pruning) or "fidelity" (the first O4,
    # HistoryGovernor), kept so the two designs can be compared.
    o4_mode: str = "provenance"
    o4_budget_tokens: int = 4000   # cap on the provenance governor's setpoint (it also tracks the baseline window)
    # Hard prompt limit shared by every config (the server's max model length minus the reply). 0 = off.
    # Without it a long episode overflows the server's context and the whole episode fails on an HTTP 400,
    # which would count against whichever config keeps more history.
    context_limit_tokens: int = 0

    @staticmethod
    def baseline() -> "AgentConfig":
        return AgentConfig(False, False, False, False, False)

    def label(self) -> str:
        on = [k for k in ("o1_retrieval", "o2_routing", "o3_consensus", "o4_history", "g1_policy") if getattr(self, k)]
        return "+".join(on) or "baseline"


@dataclass
class EpisodeLog:
    task_id: str
    steps: int = 0
    completed: bool = False
    stop_reason: str = ""
    errors: int = 0
    error_samples: list[str] = field(default_factory=list)  # last line of up to 15 error outputs
    policy_denials: int = 0
    consensus: dict[str, int] = field(default_factory=dict)
    router: dict[str, int] = field(default_factory=dict)
    fidelity_mean: float = 1.0
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    window_dropped: int = 0      # history turns dropped to respect context_limit_tokens
    success: bool | None = None  # AppWorld's own per-task test result (train/dev), recorded by the runner
    history_tokens: float = 0.0  # mean estimated tokens of history (and O4's variable ledger) per step
    o4: dict[str, int] = field(default_factory=dict)   # what O4 pruned (provenance mode)
    refetch_calls: int = 0       # read calls repeated with the same call text in a later committed step
    login_calls: int = 0         # login calls in committed steps
    prov_denials: int = 0        # G1 R-AW-PROV denials (an id or email that was never observed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AppWorldPolicy:
    """G1 for AppWorld, built on the generic policy engine's semantics.

    R-AW-DEL  a DELETE-type call is allowed only if the task asks to delete/remove
    R-AW-SCOPE  writes only to apps the task allows (AppWorld's own scope)
    R-AW-PROV  an id, email or phone number written as a literal by a state-changing call must appear in the
               task or in something the agent was shown (O4's provenance ledger; only when O4 is on)
    Denied blocks are not executed; the agent is told why (approve-before-execute).
    """

    DELETE_WORDS = re.compile(r"\b(delete|remove|clear|unfollow|cancel|empty|erase|unsubscribe)\b", re.I)

    def __init__(self, catalog: ApiCatalog, instruction: str, allowed_apps: list[str]):
        self.catalog = catalog
        self.instruction = instruction
        self.allowed = set(allowed_apps) | {"supervisor", "api_docs"}
        self.decisions: list[dict[str, Any]] = []

    def check(self, code: str, ledger: ProvenanceLedger | None = None) -> tuple[bool, str]:
        writes = write_calls(code, self.catalog)
        for key in writes:
            spec = self.catalog.get(key)
            app = key.split(".")[0]
            if app not in self.allowed:
                return self._deny("R-AW-SCOPE", f"{key}: app not allowed for this task")
            if spec is not None and spec.deletes and not self.DELETE_WORDS.search(self.instruction):
                return self._deny("R-AW-DEL", f"{key}: deletion not requested by the supervisor")
        if ledger is not None and writes:
            unseen = ledger.unobserved(code, set(writes))
            if unseen:
                key, param, value = unseen[0]
                return self._deny("R-AW-PROV", f"{key}: {param}={value!r} does not appear in the task or in any "
                                               "output so far, so it looks guessed; look the value up first")
        self.decisions.append({"allow": True})
        return True, ""

    def _deny(self, rule: str, why: str) -> tuple[bool, str]:
        self.decisions.append({"allow": False, "rule": rule, "reason": why})
        return False, f"[policy {rule}] blocked: {why}. Choose an action that does only what the task asks."


class ACRSAppWorldAgent:
    def __init__(self, world: Any, small: ChatClient, large: ChatClient, config: AgentConfig, seed: int = 0,
                 demos: str = ""):
        """``demos``: worked examples for the system prompt (see paim.appworld.demos), the same for every config."""
        self.world = world
        self.demos = demos.strip()
        self.task = world.task
        self.cfg = config
        self.clients = {"small": small, "large": large}
        self.catalog = ApiCatalog.from_api_docs(self.task.api_docs)
        self.retriever = SessionApiRetriever(self.catalog, k=config.k_apis, enabled=config.o1_retrieval)
        self.router = TierRouter(enabled=config.o2_routing, seed=seed)
        self.consensus = OutcomeConsensus(self.catalog, n=config.consensus_n, enabled=config.o3_consensus)
        self.state = InterpreterState(world)   # the agent's REPL variables (O3 rolls trial bindings back)
        self.ledger: ProvenanceLedger | None = None
        if config.o4_history and config.o4_mode == "provenance":
            self.ledger = ProvenanceLedger(self.state, f"{self.task.instruction}\n{self._supervisor_text()}")
            self.governor: ProvenanceGovernor | HistoryGovernor = ProvenanceGovernor(budget_tokens=config.o4_budget_tokens)
        else:
            self.governor = HistoryGovernor(budget_tokens=config.budget_tokens, enabled=config.o4_history)
        self.policy = AppWorldPolicy(self.catalog, self.task.instruction, list(self.task.allowed_apps)) if config.g1_policy else None
        self.turns: list[tuple[str, str]] = []
        self.metas: list[TurnMeta] = []
        self.fidelities: list[float] = []
        self._lossy = False
        self._history_tokens: list[int] = []
        self._reads_seen: set[str] = set()
        self.log = EpisodeLog(task_id=str(getattr(self.task, "id", "")))

    def _supervisor_text(self) -> str:
        sup = self.task.supervisor
        return (f"{getattr(sup, 'first_name', '')} {getattr(sup, 'last_name', '')}, email {getattr(sup, 'email', '')}, "
                f"phone {getattr(sup, 'phone_number', '')}")

    # ------------------------------------------------------------------ prompt
    def _messages(self) -> list[dict[str, str]]:
        recent = " ".join(c for c, _ in self.turns[-2:])
        authenticated = self.ledger.authenticated_apps() if self.ledger is not None else frozenset()
        specs = self.retriever.retrieve(self.task.instruction, recent, authenticated=authenticated)
        docs = iter_rendered(specs)
        ledger_text = summary = ""
        if isinstance(self.governor, ProvenanceGovernor) and self.ledger is not None:
            history, fid, self._lossy, summary = self.governor.compress(self.turns, self.metas, self.ledger)
            ledger_text = self.ledger.render()
        else:
            history, fid = self.governor.compress(self.turns)
        self.fidelities.append(fid)
        # The demos sit before the per-step API docs, so the prompt prefix stays the same across steps.
        demos = f"\n{self.demos}\n" if self.demos else ""
        msgs = [
            {"role": "system", "content": SYSTEM + demos + "\nRelevant APIs:\n" + docs},
            {"role": "user", "content": f"Supervisor: {self._supervisor_text()}\nTask: {self.task.instruction}\n"
                                        f"Apps available: {', '.join(self.task.allowed_apps)}"
                                        + (f"\n\n{summary}" if summary else "")},
        ]
        for code, out in history:
            msgs.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            msgs.append({"role": "user", "content": f"Output:\n{out[:6000]}"})
        if ledger_text:   # after the latest output: the current state of the session
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n\n" + ledger_text}
        self._history_tokens.append((sum(len(m["content"]) for m in msgs[2:]) + len(summary)) // 4)
        return self._fit_limit(msgs)

    def _fit_limit(self, msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop the oldest (code, output) turns until the prompt fits context_limit_tokens.

        Conservative estimate (3 characters per token, since JSON and code tokenise densely). The system prompt,
        the task and the latest turn are always kept.
        """
        limit = self.cfg.context_limit_tokens
        if limit <= 0:
            return msgs
        budget = limit - self.cfg.max_tokens_step

        def size() -> int:
            return sum(len(m["content"]) for m in msgs) // 3 + 8 * len(msgs)

        dropped = 0
        while size() > budget and len(msgs) > 4:
            del msgs[2:4]
            dropped += 1
        if dropped:
            self.log.window_dropped += dropped
            msgs[1] = {"role": "user", "content": msgs[1]["content"] + f"\n({dropped} earlier steps not shown: context limit)"}
        return msgs

    def _total_tokens(self) -> int:
        return sum(c.usage.prompt_tokens + c.usage.completion_tokens for c in self.clients.values())

    _CALL = re.compile(r"\bapis\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)")

    def _count_calls(self, code: str) -> None:
        """Logins, and read calls repeated with the same call text (a text-level proxy for re-fetching)."""
        for m in self._CALL.finditer(code or ""):
            key = f"{m.group(1)}.{m.group(2)}"
            if m.group(2) == "login":
                self.log.login_calls += 1
            spec = self.catalog.get(key)
            if m.group(1) != "api_docs" and (spec is None or spec.writes):
                continue
            args = re.sub(r"\s+", "", m.group(3))
            sig = f"{key}({args})"
            if sig in self._reads_seen:
                self.log.refetch_calls += 1
            self._reads_seen.add(sig)

    def _add_turn(self, code: str, output: str, meta: TurnMeta) -> None:
        self.turns.append((code, output))
        self.metas.append(meta)

    def _deny(self, step: int, code: str, why: str) -> None:
        self.log.policy_denials += 1
        if "R-AW-PROV" in why:
            self.log.prov_denials += 1
        self._add_turn(code, why, TurnMeta(step=step, kind="denied", error=True))

    def _record_error(self, output: str) -> None:
        self.log.errors += 1
        if len(self.log.error_samples) < 15:
            lines = [l.strip() for l in str(output).strip().splitlines() if l.strip()]
            self.log.error_samples.append(lines[-1][:160] if lines else "")

    @staticmethod
    def _abstain_output(res: Any) -> str:
        """What the agent sees when O3 does not commit the block."""
        if res.all_failed and res.candidates:
            others = len(res.candidates) - 1
            alts = f", and so did {others} sampled alternative{'s' if others != 1 else ''}" if others else ""
            # Same error the agent would see without O3; the trial run was rolled back.
            return (f"{res.candidates[0].output}\n"
                    f"[consensus] This block raised the error above{alts}. Nothing was committed: "
                    "the world state is as it was before this step.")
        summary = "; ".join(
            f"cand{i}: {'error' if c.error else 'ok'} {list(c.signature)[:2]}" for i, c in enumerate(res.candidates)
        )
        return (f"[consensus] Not executed: {res.reason}. {summary}. Nothing was committed. "
                "Re-check the required values with read-only calls, then try again.")

    # ------------------------------------------------------------------ loop
    def run(self) -> EpisodeLog:
        t0 = clock.perf_counter()
        last_error = False
        for step in range(self.cfg.max_steps):
            if self._total_tokens() > self.cfg.max_total_tokens:
                self.log.stop_reason = "token ceiling"
                break
            self.log.steps = step + 1   # every model turn counts, including blocked and abstained ones
            msgs = self._messages()
            write_likely = any(s.writes for s in self.retriever.retrieve(self.task.instruction, "")[:3])
            ctx = TierRouter.context(last_error, write_likely, step, lossy=self._lossy)
            arm = self.router.choose(ctx)
            client = self.clients[arm]
            text = client.complete(msgs, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens_step)[0]
            code = show_last_expression(extract_code(text))

            # G1: policy before any execution
            if self.policy is not None:
                ok, why = self.policy.check(code, self.ledger)
                if not ok:
                    self._deny(step, code, why)
                    self.router.update(ctx, arm, False)
                    last_error = True
                    continue

            # O3: consensus on state-changing steps
            if self.consensus.needed(code):
                def sample(k: int, msgs=msgs, client=client) -> list[str]:
                    if k <= 0:
                        return []
                    outs = client.complete(msgs, temperature=self.cfg.sample_temperature,
                                           max_tokens=self.cfg.max_tokens_step, n=k)
                    return [show_last_expression(extract_code(o)) for o in outs]

                res = self.consensus.run(self.world, code, sample, step, state=self.state)
                if res.certified and res.winner is not None:
                    if res.winner.code != code and self.policy is not None:
                        # G1 approved the greedy block; the block O3 commits must pass G1 too.
                        ok, why = self.policy.check(res.winner.code, self.ledger)
                        if not ok:
                            self._deny(step, res.winner.code, why)
                            self.router.update(ctx, arm, False)
                            last_error = True
                            continue
                    code = res.winner.code
                    output = with_silence_hint(code, self.world.execute(code))
                else:
                    # abstain: do not commit; tell the agent and escalate next step
                    output = self._abstain_output(res)
                    if res.all_failed:   # handled like a failed step without O3, minus the side effects
                        self._record_error(res.candidates[0].output)
                        self.retriever.observe(code)
                        if self.ledger is not None:
                            self.ledger.observe(res.candidates[0].output)
                    self._add_turn(code, output, TurnMeta(step=step, kind="abstained", error=True))
                    self.router.update(ctx, arm, False)
                    last_error = True
                    continue
            else:
                output = with_silence_hint(code, self.world.execute(code))

            err = is_error(output)
            if err:
                self._record_error(output)
            self.router.update(ctx, arm, not err)
            self.retriever.observe(code)
            self._count_calls(code)
            meta = self.ledger.record(step, code, output, err) if self.ledger is not None else TurnMeta(step, error=err)
            self._add_turn(code, output, meta)
            last_error = err
            if self.world.task_completed():
                self.log.completed = True
                self.log.stop_reason = "completed"
                break
        else:
            self.log.stop_reason = "step budget"
        self.log.consensus = dict(self.consensus.stats)
        self.log.router = dict(self.router.decisions)
        self.log.fidelity_mean = sum(self.fidelities) / len(self.fidelities) if self.fidelities else 1.0
        self.log.history_tokens = sum(self._history_tokens) / len(self._history_tokens) if self._history_tokens else 0.0
        if isinstance(self.governor, ProvenanceGovernor):
            self.log.o4 = dict(self.governor.stats)
        u = Usage()
        for c in self.clients.values():
            u.add(c.usage)
        self.log.usage = u.to_dict()
        self.log.seconds = clock.perf_counter() - t0
        return self.log


__all__ = ["ACRSAppWorldAgent", "AgentConfig", "AppWorldPolicy", "EpisodeLog", "extract_code"]
