"""ACRS agent for AppWorld: O1-O4 integrated with G1 governance in one ReAct-style loop.

Per step:
  O1  retrieve the API docs for this step (session-aware)
  O4  compress the episode history under a token budget with a fidelity floor
  O2  route the step to the small or large model
      the model writes one Python code block
  G1  policy check on the state-changing calls in that block
  O3  if the block changes state: sample alternatives, execute all from a saved
      state, commit only a certified consensus; otherwise execute directly
      feedback: router reward, retriever memory, history
The loop stops when the task is marked complete, at the step budget, or at the
token ceiling (bounded agent loop).
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from paim.appworld.catalog import ApiCatalog, is_error, iter_rendered, write_calls
from paim.appworld.llm import ChatClient, Usage
from paim.appworld.objectives import (
    HistoryGovernor,
    OutcomeConsensus,
    SessionApiRetriever,
    TierRouter,
)

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AppWorldPolicy:
    """G1 for AppWorld, built on the generic policy engine's semantics.

    R-AW-DEL  a DELETE-type call is allowed only if the task asks to delete/remove
    R-AW-SCOPE  writes only to apps the task allows (AppWorld's own scope)
    Denied blocks are not executed; the agent is told why (approve-before-execute).
    """

    DELETE_WORDS = re.compile(r"\b(delete|remove|clear|unfollow|cancel|empty|erase|unsubscribe)\b", re.I)

    def __init__(self, catalog: ApiCatalog, instruction: str, allowed_apps: list[str]):
        self.catalog = catalog
        self.instruction = instruction
        self.allowed = set(allowed_apps) | {"supervisor", "api_docs"}
        self.decisions: list[dict[str, Any]] = []

    def check(self, code: str) -> tuple[bool, str]:
        for key in write_calls(code, self.catalog):
            spec = self.catalog.get(key)
            app = key.split(".")[0]
            if app not in self.allowed:
                return self._deny("R-AW-SCOPE", f"{key}: app not allowed for this task")
            if spec is not None and spec.deletes and not self.DELETE_WORDS.search(self.instruction):
                return self._deny("R-AW-DEL", f"{key}: deletion not requested by the supervisor")
        self.decisions.append({"allow": True})
        return True, ""

    def _deny(self, rule: str, why: str) -> tuple[bool, str]:
        self.decisions.append({"allow": False, "rule": rule, "reason": why})
        return False, f"[policy {rule}] blocked: {why}. Choose an action that does only what the task asks."


class ACRSAppWorldAgent:
    def __init__(self, world: Any, small: ChatClient, large: ChatClient, config: AgentConfig, seed: int = 0):
        self.world = world
        self.task = world.task
        self.cfg = config
        self.clients = {"small": small, "large": large}
        self.catalog = ApiCatalog.from_api_docs(self.task.api_docs)
        self.retriever = SessionApiRetriever(self.catalog, k=config.k_apis, enabled=config.o1_retrieval)
        self.router = TierRouter(enabled=config.o2_routing, seed=seed)
        self.consensus = OutcomeConsensus(self.catalog, n=config.consensus_n, enabled=config.o3_consensus)
        self.governor = HistoryGovernor(budget_tokens=config.budget_tokens, enabled=config.o4_history)
        self.policy = AppWorldPolicy(self.catalog, self.task.instruction, list(self.task.allowed_apps)) if config.g1_policy else None
        self.turns: list[tuple[str, str]] = []
        self.fidelities: list[float] = []
        self.log = EpisodeLog(task_id=str(getattr(self.task, "id", "")))

    # ------------------------------------------------------------------ prompt
    def _messages(self) -> list[dict[str, str]]:
        sup = self.task.supervisor
        sup_txt = f"{getattr(sup, 'first_name', '')} {getattr(sup, 'last_name', '')}, email {getattr(sup, 'email', '')}, phone {getattr(sup, 'phone_number', '')}"
        recent = " ".join(c for c, _ in self.turns[-2:])
        specs = self.retriever.retrieve(self.task.instruction, recent)
        docs = iter_rendered(specs)
        history, fid = self.governor.compress(self.turns)
        self.fidelities.append(fid)
        msgs = [
            {"role": "system", "content": SYSTEM + "\nRelevant APIs:\n" + docs},
            {"role": "user", "content": f"Supervisor: {sup_txt}\nTask: {self.task.instruction}\nApps available: {', '.join(self.task.allowed_apps)}"},
        ]
        for code, out in history:
            msgs.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            msgs.append({"role": "user", "content": f"Output:\n{out[:6000]}"})
        return msgs

    def _total_tokens(self) -> int:
        return sum(c.usage.prompt_tokens + c.usage.completion_tokens for c in self.clients.values())

    # ------------------------------------------------------------------ loop
    def run(self) -> EpisodeLog:
        t0 = time.time()
        last_error = False
        for step in range(self.cfg.max_steps):
            if self._total_tokens() > self.cfg.max_total_tokens:
                self.log.stop_reason = "token ceiling"
                break
            msgs = self._messages()
            write_likely = any(s.writes for s in self.retriever.retrieve(self.task.instruction, "")[:3])
            ctx = TierRouter.context(last_error, write_likely, step)
            arm = self.router.choose(ctx)
            client = self.clients[arm]
            text = client.complete(msgs, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens_step)[0]
            code = extract_code(text)

            # G1: policy before any execution
            if self.policy is not None:
                ok, why = self.policy.check(code)
                if not ok:
                    self.log.policy_denials += 1
                    self.turns.append((code, why))
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
                    return [extract_code(o) for o in outs]

                res = self.consensus.run(self.world, code, sample, step)
                if res.certified and res.winner is not None:
                    code = res.winner.code
                    output = self.world.execute(code)
                else:
                    # abstain: do not commit; tell the agent and escalate next step
                    summary = "; ".join(
                        f"cand{i}: {'error' if c.error else 'ok'} {list(c.signature)[:2]}" for i, c in enumerate(res.candidates)
                    )
                    output = (f"[consensus] Not executed: {res.reason}. {summary}. "
                              "Re-check the required values with read-only calls, then try again.")
                    self.turns.append((code, output))
                    self.router.update(ctx, arm, False)
                    last_error = True
                    continue
            else:
                output = self.world.execute(code)

            err = is_error(output)
            self.log.errors += int(err)
            if err and len(self.log.error_samples) < 15:
                lines = [l.strip() for l in str(output).strip().splitlines() if l.strip()]
                self.log.error_samples.append(lines[-1][:160] if lines else "")
            self.router.update(ctx, arm, not err)
            self.retriever.observe(code)
            self.turns.append((code, output))
            last_error = err
            self.log.steps = step + 1
            if self.world.task_completed():
                self.log.completed = True
                self.log.stop_reason = "completed"
                break
        else:
            self.log.stop_reason = "step budget"
        self.log.consensus = dict(self.consensus.stats)
        self.log.router = dict(self.router.decisions)
        self.log.fidelity_mean = sum(self.fidelities) / len(self.fidelities) if self.fidelities else 1.0
        u = Usage()
        for c in self.clients.values():
            u.add(c.usage)
        self.log.usage = u.to_dict()
        self.log.seconds = time.time() - t0
        return self.log


__all__ = ["ACRSAppWorldAgent", "AgentConfig", "AppWorldPolicy", "EpisodeLog", "extract_code"]
