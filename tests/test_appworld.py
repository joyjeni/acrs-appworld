"""Offline tests for the AppWorld integration (no appworld package, no GPU)."""

from __future__ import annotations

import copy
from types import SimpleNamespace

from paim.appworld import (
    ACRSAppWorldAgent,
    AgentConfig,
    ApiCatalog,
    HistoryGovernor,
    OutcomeConsensus,
    SessionApiRetriever,
    TierRouter,
)
from paim.appworld.catalog import call_signatures, extract_calls, write_calls
from paim.appworld.llm import ChatClient, ModelTier

DOCS = {
    "api_docs": {"show_api_doc": {"method": "GET", "description": "docs"}},
    "supervisor": {
        "show_account_passwords": {"method": "GET", "description": "Show your account passwords."},
        "complete_task": {"method": "POST", "description": "Mark the task complete with an optional answer.",
                          "parameters": [{"name": "answer", "type": "any", "required": False}]},
    },
    "venmo": {
        "login": {"method": "POST", "description": "Login to venmo and get an access token.",
                  "parameters": [{"name": "username", "type": "string", "required": True}]},
        "show_transactions": {"method": "GET", "description": "Show venmo transactions paginated.",
                              "parameters": [{"name": "access_token", "type": "string", "required": True}]},
        "create_transaction": {"method": "POST", "description": "Send money to a user on venmo.",
                               "parameters": [{"name": "amount", "type": "number", "required": True}]},
        "delete_payment_card": {"method": "DELETE", "description": "Delete a payment card from venmo."},
    },
    "spotify": {
        "login": {"method": "POST", "description": "Login to spotify."},
        "show_song_library": {"method": "GET", "description": "Show songs in your spotify library."},
    },
}


class FakeWorld:
    """Tiny stand-in for appworld.AppWorld with savable state."""

    def __init__(self, instruction="Send 20 dollars to Ana on venmo for dinner.", apps=("venmo", "spotify")):
        self.task = SimpleNamespace(
            id="fake_1", instruction=instruction, allowed_apps=list(apps), api_docs=DOCS,
            supervisor=SimpleNamespace(first_name="Joy", last_name="T", email="joy@x.com", phone_number="1"),
        )
        self.state = {"sent": [], "done": False}
        self._saved = {}
        self.executed = []

    def execute(self, code):
        self.executed.append(code)
        if "raise" in code:
            return "Execution failed. Traceback: boom"
        if "create_transaction" in code:
            amt = "20" if "amount=20" in code else "other"
            self.state["sent"].append(amt)
        if "complete_task" in code:
            self.state["done"] = True
        return "ok"

    def save_state(self, sid):
        self._saved[sid] = copy.deepcopy(self.state)

    def load_state(self, sid):
        self.state = copy.deepcopy(self._saved[sid])

    def task_completed(self):
        return self.state["done"]


class ScriptedClient(ChatClient):
    def __init__(self, script, samples=None, tier="large"):
        super().__init__(ModelTier(name="fake", base_url="http://x", tier=tier))
        self.script = list(script)
        self.samples = samples or []
        self.calls = 0

    def complete(self, messages, temperature=0.0, max_tokens=1024, n=1, seed=None, stop=None):
        self.calls += 1
        self.usage.calls += 1
        self.usage.prompt_tokens += 100
        if n > 1 or temperature > 0:
            out = self.samples[:n] if self.samples else [self.last] * n
            self.samples = self.samples[n:]
            return [f"```python\n{c}\n```" for c in out]
        code = self.script.pop(0) if self.script else "apis.supervisor.complete_task()"
        self.last = code
        return [f"```python\n{code}\n```"]


def test_catalog_classifies_side_effects_from_http_method():
    cat = ApiCatalog.from_api_docs(DOCS)
    assert not cat["venmo.show_transactions"].writes
    assert cat["venmo.create_transaction"].writes and cat["venmo.delete_payment_card"].deletes
    code = "t=apis.venmo.login(username='a')\napis.venmo.create_transaction(amount=20)\napis.venmo.show_transactions(access_token=t)"
    assert extract_calls(code) == ["venmo.login", "venmo.create_transaction", "venmo.show_transactions"]
    assert write_calls(code, cat) == ["venmo.login", "venmo.create_transaction"]
    assert call_signatures(code, cat) == ("venmo.login(username='a')", "venmo.create_transaction(amount=20)")


def test_o1_session_memory_and_auth_completion():
    cat = ApiCatalog.from_api_docs(DOCS)
    r = SessionApiRetriever(cat, k=3)
    first = [s.key for s in r.retrieve("send money on venmo")]
    assert "venmo.create_transaction" in first and "venmo.login" in first  # auth completion
    r.observe("apis.spotify.show_song_library()")
    later = [s.key for s in r.retrieve("show my library", "apis.spotify.show_song_library()")]
    assert "spotify.show_song_library" in later
    static = SessionApiRetriever(cat, k=3, enabled=False)
    assert static.retrieve("send money on venmo") == static.retrieve("totally different text")


def test_o2_router_escalates_risky_contexts_and_learns():
    r = TierRouter(seed=1)
    risky = TierRouter.context(True, True, 3)
    picks = [r.choose(risky) for _ in range(200)]
    assert picks.count("large") > picks.count("small")
    safe = TierRouter.context(False, False, 3)
    for _ in range(60):
        r.update(safe, "small", True)
        r.update(safe, "large", True)
    picks = [r.choose(safe) for _ in range(200)]
    assert picks.count("small") > 100            # cheap tier wins where it is proven
    assert TierRouter(enabled=False).choose(safe) == "large"


def test_o3_outcome_consensus_certifies_agreement_and_rolls_back():
    cat = ApiCatalog.from_api_docs(DOCS)
    w = FakeWorld()
    oc = OutcomeConsensus(cat, n=3)
    good = "apis.venmo.create_transaction(amount=20)"
    res = oc.run(w, good, lambda k: [good, "apis.venmo.create_transaction(amount=99)"][:k], step=1)
    assert res.certified and res.winner.code == good
    assert w.state["sent"] == []  # exploration left no side effects


def test_o3_abstains_on_disagreement_and_on_all_errors():
    cat = ApiCatalog.from_api_docs(DOCS)
    w = FakeWorld()
    oc = OutcomeConsensus(cat, n=3)
    a, b, c = ("apis.venmo.create_transaction(amount=1)", "apis.venmo.create_transaction(amount=2)",
               "apis.venmo.create_transaction(amount=3)")
    assert not oc.run(w, a, lambda k: [b, c][:k], step=2).certified
    bad = "raise X; apis.venmo.create_transaction(amount=1)"
    assert not oc.run(w, bad, lambda k: [bad, bad][:k], step=3).certified


def test_o4_history_keeps_critical_facts_under_budget():
    g = HistoryGovernor(budget_tokens=400, keep_last=1, floor=0.9)
    turns = [(f"step{i}", ("x" * 3000) + "\naccess_token: abc123\n" + ("y" * 3000)) for i in range(6)]
    out, fid = g.compress(turns)
    assert fid >= 0.9
    assert all("access_token" in o for _, o in out)
    assert sum(len(o) for _, o in out) < sum(len(o) for _, o in turns)


def test_g1_blocks_unrequested_deletion():
    w = FakeWorld(instruction="Send 20 dollars to Ana on venmo.")
    script = ["apis.venmo.delete_payment_card()", "apis.supervisor.complete_task()"]
    cli = ScriptedClient(script)
    agent = ACRSAppWorldAgent(w, cli, cli, AgentConfig(o3_consensus=False, max_steps=5))
    log = agent.run()
    assert log.policy_denials == 1 and log.completed
    assert not any("delete_payment_card" in c for c in w.executed)


def test_full_agent_completes_with_certified_write():
    w = FakeWorld()
    script = ["apis.venmo.create_transaction(amount=20)", "apis.supervisor.complete_task()"]
    cli = ScriptedClient(script, samples=["apis.venmo.create_transaction(amount=20)"] * 2)
    agent = ACRSAppWorldAgent(w, cli, cli, AgentConfig(max_steps=6))
    log = agent.run()
    assert log.completed and w.state["sent"] == ["20"]
    assert log.consensus.get("certified", 0) >= 1


def test_baseline_config_is_plain_react():
    w = FakeWorld()
    cli = ScriptedClient(["apis.venmo.create_transaction(amount=20)", "apis.supervisor.complete_task()"])
    agent = ACRSAppWorldAgent(w, cli, cli, AgentConfig.baseline())
    log = agent.run()
    assert log.completed and log.consensus == {} and agent.policy is None


def test_restore_state_restarts_frozen_clock():
    """AppWorld 0.1.3 load_state stops the task clock; restore_state must restart it."""
    from paim.appworld.objectives import restore_state

    class ClockWorld:
        remote_environment_url = None

        def __init__(self):
            self.clock_running = True
            self.restarts = 0

        def load_state(self, sid):
            self.clock_running = False  # close_all() stops every time freezer

        def _set_datetime(self):
            self.clock_running = True
            self.restarts += 1

    w = ClockWorld()
    restore_state(w, "s1")
    assert w.clock_running and w.restarts == 1
