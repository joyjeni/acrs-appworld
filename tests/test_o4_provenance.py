"""O4 provenance pruning and its links to O1, O2, O3 and G1.

ExecWorld runs code for real in one persistent namespace, like AppWorld's IPython shell: ``save_state`` /
``load_state`` restore the app data but not the Python variables, exactly as AppWorld 0.1.3 does. Offline: no
appworld package, no GPU.
"""

from __future__ import annotations

import copy
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace

from paim.appworld import ACRSAppWorldAgent, AgentConfig, ApiCatalog, OutcomeConsensus, SessionApiRetriever, TierRouter
from paim.appworld.agent import AppWorldPolicy
from paim.appworld.objectives import HistoryGovernor
from paim.appworld.provenance import (InterpreterState, ProvenanceGovernor, ProvenanceLedger, TurnMeta, bindings,
                                      describe, printed)

from test_appworld import ScriptedClient
from test_kaggle_harness import _runner

DOCS = {
    "api_docs": {"show_api_doc": {"method": "GET", "description": "docs"}},
    "supervisor": {
        "show_profile": {"method": "GET", "description": "Show your profile."},
        "show_account_passwords": {"method": "GET", "description": "Show your account passwords."},
        "complete_task": {"method": "POST", "description": "Mark the task complete.",
                          "parameters": [{"name": "answer", "type": "any", "required": False}]},
    },
    "venmo": {
        "login": {"method": "POST", "description": "Login to venmo and get an access token.",
                  "parameters": [{"name": "username", "type": "string", "required": True}]},
        "show_contacts": {"method": "GET", "description": "Show venmo contacts, paginated.",
                          "parameters": [{"name": "access_token", "type": "string", "required": True}]},
        "create_transaction": {"method": "POST", "description": "Send money to a venmo user.",
                               "parameters": [{"name": "receiver_email", "type": "string", "required": True},
                                              {"name": "amount", "type": "number", "required": True}]},
    },
}
CONTACTS = [{"id": 4821 + i, "email": f"user{i}@x.com", "name": f"User {i}"} for i in range(30)] + \
           [{"id": 9001, "email": "ana@x.com", "name": "Ana"}]


class _App:
    def __init__(self, world, name):
        self._w, self._name = world, name

    def __getattr__(self, api):
        def call(**kw):
            return self._w.handle(self._name, api, kw)
        return call


class _Apis:
    def __init__(self, world):
        self._w = world

    def __getattr__(self, app):
        return _App(self._w, app)


class ExecWorld:
    """Executes code in one namespace; app data is saved and restored, Python variables are not (as in AppWorld)."""

    def __init__(self, instruction="Send 20 dollars to Ana on venmo."):
        self.task = SimpleNamespace(
            id="exec_1", instruction=instruction, allowed_apps=["venmo"], api_docs=DOCS,
            supervisor=SimpleNamespace(first_name="Joy", last_name="T", email="joy@x.com", phone_number="5550100"))
        self.db = {"sent": [], "done": False}
        self._saved = {}
        self.shell = SimpleNamespace(user_ns={"apis": _Apis(self), "json": json, "In": [], "Out": {}})
        self.executed: list[str] = []

    def handle(self, app, api, kw):
        if (app, api) == ("supervisor", "show_profile"):
            return {"first_name": "Joy", "last_name": "T", "email": "joy@x.com", "phone_number": "5550100"}
        if (app, api) == ("supervisor", "show_account_passwords"):
            return [{"account_name": "venmo", "password": "pw-secret"}]
        if (app, api) == ("venmo", "login"):
            return {"access_token": "tok-123456", "token_type": "Bearer"}
        if (app, api) == ("venmo", "show_contacts"):
            return copy.deepcopy(CONTACTS) if kw.get("page_index", 0) == 0 else []
        if (app, api) == ("venmo", "create_transaction"):
            self.db["sent"].append(f"{kw['receiver_email']}:{kw['amount']}")
            return {"message": "Transaction created."}
        if (app, api) == ("supervisor", "complete_task"):
            self.db["done"] = True
            return {"message": "Marked the active task complete."}
        raise KeyError(f"No API named '{api}' found in the {app} app.")

    def execute(self, code):
        self.executed.append(code)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self.shell.user_ns)  # noqa: S102  (test double for AppWorld's REPL)
        except Exception as e:  # noqa: BLE001
            return buf.getvalue() + f"Execution failed. Traceback:\n  File \"<cell>\", line 1\n{type(e).__name__}: {e}"
        return buf.getvalue() or "Execution successful."

    def save_state(self, sid):
        self._saved[sid] = copy.deepcopy(self.db)

    def load_state(self, sid):
        self.db = copy.deepcopy(self._saved[sid])

    def task_completed(self):
        return self.db["done"]


class RecordingClient(ScriptedClient):
    def __init__(self, script, samples=None):
        super().__init__(script, samples)
        self.prompts: list[list[dict]] = []

    def complete(self, messages, temperature=0.0, max_tokens=1024, n=1, seed=None, stop=None):
        if n == 1 and temperature == 0.0:
            self.prompts.append([dict(m) for m in messages])
        return super().complete(messages, temperature, max_tokens, n, seed, stop)


LOGIN = ('profile = apis.supervisor.show_profile()\n'
         'passwords = {p["account_name"]: p["password"] for p in apis.supervisor.show_account_passwords()}\n'
         'token = apis.venmo.login(username=profile["email"], password=passwords["venmo"])["access_token"]')
SHOW = "contacts = apis.venmo.show_contacts(access_token=token)\nprint(contacts)"


def _step(world, ledger, step, code):
    out = world.execute(code)
    return code, out, ledger.record(step, code, out, "Execution failed" in out)


# ------------------------------------------------------------------ the ledger


def test_interpreter_state_reports_only_the_agents_variables():
    w = ExecWorld()
    state = InterpreterState(w)
    w.execute("import math\nx = 1\n_hidden = 2")
    assert state.names() == ["x"]                       # base names, modules and _private names are not the agent's
    assert InterpreterState(SimpleNamespace(task=None)).names() == []   # remote world: no namespace


def test_ledger_records_where_each_value_came_from_without_showing_secrets():
    w = ExecWorld()
    ledger = ProvenanceLedger(InterpreterState(w), "Send 20 dollars to Ana")
    _step(w, ledger, 0, LOGIN)
    _step(w, ledger, 1, SHOW)
    text = ledger.render()
    assert "- contacts: list of 31 dicts with keys: id, email, name (from venmo.show_contacts, step 2)" in text
    assert "- token: str (10 chars) (from venmo.login, step 1)" in text
    assert "passwords: dict, 1 keys: venmo (from supervisor.show_account_passwords, step 1)" in text
    assert "tok-123456" not in text and "pw-secret" not in text   # values of strings never appear
    assert ledger.authenticated_apps() == {"venmo"}
    assert describe(3) == "int 3" and describe(None) == "None"


def test_static_analysis_of_a_step():
    b = bindings("songs = []\nfor i in range(3):\n    songs += apis.spotify.show_song_library(page_index=i)\n"
                 "songs = sorted(songs)\nfor s in songs:\n    print(s['title'])", 4)
    assert b["songs"].apis == ("spotify.show_song_library",) and b["s"].loop and b["s"].derived == ("songs",)
    assert printed("print([t for t in titles], len(rows))") == ({"titles", "rows"}, False)
    assert printed("print(apis.venmo.show_contacts(access_token=t))")[1] is True


# ------------------------------------------------------------------ pruning


def _run_steps(w, ledger, codes):
    turns, metas = [], []
    for i, c in enumerate(codes):
        code, out, meta = _step(w, ledger, i, c)
        turns.append((code, out))
        metas.append(meta)
    return turns, metas


def _governor_after(w, ledger, codes, **kw):
    turns, metas = _run_steps(w, ledger, codes)
    g = ProvenanceGovernor(**kw)
    return g, g.compress(turns, metas, ledger), turns


def test_old_turns_collapse_into_a_summary_that_points_to_variables():
    w = ExecWorld()
    ledger = ProvenanceLedger(InterpreterState(w))
    turns, metas = _run_steps(w, ledger, [LOGIN, SHOW, "x = 1", "y = 2", "z = 3"])
    g = ProvenanceGovernor()
    recent, phi, lossy, summary = g.compress(turns, metas, ledger)
    assert len(turns[1][1]) > 1000 and metas[1] == TurnMeta(step=1, recoverable=True, holders=("contacts",))
    assert recent == turns[2:]                                   # the last three turns stay verbatim
    assert "[2] venmo.show_contacts -> output held in contacts" in summary
    assert "[1] supervisor.show_profile, supervisor.show_account_passwords, venmo.login -> " in summary
    assert "@x.com" not in summary                               # the 1,700-character listing is not repeated
    assert phi == 1.0 and not lossy and g.stats["held"] == 1
    # Once the variable is rebound the value is no longer held: the output is shortened instead, which is a loss.
    code, out, meta = _step(w, ledger, 5, "contacts = None")
    recent, phi, lossy, summary = g.compress(turns + [(code, out)], metas + [meta], ledger)
    assert lossy and phi < 1.0 and "no variable holds this output" in summary


def test_unstored_output_is_shortened_and_marks_the_context_lossy():
    w = ExecWorld()
    ledger = ProvenanceLedger(InterpreterState(w))
    _, (_, phi, lossy, summary), _ = _governor_after(
        w, ledger, [LOGIN, "print(apis.venmo.show_contacts(access_token=token))", "a = 1", "b = 2", "c = 3"])
    assert "[2] venmo.show_contacts -> [{" in summary and "no variable holds this output" in summary
    assert lossy and phi < 1.0


def test_old_errors_and_short_outputs_stay_in_the_summary():
    w = ExecWorld()
    ledger = ProvenanceLedger(InterpreterState(w))
    bad = "print('x' * 300)\nprofile = apis.supervisor.show_profile()\nprofile['mail']"
    _, (_, _, _, summary), turns = _governor_after(w, ledger, [bad, "n = 31\nprint(n)", "a = 1", "b = 2", "c = 3"])
    assert "KeyError" in turns[0][1] and "[1] supervisor.show_profile -> KeyError: 'mail'" in summary
    assert "[2] n = 31 -> 31" in summary


def test_repeated_steps_are_merged_in_the_outline():
    w = ExecWorld()
    ledger = ProvenanceLedger(InterpreterState(w))
    bad = "apis.venmo.no_such_api()"
    _, (_, _, _, summary), _ = _governor_after(w, ledger, [LOGIN, bad, bad, bad, "a = 1", "b = 2", "c = 3"])
    assert "[2-4] venmo.no_such_api -> KeyError: \"No API named 'no_such_api' found in the venmo app.\" (3 times)" \
        in summary


def test_pid_controller_tracks_the_token_setpoint():
    w = ExecWorld()
    ledger = ProvenanceLedger(InterpreterState(w))
    codes = [LOGIN] + [f"print({i}, apis.venmo.show_contacts(access_token=token))" for i in range(6)] + \
        ["a = 1", "b = 2", "c = 3"]
    turns, metas = _run_steps(w, ledger, codes)
    tight = ProvenanceGovernor(budget_tokens=300, match_window=False)
    loose = ProvenanceGovernor(budget_tokens=200_000, match_window=False)
    sizes = []
    for _ in range(10):
        tight.compress(turns, metas, ledger)
        loose.compress(turns, metas, ledger)
        sizes.append(tight.log[-1]["tokens"])
    assert tight._chars == tight.min_chars and loose._chars == loose.max_chars
    assert sizes[-1] < sizes[0]                          # over the setpoint the history shrinks step by step
    matched = ProvenanceGovernor(budget_tokens=200_000)  # default: 0.9 x the size of the baseline's window
    for _ in range(10):
        matched.compress(turns, metas, ledger)
    window = sum(max(1, len(c) // 4) + max(1, len(o[:6000]) // 4) for c, o in turns[-6:])
    assert matched.log[-1]["setpoint"] == int(0.9 * window)
    assert abs(matched.log[-1]["tokens"] - 0.9 * window) / window < 0.1   # tracks it


# ------------------------------------------------------------------ links to O1, O2, O3, G1


def test_g1_provenance_rule_blocks_ids_and_emails_never_observed():
    w = ExecWorld()
    cat = ApiCatalog.from_api_docs(DOCS)
    ledger = ProvenanceLedger(InterpreterState(w), "Send 20 dollars to Ana on venmo.")
    policy = AppWorldPolicy(cat, w.task.instruction, ["venmo"])
    guess = "apis.venmo.create_transaction(receiver_email='ana@venmo.com', amount=20)"
    ok, why = policy.check(guess, ledger)
    assert not ok and "R-AW-PROV" in why
    _step(w, ledger, 0, LOGIN)
    _step(w, ledger, 1, SHOW)                               # ana@x.com is now observed
    assert policy.check("apis.venmo.create_transaction(receiver_email='ana@x.com', amount=20)", ledger)[0]
    assert policy.check(guess)[0]                          # without O4's ledger the rule is off


def test_o3_trial_runs_leave_no_variables_behind():
    w = ExecWorld()
    cat = ApiCatalog.from_api_docs(DOCS)
    w.execute(LOGIN)
    state = InterpreterState(w)
    a = "a = 1\napis.venmo.create_transaction(receiver_email='ana@x.com', amount=20)"
    b = "b = 2\napis.venmo.create_transaction(receiver_email='ana@x.com', amount=20)"
    OutcomeConsensus(cat, n=3).run(w, a, lambda k: [b, b][:k], step=0, state=state)
    assert "a" not in w.shell.user_ns and "b" not in w.shell.user_ns and w.db["sent"] == []
    OutcomeConsensus(cat, n=3).run(w, a, lambda k: [b, b][:k], step=1)
    assert "b" in w.shell.user_ns                           # what AppWorld alone leaves behind


def test_o1_leaves_out_login_docs_for_apps_already_logged_in():
    r = SessionApiRetriever(ApiCatalog.from_api_docs(DOCS), k=2)
    def keys(specs):
        return {s.key for s in specs}

    assert "venmo.login" in keys(r.retrieve("send money to a venmo contact"))
    assert "venmo.login" not in keys(r.retrieve("send money to a venmo contact", authenticated={"venmo"}))


def test_o2_treats_a_lossy_context_as_risky():
    r = TierRouter(seed=1, cost_penalty=0.0, risk_margin=0.35)
    calm, lossy = TierRouter.context(False, False, 3), TierRouter.context(False, False, 3, lossy=True)
    assert len(calm) == 4 and lossy[3]
    assert [r.choose(lossy) for _ in range(300)].count("large") > [r.choose(calm) for _ in range(300)].count("large")


# ------------------------------------------------------------------ the agent


def _agent(world, script, cfg=None, samples=None):
    cli = RecordingClient(script, samples=samples)
    return ACRSAppWorldAgent(world, cli, cli, cfg or AgentConfig(max_steps=8)), cli


def test_full_agent_with_provenance_on_a_real_namespace():
    w = ExecWorld()
    pay = "apis.venmo.create_transaction(receiver_email='ana@x.com', amount=20)"
    agent, cli = _agent(w, [LOGIN, SHOW, pay, "apis.supervisor.complete_task()"])
    log = agent.run()
    assert log.completed and w.db["sent"] == ["ana@x.com:20"]
    assert log.login_calls == 1 and log.prov_denials == 0 and log.refetch_calls == 0
    assert "Variables in your Python session" in cli.prompts[2][-1]["content"]
    assert "contacts: list of 31 dicts" in cli.prompts[2][-1]["content"]
    assert log.history_tokens > 0 and "held" in log.o4


def test_agent_blocks_a_guessed_email_then_succeeds():
    w = ExecWorld()
    guess = "apis.venmo.create_transaction(receiver_email='ana@venmo.com', amount=20)"
    pay = "apis.venmo.create_transaction(receiver_email='ana@x.com', amount=20)"
    agent, _ = _agent(w, [LOGIN, guess, SHOW, pay, "apis.supervisor.complete_task()"])
    log = agent.run()
    assert log.prov_denials == 1 and log.policy_denials == 1
    assert w.db["sent"] == ["ana@x.com:20"] and log.completed


def test_o4_modes_and_runner_configs():
    w = ExecWorld()
    assert isinstance(_agent(w, [])[0].governor, ProvenanceGovernor)
    assert isinstance(_agent(w, [], AgentConfig(o4_mode="fidelity"))[0].governor, HistoryGovernor)
    off = _agent(w, [], AgentConfig(o4_history=False))[0]
    assert isinstance(off.governor, HistoryGovernor) and off.ledger is None
    cfg = _runner().CONFIGS
    assert cfg["o4_only"].o4_history and not (cfg["o4_only"].o1_retrieval or cfg["o4_only"].o3_consensus)
    assert cfg["o4_only"].o4_mode == "provenance" and cfg["o4_fidelity_only"].o4_mode == "fidelity"
    assert not cfg["no_o4"].o4_history


def test_trace_records_each_components_decision():
    w = ExecWorld()
    guess = "apis.venmo.create_transaction(receiver_email='ana@venmo.com', amount=20)"
    pay = "apis.venmo.create_transaction(receiver_email='ana@x.com', amount=20)"
    agent, _ = _agent(w, [LOGIN, guess, SHOW, pay, "apis.supervisor.complete_task()"],
                      AgentConfig(max_steps=8, trace=True))
    trace = agent.run().trace
    assert [t["outcome"] for t in trace] == ["executed", "denied", "executed", "executed", "executed"]
    assert trace[0]["o3"] == "exempt" and trace[1]["g1"] == "R-AW-PROV"
    assert trace[3]["o3"].startswith("certified") and trace[3]["g1"] == "allow"
    assert trace[3]["o4_ledger_vars"] >= 4 and "venmo.login" not in trace[3]["o1_docs"]   # logged in: no login doc
    assert set(trace[0]) >= {"arm", "risk", "o1_docs", "o4_recoverability", "tokens", "code", "output"}
    assert not ACRSAppWorldAgent(ExecWorld(), *[RecordingClient([LOGIN])] * 2, AgentConfig(max_steps=1)).run().trace


def test_runner_spreads_workers_over_model_replicas():
    tier = _runner().tier
    spec = "Qwen/Qwen3-8B-AWQ@http://localhost:8001/v1|http://localhost:8003/v1"
    assert [tier(spec, "large", i).base_url for i in range(4)] == ["http://localhost:8001/v1", "http://localhost:8003/v1"] * 2
    assert tier("m@http://x/v1", "small", 5).base_url == "http://x/v1"
