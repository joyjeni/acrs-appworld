"""A made-up episode through the real agent loop, for the viva dashboard's flow demo (no AppWorld content, no model).

    python experiments/demo_trace.py --out demo_trace.json

Nothing here comes from AppWorld. The task, the apps, their API descriptions and the data are invented for the demo,
so the trace can be published. The world is a small in-memory stand-in for AppWorld: code runs for real in one
persistent Python namespace, and save_state/load_state restore the app data but not the Python variables, as
AppWorld 0.1.3 does.

The model is a fixed script, one code block per step. Every decision in the trace is made by the real components
in paim.appworld: the API docs O1 put in the prompt, the tier O2 chose and why, G1's verdict, O3's vote, and what O4
showed of the past. The script is written so that one short episode exercises every component: a guessed email
that G1 blocks, a mistyped API, a long listing that a variable then holds, and a payment that O3 votes on.

History sizes are measured as experiments/replay_o4.py measures them on real AppWorld logs (characters / 4), for
O4 and for the baseline's last-six-turns window over the same turns.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paim.appworld import ACRSAppWorldAgent, AgentConfig  # noqa: E402
from paim.appworld.llm import ChatClient, ModelTier  # noqa: E402

TASK = "Pay Ana Silva back $20 for lunch on Venmo, with the note 'lunch'."


def _p(name: str, kind: str = "string", required: bool = True) -> dict:
    return {"name": name, "type": kind, "required": required}


DOCS = {
    "api_docs": {
        "show_api_descriptions": {"method": "GET", "description": "List an app's APIs, one line each.",
                                  "parameters": [_p("app_name")]},
        "show_api_doc": {"method": "GET", "description": "Show the full specification of one API.",
                         "parameters": [_p("app_name"), _p("api_name")]},
    },
    "supervisor": {
        "show_profile": {"method": "GET", "description": "Show your profile."},
        "show_account_passwords": {"method": "GET", "description": "Show the passwords of your app accounts."},
        "complete_task": {"method": "POST", "description": "Mark the task as complete.",
                          "parameters": [_p("answer", "any", False)]},
    },
    "venmo": {
        "login": {"method": "POST", "description": "Log in to Venmo and get an access token.",
                  "parameters": [_p("username"), _p("password")]},
        "show_profile": {"method": "GET", "description": "Show your Venmo profile.",
                         "parameters": [_p("access_token")]},
        "show_balance": {"method": "GET", "description": "Show your Venmo balance.",
                         "parameters": [_p("access_token")]},
        "show_contacts": {"method": "GET", "description": "List your Venmo contacts.",
                          "parameters": [_p("access_token"), _p("page_index", "integer", False)]},
        "show_transactions": {"method": "GET", "description": "List your recent Venmo transactions.",
                              "parameters": [_p("access_token"), _p("page_index", "integer", False)]},
        "create_transaction": {"method": "POST", "description": "Send money to another Venmo user.",
                               "parameters": [_p("access_token"), _p("receiver_email"), _p("amount", "number"),
                                              _p("description", "string", False)]},
        "add_contact": {"method": "POST", "description": "Add a Venmo user to your contacts.",
                        "parameters": [_p("access_token"), _p("email")]},
        "delete_contact": {"method": "DELETE", "description": "Remove a user from your contacts.",
                           "parameters": [_p("access_token"), _p("email")]},
    },
}

FIRST = ["Maya", "Leo", "Priya", "Omar", "Grace", "Ivan", "Chloe", "Ravi", "Nina", "Theo", "Zara", "Ken", "Lena",
         "Marco", "Aisha", "Ben", "Hana", "Luis", "Mira", "Owen"]
LAST = ["Park", "Novak", "Rao", "Haddad", "Kim", "Petrov", "Dubois", "Iyer", "Costa", "Wright"]
CONTACTS = [{"id": 1001 + i, "name": f"{FIRST[i % 20]} {LAST[(i * 3) % 10]}",
             "email": f"{FIRST[i % 20].lower()}.{LAST[(i * 3) % 10].lower()}@example.com"} for i in range(39)]
CONTACTS.insert(23, {"id": 1040, "name": "Ana Silva", "email": "ana.silva@example.com"})
PASSWORD = "demo-only"


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


class DemoWorld:
    """Runs code in one namespace. App data is saved and restored; Python variables are not (as in AppWorld)."""

    def __init__(self):
        self.task = SimpleNamespace(
            id="demo_1", instruction=TASK, allowed_apps=["venmo"], api_docs=DOCS,
            supervisor=SimpleNamespace(first_name="Sam", last_name="Lee", email="sam.lee@example.com",
                                       phone_number="5550100"))
        self.db = {"sent": [], "done": False}
        self._saved = {}
        self.shell = SimpleNamespace(user_ns={"apis": _Apis(self), "json": json, "In": [], "Out": {}})

    def handle(self, app, api, kw):
        if app not in DOCS or api not in DOCS[app]:
            raise Exception(f"Response status code is 404:\n{{'message': \"No API named '{api}' found in the {app} app.\"}}")
        if app == "api_docs":
            target = DOCS.get(kw.get("app_name"), {})
            if api == "show_api_descriptions":
                return [{"name": n, "description": s["description"]} for n, s in target.items()]
            spec = target[kw["api_name"]]
            return {"app_name": kw["app_name"], "api_name": kw["api_name"], "method": spec["method"],
                    "description": spec["description"],
                    "parameters": [dict(p, description=f"The {p['name'].replace('_', ' ')}.") for p in
                                   spec.get("parameters", [])],
                    "response_schemas": {"success": [{"id": 1, "name": "string", "email": "user@example.com"}],
                                         "failure": {"message": "string"}}}
        if (app, api) == ("supervisor", "show_profile"):
            return {"first_name": "Sam", "last_name": "Lee", "email": "sam.lee@example.com", "phone_number": "5550100"}
        if (app, api) == ("supervisor", "show_account_passwords"):
            return [{"account_name": "venmo", "password": PASSWORD}, {"account_name": "gmail", "password": PASSWORD}]
        if (app, api) == ("supervisor", "complete_task"):
            self.db["done"] = True
            return {"message": "Marked the active task complete."}
        if api == "login":
            if kw.get("password") != PASSWORD:
                raise Exception("Response status code is 401:\n{'message': 'Invalid credentials.'}")
            return {"access_token": "demo-token-7f3a", "token_type": "Bearer"}
        if kw.get("access_token") != "demo-token-7f3a":
            raise Exception("Response status code is 401:\n{'message': 'Invalid or missing access token.'}")
        if api == "show_contacts":
            return copy.deepcopy(CONTACTS) if kw.get("page_index", 0) == 0 else []
        if api == "create_transaction":
            if not any(c["email"] == kw.get("receiver_email") for c in CONTACTS):
                raise Exception("Response status code is 404:\n{'message': 'No Venmo user with this email.'}")
            self.db["sent"].append(f"{kw['receiver_email']}:{kw['amount']}")
            return {"message": f"Sent ${float(kw['amount']):.2f} to {kw['receiver_email']}.", "transaction_id": 8812}
        return {"message": "OK"}

    def execute(self, code):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self.shell.user_ns)  # noqa: S102  (stand-in for AppWorld's REPL)
        except Exception as e:  # noqa: BLE001
            return buf.getvalue() + f"Execution failed. Traceback:\n  File \"<cell>\", line 1, in <module>\n{type(e).__name__}: {e}"
        return buf.getvalue() or "Execution successful."

    def save_state(self, sid):
        self._saved[sid] = copy.deepcopy(self.db)

    def load_state(self, sid):
        self.db = copy.deepcopy(self._saved[sid])

    def task_completed(self):
        return self.db["done"]


LOGIN = ('profile = apis.supervisor.show_profile()\n'
         'passwords = {p["account_name"]: p["password"] for p in apis.supervisor.show_account_passwords()}\n'
         'token = apis.venmo.login(username=profile["email"], password=passwords["venmo"])["access_token"]')
PAY = ('apis.venmo.create_transaction(access_token=token, receiver_email="ana.silva@example.com", amount=20, '
       'description="lunch")')
DONE = "apis.supervisor.complete_task()"

# (what the step is for, the code block the scripted model writes, what the step puts to the components)
SCRIPT = [
    ("List Venmo's APIs", 'print(apis.api_docs.show_api_descriptions(app_name="venmo"))',
     "The agent reads Venmo's API list. The listing is printed straight from an API call, so no variable holds it."),
    ("Get credentials and log in", LOGIN,
     "A login changes no task state worth voting on, so O3 lets it through. The values it binds enter the ledger "
     "with the call that produced them."),
    ("Pay a guessed email", 'apis.venmo.create_transaction(access_token=token, receiver_email="ana@venmo.com", '
                            'amount=20, description="lunch")',
     "The model guesses Ana's address instead of looking it up. G1 checks the address against everything the "
     "agent has been shown before the block runs."),
    ("Call an API that does not exist", 'matches = apis.venmo.search_users(access_token=token, query="Ana Silva")\n'
                                        'print(matches)',
     "O3 treats an API it does not know as possibly state-changing, so it runs three candidates from a saved "
     "state before anything is committed."),
    ("Read the contacts API's doc", 'print(apis.api_docs.show_api_doc(app_name="venmo", api_name="show_contacts"))',
     "After an error, and with part of the past cut from the prompt, O2 weighs the step as risky."),
    ("List all contacts", "contacts = apis.venmo.show_contacts(access_token=token)\nprint(contacts)",
     "A long listing, but bound to a variable: once it is older than three steps, O4 shows a pointer to it "
     "instead of the text."),
    ("Find Ana in the list", 'ana = next(c for c in contacts if c["name"] == "Ana Silva")\nprint(ana)',
     "The agent reads her real address from the listing, so it now counts as observed."),
    ("Pay the observed email", PAY,
     "G1 checks the address again. O3 samples two alternatives and runs all three from a saved state; the trial "
     "payments are rolled back and only a certified majority is committed."),
    ("Finish", DONE,
     "Completing the task is a state change too, so O3 votes on it before it runs."),
]
# O3's sampled alternatives for the steps it votes on (index into SCRIPT). O3 treats an API it does not know as
# possibly state-changing, so it also votes on the mistyped call; there every candidate raises. At the payment, one
# of the two alternatives pays $25.
SAMPLES = {
    3: ['matches = apis.venmo.search_users(access_token=token, query="Ana")\nprint(matches)',
        'matches = apis.venmo.find_user(access_token=token, name="Ana Silva")\nprint(matches)'],
    7: [PAY, PAY.replace("amount=20", "amount=25")],
    8: [DONE, DONE],
}


class DemoClient(ChatClient):
    """The scripted model. Both tiers share one script; each counts its own tokens (characters / 4)."""

    def __init__(self, tier: str, shared: dict):
        super().__init__(ModelTier(name=f"scripted-{tier}", base_url="http://demo", tier=tier))
        self.shared = shared

    def complete(self, messages, temperature=0.0, max_tokens=1024, n=1, seed=None, stop=None):
        self.usage.calls += 1
        self.usage.prompt_tokens += sum(len(m["content"]) for m in messages) // 4
        if n > 1 or temperature > 0:   # O3 samples alternatives to the block the script gave at this step
            out = SAMPLES.get(self.shared["step"], [self.shared["last"]] * n)[:n]
        else:
            self.shared["step"] += 1
            step = self.shared["step"]
            out = [SCRIPT[step][1] if step < len(SCRIPT) else DONE]
            self.shared["last"] = out[0]
        self.usage.completion_tokens += sum(len(c) for c in out) // 4
        return [f"```python\n{c}\n```" for c in out]


def size(history, extra: str = "") -> int:
    """As experiments/replay_o4.py: estimated tokens of the history messages (characters / 4)."""
    chars = sum(len(f"```python\n{code}\n```") + len(f"Output:\n{out[:6000]}") for code, out in history)
    return (chars + (len(extra) + 2 if extra else 0)) // 4


def run(seed: int = 0) -> dict:
    world = DemoWorld()
    shared = {"step": -1, "last": ""}
    agent = ACRSAppWorldAgent(world, DemoClient("small", shared), DemoClient("large", shared),
                              AgentConfig(max_steps=len(SCRIPT) + 2, trace=True), seed=seed)
    views: list[dict] = []
    seen: dict = {}

    compress = agent.governor.compress

    def watched_compress(turns, metas, ledger):
        out = compress(turns, metas, ledger)
        seen["history"], seen["summary"] = out[0], out[3]
        return out

    agent.governor.compress = watched_compress
    messages = agent._messages

    def watched_messages():
        window = size(agent.turns[-6:])          # what the baseline's window would show for the same turns
        msgs = messages()
        ledger = agent.ledger.render()
        # As in replay_o4.py: the window keeps a past observation usable only while it is among the last six turns.
        big = [len(o.strip()) > agent.governor.inline_chars for _, o in agent.turns]
        views.append({"window": window, "o4": size(seen["history"], ledger + seen["summary"]),
                      "window_phi": round(sum(big[-6:]) / sum(big), 3) if any(big) else 1.0,
                      "outline": seen["summary"], "ledger": ledger, "verbatim": len(seen["history"])})
        return msgs

    agent._messages = watched_messages
    vote = agent.consensus.run
    votes: dict[int, list[str]] = {}

    def watched_vote(world_, first, sample, step, state=None):
        res = vote(world_, first, sample, step, state=state)
        votes[step + 1] = [f"{'raised' if c.error else 'ran'}: {', '.join(c.signature) or '(no state change)'}"
                           for c in res.candidates]
        return res

    agent.consensus.run = watched_vote
    log = agent.run()

    rows = []
    catalog = sum(len(apis) for apis in DOCS.values())
    for (title, _, purpose), t, v in zip(SCRIPT, log.trace, views):
        risk = [k.replace("_", " ") for k, on in t["risk"].items() if on]
        rows.append({
            "step": t["step"], "title": title, "purpose": purpose, "code": t["code"], "output": t["output"],
            "outcome": t["outcome"], "error": t["error"],
            "o1_docs": ", ".join(t["o1_docs"]), "o1_doc_count": len(t["o1_docs"]), "o1_catalog_apis": catalog,
            "o1_login_skipped": ", ".join(t["o1_login_skipped"]) or "none",
            "o2_tier": t["arm"], "o2_risk": ", ".join(risk) or "none",
            "g1": t["g1"], "o3": t["o3"], "o3_candidates": "\n".join(votes.get(t["step"], [])),
            "o4_recoverability": t["o4_recoverability"], "window_recoverability": v["window_phi"],
            "o4_lossy": t["o4_lossy"],
            "o4_outline": v["outline"], "o4_ledger": v["ledger"], "o4_outline_lines": t["o4_outline_lines"],
            "o4_verbatim_turns": v["verbatim"], "o4_ledger_vars": t["o4_ledger_vars"],
            "history_tokens_o4": v["o4"], "history_tokens_window": v["window"], "model_tokens": t["tokens"],
        })
    return {"task": TASK, "completed": log.completed, "payments": world.db["sent"], "steps": log.steps,
            "policy_denials": log.policy_denials, "prov_denials": log.prov_denials, "login_calls": log.login_calls,
            "consensus": log.consensus, "router": log.router, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    result = run(args.seed)
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    brief = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(brief, indent=1))
    for r in result["rows"]:
        print(r["step"], r["title"], "|", r["o2_tier"], r["o2_risk"], "|", r["g1"], "|", r["o3"], "|",
              r["outcome"], "| phi", r["o4_recoverability"], "vs", r["window_recoverability"], "| hist",
              r["history_tokens_o4"], "vs", r["history_tokens_window"])


if __name__ == "__main__":
    main()
