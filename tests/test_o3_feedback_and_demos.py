"""Fixes from the Nov 2026 dev pilot: O3 error feedback, login exemption, and train-split worked examples.

Offline: fake world, fake AppWorld data directory, no appworld package, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paim.appworld import ACRSAppWorldAgent, AgentConfig, ApiCatalog, OutcomeConsensus
from paim.appworld.demos import (free_names, load_demo, missing_names, rejection, render_demos, select_demos,
                                 solution_body, split_ids, strip_comments)

from test_appworld import DOCS, FakeWorld, ScriptedClient
from test_kaggle_harness import _runner

BAD = "raise X; apis.venmo.create_transaction(amount=1)"


# ------------------------------------------------------------------ O3 feedback


def test_o3_all_failed_shows_the_error_and_commits_nothing():
    w = FakeWorld()
    cli = ScriptedClient([BAD, "apis.supervisor.complete_task()"], samples=[BAD, BAD])
    agent = ACRSAppWorldAgent(w, cli, cli, AgentConfig(max_steps=4))
    log = agent.run()
    first_output = agent.turns[0][1]
    assert "Traceback: boom" in first_output                  # the agent now sees what went wrong
    assert "Nothing was committed" in first_output and "2 sampled alternatives" in first_output
    assert w.state["sent"] == []                              # trial runs were rolled back
    assert log.consensus["abstain_all_failed"] == 1
    assert log.errors >= 1 and "Execution failed. Traceback: boom" in log.error_samples
    assert log.completed and log.steps == 2                   # the abstained turn counts as a step


def test_o3_disagreement_message_says_nothing_was_committed():
    w = FakeWorld()
    oc = OutcomeConsensus(ApiCatalog.from_api_docs(DOCS), n=3)
    res = oc.run(w, "apis.venmo.create_transaction(amount=1)",
                 lambda k: ["apis.venmo.create_transaction(amount=2)", "apis.venmo.create_transaction(amount=3)"][:k],
                 step=0)
    assert not res.certified and not res.all_failed
    msg = ACRSAppWorldAgent._abstain_output(res)
    assert "candidates disagree" in msg and "Nothing was committed" in msg


def test_o3_login_is_exempt_from_the_vote():
    oc = OutcomeConsensus(ApiCatalog.from_api_docs(DOCS), n=3)
    assert not oc.needed("tok = apis.venmo.login(username='joy@x.com')")
    assert oc.stats["exempt_steps"] == 1
    assert oc.needed("tok = apis.venmo.login(username='joy@x.com')\napis.venmo.create_transaction(amount=20)")
    # Candidates that spell the login differently but make the same write agree.
    w = FakeWorld()
    a = "t = apis.venmo.login(username='joy@x.com')\napis.venmo.create_transaction(amount=20)"
    b = "t = apis.venmo.login(username=profile['email'])\napis.venmo.create_transaction(amount=20)"
    res = oc.run(w, a, lambda k: [b, b][:k], step=1)
    assert res.certified and res.winner.code == a
    assert OutcomeConsensus(ApiCatalog.from_api_docs(DOCS), exempt_apis=()).needed("apis.venmo.login(username='a')")


def test_agent_runs_a_login_once():
    w = FakeWorld()
    cli = ScriptedClient(["tok = apis.venmo.login(username='joy@x.com')", "apis.supervisor.complete_task()"])
    log = ACRSAppWorldAgent(w, cli, cli, AgentConfig(max_steps=4)).run()
    assert sum("login" in c for c in w.executed) == 1         # was 4: three candidates plus the commit
    assert log.consensus["exempt_steps"] == 1 and log.completed


# ------------------------------------------------------------------ worked examples

SOLUTION = '''# Canary String: {canary}
from appworld.collections.apis import ApiCollection
from appworld.requester import Requester


def solution(apis: ApiCollection, requester: Requester):
    """Solve the task."""
    # get the supervisor's profile and passwords.
    profile = apis.supervisor.show_profile()
    passwords = {{p["account_name"]: p["password"] for p in apis.supervisor.show_account_passwords()}}
    token = apis.{app}.login(username=profile["email"], password=passwords["{app}"])["access_token"]
    items = []
    page_index = 0
    while True:  # all pages
        page = apis.{app}.{api}(access_token=token, page_index=page_index)
        if not page:
            break
        items.extend(page)
        page_index += 1
    label = "count # of items"
    {extra}apis.supervisor.complete_task(answer=len(items))
'''


def _task(data: Path, tid: str, app: str, api: str = "show_items", extra: str = "", instruction: str = ""):
    d = data / "tasks" / tid / "ground_truth"
    d.mkdir(parents=True)
    (data / "tasks" / tid / "specs.json").write_text(json.dumps({"instruction": instruction or f"Count my {app} items.",
                                                                 "db_version": "0.1"}))
    (d / "compiled_solution.py").write_text(SOLUTION.format(canary="xyz", app=app, api=api, extra=extra))


@pytest.fixture
def data(tmp_path):
    data = tmp_path / "data"
    (data / "datasets").mkdir(parents=True)
    (data / "datasets" / "train.txt").write_text("82e2fac_1\naaa_1\naaa_2\nbbb_1:tagged\nccc_1\nddd_1\nfff_1\n")
    (data / "datasets" / "dev.txt").write_text("eee_1\n")
    _task(data, "82e2fac_1", "spotify")
    _task(data, "aaa_1", "spotify")                               # same app as the preferred demo
    _task(data, "aaa_2", "venmo")                                 # same scenario as aaa_1
    _task(data, "bbb_1", "gmail")
    _task(data, "ccc_1", "venmo", extra="requester.post('/x')\n    ")  # uses AppWorld's low-level requester
    _task(data, "fff_1", "gmail", extra="x = find_all_from_pages(apis.gmail.show_inbox)\n    ")  # generator helper
    _task(data, "eee_1", "todoist")                               # dev: must never be used
    # ddd_1 has no ground truth
    return data


def test_solution_body_is_the_dedented_function_body():
    body = solution_body(SOLUTION.format(canary="xyz", app="spotify", api="show_items", extra=""))
    assert body.startswith("# get the supervisor's profile")
    assert "def solution" not in body and "Canary" not in body and "Solve the task" not in body
    assert body.splitlines()[-1] == "apis.supervisor.complete_task(answer=len(items))"
    assert solution_body("x = 1\n") is None and solution_body("def solution(:") is None


def test_strip_comments_keeps_hashes_inside_strings():
    code = 'a = "x # y"  # note\n# whole line\n\n\n\nb = 2'
    assert strip_comments(code) == 'a = "x # y"\n\nb = 2'


def test_select_demos_prefers_authors_tasks_and_never_uses_held_out(data):
    assert split_ids(data, "train")[3] == "bbb_1"                 # tags removed
    demos = select_demos(data, 3, preferred=("eee_1", "82e2fac_1"))
    ids = [d.task_id for d in demos]
    assert ids[0] == "82e2fac_1"                                  # the preferred train task comes first
    assert "eee_1" not in ids and "ccc_1" not in ids and "ddd_1" not in ids and "fff_1" not in ids
    assert len({t.split("_")[0] for t in ids}) == len(ids)       # one per scenario
    assert set(ids) == {"82e2fac_1", "aaa_2", "bbb_1"}            # aaa_1 repeats spotify, so aaa_2 stands in
    assert all(d.apps[0] != "spotify" for d in demos[1:])         # new apps before repeated ones
    assert select_demos(data, 3) == select_demos(data, 3)        # deterministic
    assert select_demos(data, 0) == []
    assert rejection(load_demo(data, "ccc_1")) == "uses names the agent does not have"
    assert rejection(load_demo(data, "fff_1")) == "uses names the agent does not have"
    assert load_demo(data, "ddd_1") is None


def test_missing_names_follow_the_repl():
    code = ("import math\nfrom collections import Counter as C\nrows = [r for r in apis.a.b()]\n"
            "def f(x, *a, k=1, **kw):\n    return x\ntry:\n    y = DateTime.now()\nexcept Exception as e:\n"
            "    print(e)\nz = main_user.email + public_data.name + len(rows)")
    assert free_names(code) == {"apis", "DateTime", "main_user", "public_data"}
    assert missing_names(code) == {"main_user", "public_data"}


def test_render_demos_and_agent_prompt(data):
    text = render_demos(select_demos(data, 2))
    assert text.count("```python") == 2 and "Example 1. Task: Count my spotify items." in text
    assert render_demos([]) == ""
    w = FakeWorld()
    cli = ScriptedClient(["apis.supervisor.complete_task()"])
    agent = ACRSAppWorldAgent(w, cli, cli, AgentConfig.baseline(), demos=text)
    system = agent._messages()[0]["content"]
    assert text in system and system.index(text) < system.index("Relevant APIs:")
    plain = ACRSAppWorldAgent(FakeWorld(), cli, cli, AgentConfig.baseline())._messages()[0]["content"]
    assert "Worked examples" not in plain


def test_experiment_name_records_few_shot():
    run = _runner()
    assert run.experiment_name("baseline", "", 0, "dev") == "acrs_baseline_s0_dev"
    assert run.experiment_name("full", "p1", 2, "dev", few_shot=3) == "acrs_full_p1_fs3_s2_dev"
