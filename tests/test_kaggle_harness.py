"""Harness fixes found while running on Kaggle (Nov 2026). Offline: fake world, no appworld, no GPU."""

from __future__ import annotations

import importlib.util
import io
import json
import time
import urllib.error
from pathlib import Path

import pytest

from paim.appworld import ACRSAppWorldAgent, AgentConfig
from paim.appworld.catalog import SILENT_OUTPUT_HINT, show_last_expression, with_silence_hint
from paim.appworld.llm import ChatClient, ModelTier
from paim.common import clock

from test_appworld import FakeWorld, ScriptedClient

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("appworld_run", ROOT / "experiments" / "appworld_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clock_survives_frozen_time(monkeypatch):
    """AppWorld freezes time.time / perf_counter in-process; latencies must still be real."""
    monkeypatch.setattr(time, "time", lambda: 1684411200.0)
    monkeypatch.setattr(time, "perf_counter", lambda: 0.0)
    a = clock.perf_counter()
    sum(range(20000))
    assert clock.perf_counter() > a and clock.wall_time() > 1.7e9


def test_trailing_expression_is_printed():
    assert show_last_expression("p = apis.supervisor.show_account_passwords()\np") .endswith("print(p)")
    assert show_last_expression("print(1)") == "print(1)"
    assert show_last_expression("apis.supervisor.complete_task(answer=3)") == "apis.supervisor.complete_task(answer=3)"
    assert show_last_expression("for i in range(2):\n    i") == "for i in range(2):\n    i"
    assert "print(a)" in show_last_expression("a = 2; a")
    assert show_last_expression("def (") == "def ("
    multi = "apis.venmo.login(\n    username='a',\n    password='b')"
    assert show_last_expression(multi).startswith("print(apis.venmo.login(")
    assert show_last_expression("p = 1\np  # the passwords") == "p = 1\nprint(p)  # the passwords"
    assert show_last_expression("s = 'é'\ns  # é") == "s = 'é'\nprint(s)  # é"


def test_silent_api_call_gets_a_hint():
    assert with_silence_hint("t = apis.venmo.login(username='a')", "Execution successful.").endswith(SILENT_OUTPUT_HINT)
    assert with_silence_hint("x = 1", "Execution successful.") == "Execution successful."
    assert with_silence_hint("print(apis.venmo.login())", "{'token': 1}") == "{'token': 1}"


def test_agent_executes_the_visible_form():
    w = FakeWorld()
    cli = ScriptedClient(["apis.supervisor.show_account_passwords()", "apis.supervisor.complete_task()"])
    ACRSAppWorldAgent(w, cli, cli, AgentConfig.baseline()).run()
    assert w.executed[0] == "print(apis.supervisor.show_account_passwords())"


def test_context_limit_drops_oldest_turns_for_every_config():
    w = FakeWorld()
    cli = ScriptedClient(["apis.supervisor.show_account_passwords()"] * 3 + ["apis.supervisor.complete_task()"])
    agent = ACRSAppWorldAgent(w, cli, cli, AgentConfig(o1_retrieval=False, o2_routing=False, o3_consensus=False,
                                                       o4_history=False, g1_policy=False, context_limit_tokens=1200,
                                                       max_tokens_step=200))
    agent.turns = [(f"code{i}", "z" * 1500) for i in range(6)]
    msgs = agent._messages()
    assert len(msgs) == 4                     # system + task + the latest turn: never dropped, even if over
    assert msgs[-1]["content"].startswith("Output:") and "code5" in msgs[-2]["content"]   # latest turn kept
    assert "not shown: context limit" in msgs[1]["content"] and agent.log.window_dropped == 5
    off = ACRSAppWorldAgent(FakeWorld(), cli, cli, AgentConfig.baseline())
    off.turns = list(agent.turns)
    assert off.log.window_dropped == 0 and len(off._messages()) > len(msgs)


def test_http_error_keeps_server_message(monkeypatch):
    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"message": "maximum context length is 16384 tokens"}'))
    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("time.sleep", lambda s: None)
    cli = ChatClient(ModelTier(name="m", base_url="http://127.0.0.1:1/v1"), retries=1)
    with pytest.raises(RuntimeError, match="maximum context length"):
        cli.complete([{"role": "user", "content": "hi"}])


def test_runner_deadline_and_recorded_result(monkeypatch):
    run = _runner()
    monkeypatch.delenv("ACRS_DEADLINE", raising=False)
    assert not run.deadline_reached()
    monkeypatch.setenv("ACRS_DEADLINE", "1")
    assert run.deadline_reached()

    class Tracker:
        success, pass_count, fail_count, num_tests = False, 2, 1, 3

    class W:
        def evaluate(self):
            return Tracker()

    class Broken:
        def evaluate(self):
            raise RuntimeError("no ground truth on this split")

    assert run.task_result(W()) == {"success": False, "pass_count": 2, "fail_count": 1, "num_tests": 3}
    assert run.task_result(Broken()) == {}
