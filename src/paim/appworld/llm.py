"""Minimal OpenAI-compatible chat client (vLLM / any OpenAI-style server).

Uses only the standard library so it runs unchanged on Kaggle/Colab. Every call
records prompt/completion tokens and latency so O2's cost and the paper's cost
tables come from measured numbers, not estimates.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ModelTier:
    name: str                 # served model id, e.g. Qwen/Qwen3-14B-AWQ
    base_url: str             # e.g. http://localhost:8000/v1
    cost_per_1k_in: float = 0.0
    cost_per_1k_out: float = 0.0
    tier: str = "large"       # small | large
    extra_body: dict = field(default_factory=dict)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cost: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.seconds += other.seconds
        self.cost += other.cost

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ChatClient:
    def __init__(self, tier: ModelTier, timeout: float = 600.0, retries: int = 3) -> None:
        self.tier = tier
        self.timeout = timeout
        self.retries = retries
        self.usage = Usage()
        self._lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        n: int = 1,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[str]:
        body: dict[str, Any] = {
            "model": self.tier.name,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = list(stop)
        body.update(self.tier.extra_body)
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("ACRS_LLM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = self.tier.base_url.rstrip("/") + "/chat/completions"
        last: Exception | None = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
                u = resp.get("usage", {}) or {}
                inc = Usage(
                    calls=1,
                    prompt_tokens=int(u.get("prompt_tokens", 0)),
                    completion_tokens=int(u.get("completion_tokens", 0)),
                    seconds=time.time() - t0,
                )
                inc.cost = (
                    inc.prompt_tokens / 1000 * self.tier.cost_per_1k_in
                    + inc.completion_tokens / 1000 * self.tier.cost_per_1k_out
                )
                with self._lock:
                    self.usage.add(inc)
                return [c["message"]["content"] or "" for c in resp.get("choices", [])]
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last}")


__all__ = ["ChatClient", "ModelTier", "Usage"]
