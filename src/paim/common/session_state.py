"""The session state substrate shared by all four objectives.

Objective O1 defines and parameterises this structure; O2 reads its risk and
budget signals to build routing contexts, O3 writes execution evidence into it,
and O4 treats it as a pruneable context source. It is the single piece of state
that makes the framework a closed loop instead of a four-stage cascade.

State components
---------------
``memory``           exponentially decayed term vector of everything the user asked
``satisfied``        term vector of intents already served (drives redundancy penalty)
``tool_usage``       how often each tool was invoked in this session
``tool_failures``    per-tool failure counts (drives the failure penalty)
``risk``             running estimate of how error-prone this session has become
``families``         decayed counts of the tool families (API providers) committed so far (SATR v2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from paim.common.text import add_scaled, bag, cosine, decay_inplace


@dataclass
class SessionState:
    """Mutable per-session state with temporal decay.

    Parameters
    ----------
    decay:
        Per-turn multiplicative decay applied to the intent memory. ``1.0``
        recovers a stateless bag-of-history; small values make the reranker
        myopic. Ablated in ``experiments/ablation``.
    satisfied_decay:
        Decay of the "already served" vector. Slower than ``decay`` because a
        satisfied sub-goal stays satisfied.
    """

    decay: float = 0.72
    satisfied_decay: float = 0.9
    memory: dict[str, float] = field(default_factory=dict)
    satisfied: dict[str, float] = field(default_factory=dict)
    tool_usage: dict[str, int] = field(default_factory=dict)
    tool_failures: dict[str, int] = field(default_factory=dict)
    categories: dict[str, float] = field(default_factory=dict)
    selected_doc_ids: list[str] = field(default_factory=list)
    families: dict[str, float] = field(default_factory=dict)
    family_decay: float = 0.8
    turn: int = 0
    risk: float = 0.0
    tokens_spent: int = 0
    cost_spent: float = 0.0

    # -- writes ------------------------------------------------------------
    def observe_intent(self, text: str, weight: float = 1.0) -> None:
        """Record a new user intent (a sub-goal opening this turn)."""
        decay_inplace(self.memory, self.decay)
        add_scaled(self.memory, bag(text), weight)

    def observe_satisfaction(self, text: str, weight: float = 1.0) -> None:
        """Record that an intent has been served, so it stops attracting tools."""
        decay_inplace(self.satisfied, self.satisfied_decay)
        add_scaled(self.satisfied, bag(text), weight)

    def observe_selection(self, doc_id: str, tool_key: str, category: str = "") -> None:
        if doc_id not in self.selected_doc_ids:
            self.selected_doc_ids.append(doc_id)
        self.tool_usage[tool_key] = self.tool_usage.get(tool_key, 0) + 1
        if category:
            self.categories[category] = self.categories.get(category, 0.0) + 1.0
        family = tool_key.split("::", 1)[0]
        if family:
            self.families[family] = self.families.get(family, 0.0) + 1.0

    def observe_failure(self, tool_key: str) -> None:
        self.tool_failures[tool_key] = self.tool_failures.get(tool_key, 0) + 1
        self.risk = min(1.0, self.risk + 0.18)

    def observe_success(self) -> None:
        self.risk = max(0.0, self.risk - 0.08)

    def advance_turn(self) -> None:
        self.turn += 1
        for f in list(self.families):
            self.families[f] *= self.family_decay

    # -- reads -------------------------------------------------------------
    def intent_affinity(self, text: str) -> float:
        """How much the session's live intent memory points at ``text``."""
        return cosine(self.memory, bag(text))

    def redundancy(self, text: str) -> float:
        """How much of ``text`` is already covered by served intents."""
        return cosine(self.satisfied, bag(text))

    def failure_penalty(self, tool_key: str) -> float:
        n = self.tool_failures.get(tool_key, 0)
        return 1.0 - 1.0 / (1.0 + n) if n else 0.0

    def saturation(self, tool_key: str) -> float:
        """Diminishing returns on calling the same endpoint repeatedly."""
        n = self.tool_usage.get(tool_key, 0)
        return 0.0 if n == 0 else min(1.0, 0.35 * n)

    def family_affinity(self, family: str) -> float:
        """Share of the session's (decayed) committed tool families that is ``family``, in [0, 1]."""
        if not self.families or not family:
            return 0.0
        top = max(self.families.values())
        return self.families.get(family, 0.0) / top if top else 0.0

    def top_families(self, n: int = 3) -> list[str]:
        return [f for f, _ in sorted(self.families.items(), key=lambda kv: -kv[1])[:n]]

    def category_prior(self, category: str) -> float:
        if not self.categories:
            return 0.0
        total = sum(self.categories.values())
        return self.categories.get(category, 0.0) / total if total else 0.0

    def snapshot(self) -> dict[str, Any]:
        top = sorted(self.memory.items(), key=lambda kv: -kv[1])[:8]
        top_sat = sorted(self.satisfied.items(), key=lambda kv: -kv[1])[:8]
        return {
            "turn": self.turn,
            "risk": round(self.risk, 4),
            "memoryTerms": [[k, round(v, 4)] for k, v in top],
            "satisfiedTerms": [[k, round(v, 4)] for k, v in top_sat],
            "toolsUsed": dict(self.tool_usage),
            "toolFailures": dict(self.tool_failures),
            "tokensSpent": self.tokens_spent,
            "costSpent": round(self.cost_spent, 4),
        }


__all__ = ["SessionState"]
