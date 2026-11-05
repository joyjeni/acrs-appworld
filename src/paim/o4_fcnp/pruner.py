"""Structured context pruning under a token budget.

Positioning
-----------
================================  ======================================================
SOTA                              Limitation this objective attacks
================================  ======================================================
LLMLingua / LongLLMLingua         Token-level perplexity compression of free prose at a
                                  *fixed* target ratio; unaware of which spans are
                                  syntactically load-bearing for a tool call.
Selective-Context                 Same family; self-information over a token stream.
H2O / SnapKV / StreamingLLM       KV-cache eviction inside one forward pass; cannot be
                                  applied to context the orchestrator assembles.
RAG top-k truncation              Keeps whole documents, so a single verbose tool schema
                                  can evict the entire session history.
================================  ======================================================

Every one of these compresses to a ratio the engineer picks. None of them can
answer "how much can I compress *this* turn before the answer degrades?".

Proposed
--------
Prune over **typed context blocks** (tool signature, required parameter,
observation, session memory, ...) with a value-density knapsack, subject to a
hard-keep set derived from schema integrity:

.. math::

    v(b) = \\beta_r\\,\\mathrm{rel}(b, g) + \\beta_c\\,\\mathrm{crit}(b)
           + \\beta_t\\,\\mathrm{recency}(b) - \\beta_d\\,\\mathrm{red}(b, K)

blocks are then admitted in decreasing :math:`v(b)/\\mathrm{tokens}(b)` order
until the budget is exhausted, with any block in the hard-keep set admitted
first regardless of density. The *budget itself* is not fixed - it is set each
turn by the controller in ``controller.py`` from the measured fidelity of the
previous turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from paim.common.text import bag, cosine
from paim.common.types import ContextBlock, Tool
from paim.o4_fcnp.fidelity import FidelityEstimator, FidelityReport

HARD_KEEP_KINDS = frozenset({"user_query", "sub_goal"})
HARD_KEEP_CRITICALITY = 0.88


@dataclass(frozen=True)
class PruneWeights:
    relevance: float = 0.45
    criticality: float = 0.35
    recency: float = 0.12
    redundancy: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return {
            "relevance": self.relevance,
            "criticality": self.criticality,
            "recency": self.recency,
            "redundancy": self.redundancy,
        }


@dataclass
class BlockScore:
    block_id: str
    kind: str
    tokens: int
    value: float
    density: float
    hard_keep: bool
    kept: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockId": self.block_id,
            "kind": self.kind,
            "tokens": self.tokens,
            "value": round(self.value, 5),
            "density": round(self.density, 6),
            "hardKeep": self.hard_keep,
            "kept": self.kept,
        }


@dataclass
class PruneResult:
    kept: list[ContextBlock]
    dropped: list[ContextBlock]
    budget: int
    kept_tokens: int
    original_tokens: int
    fidelity: FidelityReport
    scores: list[BlockScore] = field(default_factory=list)

    @property
    def compression(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return 1.0 - self.kept_tokens / self.original_tokens

    @property
    def keep_fraction(self) -> float:
        if self.original_tokens <= 0:
            return 1.0
        return self.kept_tokens / self.original_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "keptTokens": self.kept_tokens,
            "originalTokens": self.original_tokens,
            "compression": round(self.compression, 5),
            "keepFraction": round(self.keep_fraction, 5),
            "keptBlocks": len(self.kept),
            "droppedBlocks": len(self.dropped),
            "fidelity": self.fidelity.to_dict(),
            "droppedKinds": _kind_counts(self.dropped),
            "keptKinds": _kind_counts(self.kept),
        }


class FCNPPruner:
    """Value-density knapsack pruner with a schema-integrity hard-keep set."""

    objective = "O4"
    name = "FCNP"

    def __init__(
        self,
        weights: PruneWeights | None = None,
        estimator: FidelityEstimator | None = None,
    ) -> None:
        self.weights = weights or PruneWeights()
        self.estimator = estimator or FidelityEstimator()

    def hard_keep_ids(
        self, blocks: Sequence[ContextBlock], committed_tools: Iterable[Tool]
    ) -> set[str]:
        """Blocks whose removal would break a committed tool call or the goal."""
        ids = {
            b.block_id
            for b in blocks
            if b.kind in HARD_KEEP_KINDS or b.criticality >= HARD_KEEP_CRITICALITY
        }
        for tool in committed_tools:
            ids.add(f"{tool.doc_id}:head")
            for i, p in enumerate(tool.parameters):
                if p.required:
                    ids.add(f"{tool.doc_id}:param{i}")
        present = {b.block_id for b in blocks}
        return ids & present

    def prune(
        self,
        blocks: Sequence[ContextBlock],
        budget: int,
        goal_text: str,
        committed_tools: Iterable[Tool] = (),
        current_turn: int | None = None,
    ) -> PruneResult:
        blocks = list(blocks)
        original_tokens = sum(b.tokens for b in blocks)
        committed_tools = list(committed_tools)
        if not blocks:
            return PruneResult(
                [], [], budget, 0, 0,
                self.estimator.estimate([], [], goal_text, committed_tools),
            )

        hard = self.hard_keep_ids(blocks, committed_tools)
        turn = current_turn if current_turn is not None else max(b.turn for b in blocks)
        goal_bag = bag(goal_text)
        w = self.weights

        kept: list[ContextBlock] = []
        kept_bags: list[dict[str, float]] = []
        kept_tokens = 0
        scores: list[BlockScore] = []

        # Hard-keep set is admitted first and may legitimately overrun the
        # budget: a callable-but-oversized context is preferable to a
        # within-budget context that cannot be executed. The controller sees the
        # overrun through ``keep_fraction`` and raises the budget next turn.
        for b in blocks:
            if b.block_id in hard:
                kept.append(b)
                kept_bags.append(bag(b.text))
                kept_tokens += b.tokens
                scores.append(
                    BlockScore(b.block_id, b.kind, b.tokens, float("inf"), float("inf"), True, True)
                )

        remaining = [b for b in blocks if b.block_id not in hard]

        # Value density, computed greedily so redundancy is measured against the
        # blocks already admitted.
        pending = list(remaining)
        while pending:
            best_idx = -1
            best_density = float("-inf")
            best_value = 0.0
            for i, b in enumerate(pending):
                b_bag = bag(b.text)
                rel = cosine(goal_bag, b_bag)
                recency = 1.0 / (1.0 + max(0, turn - b.turn))
                red = max((cosine(b_bag, k) for k in kept_bags), default=0.0)
                value = (
                    w.relevance * rel
                    + w.criticality * b.criticality
                    + w.recency * recency
                    - w.redundancy * red
                )
                density = value / max(1, b.tokens)
                if density > best_density:
                    best_idx, best_density, best_value = i, density, value
            b = pending.pop(best_idx)
            fits = kept_tokens + b.tokens <= budget
            if fits and best_value > 0:
                kept.append(b)
                kept_bags.append(bag(b.text))
                kept_tokens += b.tokens
            scores.append(
                BlockScore(
                    b.block_id, b.kind, b.tokens, best_value, best_density, False,
                    fits and best_value > 0,
                )
            )

        kept_ids = {b.block_id for b in kept}
        dropped = [b for b in blocks if b.block_id not in kept_ids]
        # Restore document order for the assembled prompt.
        kept.sort(key=lambda b: (b.turn, blocks.index(b)))

        fidelity = self.estimator.estimate(kept, blocks, goal_text, committed_tools)
        return PruneResult(
            kept=kept,
            dropped=dropped,
            budget=budget,
            kept_tokens=kept_tokens,
            original_tokens=original_tokens,
            fidelity=fidelity,
            scores=scores,
        )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class FixedRatioPruner(FCNPPruner):
    """LLMLingua-style: compress to a fixed ratio, ignore schema integrity."""

    name = "Fixed-ratio compression"

    def __init__(self, keep_ratio: float = 0.5) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio

    def prune(self, blocks, budget, goal_text, committed_tools=(), current_turn=None):  # type: ignore[override]
        blocks = list(blocks)
        original = sum(b.tokens for b in blocks)
        target = int(original * self.keep_ratio)
        goal_bag = bag(goal_text)
        ordered = sorted(blocks, key=lambda b: -cosine(goal_bag, bag(b.text)))
        kept, tokens = [], 0
        for b in ordered:
            if tokens + b.tokens > target:
                continue
            kept.append(b)
            tokens += b.tokens
        kept_ids = {b.block_id for b in kept}
        dropped = [b for b in blocks if b.block_id not in kept_ids]
        return PruneResult(
            kept, dropped, target, tokens, original,
            self.estimator.estimate(kept, blocks, goal_text, list(committed_tools)),
        )


class TailTruncationPruner(FCNPPruner):
    """Keep the most recent blocks until the budget is full (sliding window)."""

    name = "Tail truncation"

    def prune(self, blocks, budget, goal_text, committed_tools=(), current_turn=None):  # type: ignore[override]
        blocks = list(blocks)
        original = sum(b.tokens for b in blocks)
        kept, tokens = [], 0
        for b in sorted(blocks, key=lambda b: -b.turn):
            if tokens + b.tokens > budget:
                continue
            kept.append(b)
            tokens += b.tokens
        kept_ids = {b.block_id for b in kept}
        dropped = [b for b in blocks if b.block_id not in kept_ids]
        kept.sort(key=lambda b: b.turn)
        return PruneResult(
            kept, dropped, budget, tokens, original,
            self.estimator.estimate(kept, blocks, goal_text, list(committed_tools)),
        )


class NoPruner(FCNPPruner):
    """Upper bound on fidelity, and the token cost of not pruning at all."""

    name = "No pruning"

    def prune(self, blocks, budget, goal_text, committed_tools=(), current_turn=None):  # type: ignore[override]
        blocks = list(blocks)
        total = sum(b.tokens for b in blocks)
        return PruneResult(
            blocks, [], budget, total, total,
            self.estimator.estimate(blocks, blocks, goal_text, list(committed_tools)),
        )


def _kind_counts(blocks: Sequence[ContextBlock]) -> dict[str, int]:
    out: dict[str, int] = {}
    for b in blocks:
        out[b.kind] = out.get(b.kind, 0) + 1
    return out


__all__ = [
    "BlockScore",
    "FCNPPruner",
    "FixedRatioPruner",
    "HARD_KEEP_CRITICALITY",
    "HARD_KEEP_KINDS",
    "NoPruner",
    "PruneResult",
    "PruneWeights",
    "TailTruncationPruner",
]
