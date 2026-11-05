"""Tool co-occurrence graph learned from ToolBench solution trajectories.

Motivation
----------
ToolBench queries are compound: a gold answer usually calls 2-10 endpoints that
belong together ("get the airport, then get the flights from it"). A stateless
retriever cannot use that structure - it scores each API independently against
the instruction. The co-occurrence graph makes the structure explicit: after the
session commits to endpoint *u*, endpoints that historically co-occur with *u*
get a bonus, and unrelated endpoints that merely share vocabulary do not.

Edges are weighted by positive pointwise mutual information (PPMI) so that
globally popular endpoints do not dominate every neighbourhood.

The graph is fit on the *training* qrels only. ``tests/test_o1_sasr.py``
asserts no test-query gold set leaks into it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class CooccurrenceGraph:
    """PPMI-weighted undirected graph over tool document ids.

    Parameters
    ----------
    smoothing:
        Context-distribution smoothing exponent (Levy & Goldberg, 2015). Values
        below 1 flatten the marginal, which reduces PPMI's well-known bias
        towards rare items.
    """

    smoothing: float = 0.75
    min_count: int = 1
    _pair: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    _count: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _total_pairs: float = 0.0
    _total_items: float = 0.0
    _edges: dict[str, dict[str, float]] = field(default_factory=dict)
    n_trajectories: int = 0

    def fit(self, trajectories: Iterable[Sequence[str]]) -> "CooccurrenceGraph":
        """``trajectories`` is an iterable of co-called doc-id groups."""
        self._pair = defaultdict(float)
        self._count = defaultdict(float)
        self._total_pairs = 0.0
        self._total_items = 0.0
        self.n_trajectories = 0

        for traj in trajectories:
            uniq = sorted(set(traj))
            if len(uniq) < 2:
                # Still counts towards the marginal so a solo-called endpoint is
                # correctly modelled as having weak affinity to everything.
                for d in uniq:
                    self._count[d] += 1.0
                    self._total_items += 1.0
                continue
            self.n_trajectories += 1
            for d in uniq:
                self._count[d] += 1.0
                self._total_items += 1.0
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    self._pair[(uniq[i], uniq[j])] += 1.0
                    self._total_pairs += 1.0

        self._build_edges()
        return self

    def _build_edges(self) -> None:
        self._edges = defaultdict(dict)
        if self._total_pairs <= 0 or self._total_items <= 0:
            return
        # Smoothed marginal distribution.
        marg = {d: (c ** self.smoothing) for d, c in self._count.items()}
        marg_total = sum(marg.values()) or 1.0

        for (u, v), c in self._pair.items():
            if c < self.min_count:
                continue
            p_uv = c / self._total_pairs
            p_u = marg[u] / marg_total
            p_v = marg[v] / marg_total
            denom = p_u * p_v
            if denom <= 0:
                continue
            ppmi = max(0.0, math.log(p_uv / denom))
            if ppmi <= 0:
                continue
            self._edges[u][v] = ppmi
            self._edges[v][u] = ppmi

        # Row-normalise so ``affinity`` returns a bounded, comparable score.
        for u, nbrs in self._edges.items():
            peak = max(nbrs.values()) or 1.0
            for v in nbrs:
                nbrs[v] = nbrs[v] / peak

    # -- reads -------------------------------------------------------------
    def weight(self, u: str, v: str) -> float:
        return self._edges.get(u, {}).get(v, 0.0)

    def affinity(self, doc_id: str, committed: Sequence[str]) -> float:
        """Max-pooled affinity of ``doc_id`` to already committed endpoints.

        Max rather than mean: a candidate that strongly partners with *any*
        committed endpoint is a good continuation, even if it is unrelated to
        the rest of the session.
        """
        if not committed:
            return 0.0
        return max((self.weight(doc_id, c) for c in committed), default=0.0)

    def neighbours(self, doc_id: str, k: int = 8) -> list[tuple[str, float]]:
        nbrs = self._edges.get(doc_id, {})
        return sorted(nbrs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    @property
    def n_nodes(self) -> int:
        return len(self._edges)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self._edges.values()) // 2

    def contains_any(self, doc_ids: Iterable[str]) -> bool:
        """Leakage check used by the test suite."""
        return any(d in self._count for d in doc_ids)

    def stats(self) -> dict[str, float]:
        return {
            "nodes": float(self.n_nodes),
            "edges": float(self.n_edges),
            "trajectories": float(self.n_trajectories),
            "avgDegree": round(2 * self.n_edges / self.n_nodes, 4) if self.n_nodes else 0.0,
        }


__all__ = ["CooccurrenceGraph"]
