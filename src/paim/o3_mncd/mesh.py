"""Peer-to-peer mesh topology and gossip belief propagation.

Positioning
-----------
Multi-agent LLM systems overwhelmingly use a **star**: a hub (planner,
orchestrator, judge) talks to every worker and workers never talk to each
other. Multi-agent debate and self-consistency use a **complete** graph, where
every agent sees every other agent's answer each round. The first has a single
point of failure and O(n) rounds of hub reasoning; the second costs O(n^2)
messages per round and is what makes debate expensive.

This module implements a **degree-bounded connected mesh** (a ring plus
deterministic chords) with gossip belief propagation. Messages per round are
O(n*deg) with deg fixed, and the graph's spectral gap bounds how many rounds are
needed for beliefs to mix, so consensus quality can be traded against message
budget explicitly instead of by prompt engineering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class MeshTopology:
    """Connected, degree-bounded, deterministic peer graph.

    Construction: a Hamiltonian ring over the node order guarantees
    connectivity, then chords at stride ``2^i`` add logarithmic-diameter
    shortcuts. Deterministic, so a run is replayable from the node list alone.
    """

    node_ids: tuple[str, ...]
    degree: int = 4
    adjacency: dict[str, tuple[str, ...]] = field(default_factory=dict)
    topology: str = "mesh"

    def __post_init__(self) -> None:
        if not self.adjacency:
            self.adjacency = self._build()

    def _build(self) -> dict[str, tuple[str, ...]]:
        n = len(self.node_ids)
        adj: dict[str, set[str]] = {nid: set() for nid in self.node_ids}
        if n < 2:
            return {nid: () for nid in self.node_ids}

        for i, nid in enumerate(self.node_ids):
            ring_next = self.node_ids[(i + 1) % n]
            adj[nid].add(ring_next)
            adj[ring_next].add(nid)

        stride = 2
        while stride < n and max((len(v) for v in adj.values()), default=0) < self.degree:
            for i, nid in enumerate(self.node_ids):
                j = (i + stride) % n
                other = self.node_ids[j]
                if other == nid:
                    continue
                if len(adj[nid]) >= self.degree or len(adj[other]) >= self.degree:
                    continue
                adj[nid].add(other)
                adj[other].add(nid)
            stride *= 2

        return {nid: tuple(sorted(adj[nid])) for nid in self.node_ids}

    # -- structure ---------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.adjacency.values()) // 2

    @property
    def avg_degree(self) -> float:
        return 2 * self.n_edges / self.n_nodes if self.n_nodes else 0.0

    def is_connected(self) -> bool:
        if not self.node_ids:
            return True
        seen = {self.node_ids[0]}
        stack = [self.node_ids[0]]
        while stack:
            cur = stack.pop()
            for nb in self.adjacency.get(cur, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == self.n_nodes

    def diameter(self) -> int:
        """Exact diameter by BFS from every node (n is small: <= tens)."""
        best = 0
        for src in self.node_ids:
            dist = {src: 0}
            frontier = [src]
            while frontier:
                nxt = []
                for cur in frontier:
                    for nb in self.adjacency.get(cur, ()):
                        if nb not in dist:
                            dist[nb] = dist[cur] + 1
                            nxt.append(nb)
                frontier = nxt
            if len(dist) < self.n_nodes:
                return -1  # disconnected
            best = max(best, max(dist.values()))
        return best

    def messages_per_round(self) -> int:
        return 2 * self.n_edges

    def rounds_needed(self) -> int:
        """Heuristic mixing time: diameter is a lower bound; use diameter + 1."""
        d = self.diameter()
        return max(1, (d if d > 0 else 1) + 1)

    # -- gossip ------------------------------------------------------------
    def gossip(
        self,
        beliefs: dict[str, dict[str, float]],
        rounds: int | None = None,
        mixing: float = 0.5,
    ) -> tuple[dict[str, dict[str, float]], list[dict[str, float]]]:
        """Run synchronous gossip averaging on per-node belief distributions.

        Each round every node replaces its belief with a convex combination of
        its own and the mean of its neighbours'. Returns the final beliefs plus
        a per-round convergence log (max total-variation distance between any
        node pair), which is what the experiment plots.
        """
        rounds = self.rounds_needed() if rounds is None else rounds
        state = {nid: dict(beliefs.get(nid, {})) for nid in self.node_ids}
        log: list[dict[str, float]] = [
            {"round": 0.0, "maxDisagreement": round(_max_tv(state), 6)}
        ]

        for r in range(1, rounds + 1):
            new_state: dict[str, dict[str, float]] = {}
            for nid in self.node_ids:
                nbrs = self.adjacency.get(nid, ())
                own = state[nid]
                if not nbrs:
                    new_state[nid] = dict(own)
                    continue
                acc: dict[str, float] = {}
                for nb in nbrs:
                    for k, v in state[nb].items():
                        acc[k] = acc.get(k, 0.0) + v / len(nbrs)
                merged: dict[str, float] = {}
                for k in set(own) | set(acc):
                    merged[k] = (1 - mixing) * own.get(k, 0.0) + mixing * acc.get(k, 0.0)
                new_state[nid] = _normalise(merged)
            state = new_state
            log.append({"round": float(r), "maxDisagreement": round(_max_tv(state), 6)})

        return state, log

    def stats(self) -> dict[str, float]:
        return {
            "nodes": float(self.n_nodes),
            "edges": float(self.n_edges),
            "avgDegree": round(self.avg_degree, 3),
            "diameter": float(self.diameter()),
            "messagesPerRound": float(self.messages_per_round()),
            "rounds": float(self.rounds_needed()),
            "connected": 1.0 if self.is_connected() else 0.0,
        }


@dataclass
class StarTopology(MeshTopology):
    """Baseline: hub-and-spoke, the standard orchestrator pattern."""

    topology: str = "star"

    def _build(self) -> dict[str, tuple[str, ...]]:
        if not self.node_ids:
            return {}
        hub, spokes = self.node_ids[0], self.node_ids[1:]
        adj = {hub: tuple(sorted(spokes))}
        for s in spokes:
            adj[s] = (hub,)
        return adj


@dataclass
class CompleteTopology(MeshTopology):
    """Baseline: multi-agent-debate style all-to-all graph."""

    topology: str = "complete"

    def _build(self) -> dict[str, tuple[str, ...]]:
        return {
            nid: tuple(sorted(o for o in self.node_ids if o != nid)) for nid in self.node_ids
        }


def _normalise(d: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in d.values())
    if total <= 0:
        n = len(d) or 1
        return {k: 1.0 / n for k in d}
    return {k: max(0.0, v) / total for k, v in d.items()}


def _max_tv(state: dict[str, dict[str, float]]) -> float:
    """Max total-variation distance between any two nodes' beliefs."""
    nodes = list(state)
    worst = 0.0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = state[nodes[i]], state[nodes[j]]
            keys = set(a) | set(b)
            tv = 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
            worst = max(worst, tv)
    return worst


def byzantine_quorum(n: int, f: int) -> int:
    """Minimum reports needed to certify a claim with ``f`` faulty peers."""
    return min(n, 2 * f + 1)


def max_tolerated_faults(n: int) -> int:
    """Largest f with n >= 3f + 1 (classic BFT bound)."""
    return max(0, (n - 1) // 3)


def spectral_gap_estimate(mesh: MeshTopology) -> float:
    """Cheap proxy for the mixing rate: 1 / (diameter * log(n+1))."""
    d = mesh.diameter()
    if d <= 0:
        return 0.0
    return 1.0 / (d * math.log(mesh.n_nodes + 1))


__all__ = [
    "CompleteTopology",
    "MeshTopology",
    "StarTopology",
    "byzantine_quorum",
    "max_tolerated_faults",
    "spectral_gap_estimate",
]
