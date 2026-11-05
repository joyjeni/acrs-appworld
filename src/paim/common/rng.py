"""Reproducible randomness.

The pipeline is stochastic in three places (agent execution outcomes, bandit
exploration, gossip peer sampling). All three draw from streams derived from a
single seed so that a whole experiment is replayable from its config, and so
the TypeScript port can reproduce the same trace.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


def derive_seed(seed: int, *labels: str) -> int:
    """Deterministically derive a sub-stream seed from a base seed and labels."""
    h = hashlib.sha256(f"{seed}|{'|'.join(labels)}".encode()).digest()
    return int.from_bytes(h[:8], "big")


@dataclass
class SeededStreams:
    """Named independent random streams derived from one base seed."""

    seed: int = 20260905

    def stream(self, *labels: str) -> random.Random:
        return random.Random(derive_seed(self.seed, *labels))

    def bernoulli(self, p: float, *labels: str) -> bool:
        return self.stream(*labels).random() < p

    def uniform(self, lo: float, hi: float, *labels: str) -> float:
        return self.stream(*labels).uniform(lo, hi)


__all__ = ["SeededStreams", "derive_seed"]
