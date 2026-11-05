"""Deterministic text processing.

Everything here is pure Python so that the reference implementation produces
byte-identical results on a laptop, in a Kaggle notebook with the internet
disabled, and in the TypeScript port that serves the web demo. The parity test
suite (``tests/test_parity.py`` plus ``lib/paim/__tests__``) depends on that.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    """
a about above after again against all am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do does
doesn doing don down during each few for from further had hadn has hasn have haven
having he her here hers herself him himself his how i if in into is isn it its
itself just ll me more most mustn my myself no nor not now of off on once only or
other ought our ours ourselves out over own re s same shan she should shouldn so
some such t than that the their theirs them themselves then there these they this
those through to too under until up ve very was wasn we were weren what when where
which while who whom why will with won would wouldn you your yours yourself
yourselves also additionally furthermore moreover please want need would like get
provide me my give show tell find help thanks thank you
""".split()
)

# Discourse markers that begin a new intent inside a compound ToolBench request.
SUB_GOAL_MARKERS: tuple[str, ...] = (
    "additionally",
    "also",
    "furthermore",
    "moreover",
    "in addition",
    "besides that",
    "besides",
    "on top of that",
    "apart from that",
    "lastly",
    "finally",
    "next",
    "then",
    "as well as that",
    "second",
    "secondly",
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=200_000)
def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase alphanumeric tokenisation with stopword removal."""
    return tuple(t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1)


def tokenize_keep_stopwords(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.lower()))


def term_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def bag(text: str) -> dict[str, float]:
    """L2-normalised term-frequency bag of words."""
    counts = term_counts(text)
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity of two (already normalised or not) sparse bags."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def add_scaled(target: dict[str, float], source: dict[str, float], scale: float) -> None:
    for k, v in source.items():
        target[k] = target.get(k, 0.0) + v * scale


def decay_inplace(target: dict[str, float], factor: float, floor: float = 1e-6) -> None:
    """Multiply every weight by ``factor`` and drop numerically dead terms."""
    dead = []
    for k in target:
        target[k] *= factor
        if target[k] < floor:
            dead.append(k)
    for k in dead:
        del target[k]


def split_sub_goals(query: str) -> list[str]:
    """Segment a compound request into ordered atomic intents.

    ToolBench G1/G2/G3 instructions are deliberately compound ("... .
    Additionally, I need ..."). Splitting them is what turns a single-shot
    retrieval benchmark into a *session* benchmark, which is the setting O1
    targets. The rule set is intentionally simple and inspectable: sentence
    boundaries first, then discourse markers inside long sentences.
    """
    query = (query or "").strip()
    if not query:
        return []

    spans: list[str] = []
    for sentence in _SENTENCE_RE.split(query):
        sentence = sentence.strip()
        if not sentence:
            continue
        spans.extend(_split_on_markers(sentence))

    spans = [s.strip(" ,;.") for s in spans]
    spans = [s for s in spans if len(tokenize(s)) >= 2]
    if not spans:
        return [query]

    # A trailing fragment with no content verb is a continuation, not an intent.
    merged: list[str] = []
    for span in spans:
        if merged and len(tokenize(span)) < 3:
            merged[-1] = f"{merged[-1]} {span}"
        else:
            merged.append(span)
    return merged


def _split_on_markers(sentence: str) -> list[str]:
    lowered = sentence.lower()
    cuts = [0]
    for marker in SUB_GOAL_MARKERS:
        start = 0
        while True:
            idx = lowered.find(marker, start)
            if idx == -1:
                break
            start = idx + len(marker)
            # Only treat it as a cut when it opens a clause.
            if idx == 0:
                continue
            prev = lowered[idx - 1]
            if prev in ",;. " and idx > 12:
                cuts.append(idx)
    cuts = sorted(set(cuts))
    cuts.append(len(sentence))
    out = []
    for i in range(len(cuts) - 1):
        piece = sentence[cuts[i] : cuts[i + 1]].strip()
        if piece:
            out.append(piece)
    return out or [sentence]


def approx_tokens(text: str) -> int:
    """Deterministic stand-in for a BPE token count."""
    words = max(1, len((text or "").split()))
    return max(words, int(math.ceil(len(text or "") / 4)))


__all__ = [
    "STOPWORDS",
    "SUB_GOAL_MARKERS",
    "add_scaled",
    "approx_tokens",
    "bag",
    "cosine",
    "decay_inplace",
    "jaccard",
    "split_sub_goals",
    "term_counts",
    "tokenize",
    "tokenize_keep_stopwords",
]
