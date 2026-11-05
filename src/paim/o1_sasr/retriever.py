"""First-stage tool retrieval over the ToolBench corpus.

SOTA reference
--------------
ToolBench's own retriever (``ToolRetriever``, Qin et al., 2023) is a fine-tuned
Sentence-BERT bi-encoder that embeds the *whole user instruction* once and does
a single nearest-neighbour lookup over the API corpus. It is stateless: turn 5
of a session is scored exactly like turn 1.

What lives here
---------------
Only the *stateless first stage*, deliberately kept simple and reproducible:

``BM25Retriever``    Okapi BM25 with an inverted index. No model download, no
                     GPU, identical results everywhere. This is the default so
                     the pipeline runs in an offline Kaggle kernel.
``DenseRetriever``   Optional Sentence-BERT bi-encoder that reproduces the
                     ToolRetriever setup when ``sentence-transformers`` and a
                     local model are available.
``HybridRetriever``  Reciprocal-rank fusion of the two.

The session-aware part - the actual novelty of O1 - is in ``reranker.py``.
"""

from __future__ import annotations

import os

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

from paim.common.text import tokenize
from paim.common.types import Tool


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    score: float
    rank: int


class Retriever(Protocol):
    def search(self, query: str, k: int = 50) -> list[Candidate]: ...

    @property
    def name(self) -> str: ...


@dataclass
class BM25Retriever:
    """Okapi BM25 over rendered tool documents."""

    k1: float = 1.2
    b: float = 0.75
    _index: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    _doc_ids: list[str] = field(default_factory=list)
    _doc_len: list[int] = field(default_factory=list)
    _avg_len: float = 0.0
    _idf: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "bm25"

    def fit(self, tools: Iterable[Tool]) -> "BM25Retriever":
        self._index = defaultdict(list)
        self._doc_ids = []
        self._doc_len = []
        df: dict[str, int] = defaultdict(int)

        for tool in tools:
            idx = len(self._doc_ids)
            self._doc_ids.append(tool.doc_id)
            terms = tokenize(tool.document_text())
            self._doc_len.append(len(terms))
            tf: dict[str, int] = defaultdict(int)
            for t in terms:
                tf[t] += 1
            for term, count in tf.items():
                self._index[term].append((idx, count))
                df[term] += 1

        n = max(1, len(self._doc_ids))
        self._avg_len = sum(self._doc_len) / n
        # BM25 idf with the +0.5 smoothing, floored so common terms never go
        # negative (which would reward documents for *lacking* a query term).
        self._idf = {
            term: max(1e-6, math.log((n - d + 0.5) / (d + 0.5) + 1.0)) for term, d in df.items()
        }
        return self

    def search(self, query: str, k: int = 50) -> list[Candidate]:
        terms = tokenize(query)
        if not terms or not self._doc_ids:
            return []
        qtf: dict[str, int] = defaultdict(int)
        for t in terms:
            qtf[t] += 1

        scores: dict[int, float] = defaultdict(float)
        for term, qcount in qtf.items():
            postings = self._index.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_idx, tf in postings:
                dl = self._doc_len[doc_idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1.0))
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom * math.log(1 + qcount)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], self._doc_ids[kv[0]]))[:k]
        return [Candidate(self._doc_ids[i], round(s, 8), r) for r, (i, s) in enumerate(ordered)]

    @property
    def size(self) -> int:
        return len(self._doc_ids)


_MODEL_CACHE: dict = {}
_EMB_CACHE: dict = {}
_QUERY_CACHE: dict = {}


@dataclass
class DenseRetriever:
    """Sentence-BERT bi-encoder stage (the ToolRetriever configuration).

    Optional by design: ``available`` is ``False`` when
    ``sentence-transformers`` or the model weights are missing, and the pipeline
    silently uses BM25 instead. That keeps the end-to-end demo runnable on a CPU
    box with no network.
    """

    # Override on GPU runs, e.g. ACRS_DENSE_MODEL=BAAI/bge-base-en-v1.5
    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "ACRS_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    _model: object | None = None
    _doc_ids: list[str] = field(default_factory=list)
    _matrix: object | None = None

    @property
    def name(self) -> str:
        return f"dense[{self.model_name.split('/')[-1]}]"

    @property
    def available(self) -> bool:
        return self._model is not None

    def fit(self, tools: Iterable[Tool]) -> "DenseRetriever":
        tools = list(tools)
        try:  # pragma: no cover - exercised only when the extra is installed
            import numpy as np
            from sentence_transformers import SentenceTransformer

            # Process-level cache: experiments build one pipeline per seed and
            # configuration; re-loading the encoder and re-embedding a 10k-tool
            # corpus each time dominated GPU runtime. The cache key is the model
            # and the exact rendered corpus, so a changed corpus re-embeds.
            if self.model_name not in _MODEL_CACHE:
                _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
            self._model = _MODEL_CACHE[self.model_name]
            self._doc_ids = [t.doc_id for t in tools]
            texts = [t.document_text() for t in tools]
            key = (self.model_name, hash(tuple(zip(self._doc_ids, texts))))
            if key not in _EMB_CACHE:
                emb = self._model.encode(
                    texts,
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                _EMB_CACHE[key] = np.asarray(emb, dtype="float32")
            self._matrix = _EMB_CACHE[key]
        except Exception:
            self._model = None
            self._matrix = None
        return self

    def search(self, query: str, k: int = 50) -> list[Candidate]:
        if not self.available or self._matrix is None:  # pragma: no cover
            return []
        import numpy as np  # pragma: no cover

        qkey = (self.model_name, query)
        if qkey not in _QUERY_CACHE:
            _QUERY_CACHE[qkey] = self._model.encode(  # type: ignore[union-attr]
                [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )[0]
        sims = np.asarray(self._matrix) @ np.asarray(_QUERY_CACHE[qkey])
        top = np.argsort(-sims)[:k]
        return [Candidate(self._doc_ids[int(i)], float(sims[int(i)]), r) for r, i in enumerate(top)]


@dataclass
class HybridRetriever:
    """Reciprocal-rank fusion of a lexical and a dense stage."""

    lexical: BM25Retriever
    dense: DenseRetriever | None = None
    rrf_k: int = 60

    @property
    def name(self) -> str:
        return "hybrid" if self.dense and self.dense.available else "bm25"

    def search(self, query: str, k: int = 50) -> list[Candidate]:
        runs: list[list[Candidate]] = [self.lexical.search(query, k * 2)]
        if self.dense is not None and self.dense.available:
            runs.append(self.dense.search(query, k * 2))
        if len(runs) == 1:
            return runs[0][:k]

        fused: dict[str, float] = defaultdict(float)
        for run in runs:
            for c in run:
                fused[c.doc_id] += 1.0 / (self.rrf_k + c.rank + 1)
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Candidate(d, round(s, 8), r) for r, (d, s) in enumerate(ordered)]


def rank_ids(candidates: Sequence[Candidate]) -> list[str]:
    return [c.doc_id for c in candidates]


__all__ = [
    "BM25Retriever",
    "Candidate",
    "DenseRetriever",
    "HybridRetriever",
    "Retriever",
    "rank_ids",
]
