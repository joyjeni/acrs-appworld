"""Download-free dense stage for O1: TF-IDF + truncated SVD (latent semantic analysis)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from paim.common.types import Tool
from paim.o1_sasr.retriever import Candidate


@dataclass
class LSARetriever:
    """Latent-semantic (TF-IDF + truncated SVD) dense stage that needs no downloads.

    Used as the embedding half of the hybrid first stage when pretrained
    sentence encoders are unavailable (offline / restricted networks). With
    ``sentence-transformers`` installed, ``DenseRetriever`` is the stronger choice.
    """

    dims: int = 256
    seed: int = 0
    _vec: object | None = None
    _svd: object | None = None
    _matrix: object | None = None
    _doc_ids: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"lsa[{self.dims}]"

    @property
    def available(self) -> bool:
        return self._matrix is not None

    def fit(self, tools: Iterable[Tool]) -> "LSARetriever":
        try:
            import numpy as np
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            tools = list(tools)
            self._doc_ids = [t.doc_id for t in tools]
            self._vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000)
            x = self._vec.fit_transform([t.document_text() for t in tools])
            self._svd = TruncatedSVD(self.dims, random_state=self.seed)
            m = self._svd.fit_transform(x)
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
            self._matrix = m.astype("float32")
        except Exception:
            self._matrix = None
        return self

    def search(self, query: str, k: int = 50) -> list[Candidate]:
        if not self.available:
            return []
        import numpy as np

        q = self._svd.transform(self._vec.transform([query]))[0]  # type: ignore[union-attr]
        q /= np.linalg.norm(q) + 1e-9
        sims = self._matrix @ q  # type: ignore[operator]
        top = np.argsort(-sims)[:k]
        return [Candidate(self._doc_ids[int(i)], float(sims[int(i)]), r) for r, i in enumerate(top)]


__all__ = ["LSARetriever"]
