"""Two retrievers with the same interface: TF-IDF (baseline) and SBERT (semantic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import Paper
from .preprocess import clean_text, split_chunks

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Hit:
    paper: Paper
    score: float
    rank: int


class BaseRetriever:
    name = "base"

    def __init__(self, papers: list[Paper]):
        self.papers = papers

    def scores(self, query: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[Hit]:
        s = self.scores(query)
        order = np.argsort(-s)[:k]
        return [Hit(self.papers[i], float(s[i]), r + 1) for r, i in enumerate(order)]


class TfidfRetriever(BaseRetriever):
    """Keyword baseline: TF-IDF vectors + cosine similarity."""

    name = "TF-IDF"

    def __init__(self, papers: list[Paper]):
        super().__init__(papers)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(clean_text(p.text) for p in papers)

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([clean_text(query)])
        # TF-IDF rows are L2-normalised, so dot product == cosine similarity.
        return (self.matrix @ q.T).toarray().ravel()


class SbertRetriever(BaseRetriever):
    """Semantic retriever: Sentence-BERT embeddings + cosine similarity."""

    name = "SBERT"

    def __init__(self, papers: list[Paper], model_name: str = DEFAULT_MODEL,
                 cache_dir: str | Path | None = None, model=None):
        super().__init__(papers)
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        self.model = model
        self.embeddings = self._load_or_encode(model_name, cache_dir)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                              batch_size=32)
        )

    def _load_or_encode(self, model_name: str, cache_dir) -> np.ndarray:
        texts = [p.text for p in self.papers]
        if cache_dir is None:
            return self._encode(texts)
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"emb_{model_name.replace('/', '_')}_{len(texts)}.npy"
        ids = cache / f"{f.stem}.ids"
        want = "\n".join(p.id for p in self.papers)
        if f.exists() and ids.exists() and ids.read_text() == want:
            return np.load(f)
        emb = self._encode(texts)
        np.save(f, emb)
        ids.write_text(want)
        return emb

    LONG_QUERY_WORDS = 150

    def scores(self, query: str) -> np.ndarray:
        # Embeddings are normalised, so dot product == cosine similarity.
        if len(query.split()) <= self.LONG_QUERY_WORDS:
            return self.embeddings @ self._encode([query])[0]
        # Long pasted idea: embed it in chunks. A paper that closely matches any one
        # part of the idea scores well (max), but matching several parts is rewarded (mean).
        sims = self.embeddings @ self._encode(split_chunks(query)).T  # (papers, chunks)
        return 0.5 * sims.mean(axis=1) + 0.5 * sims.max(axis=1)
