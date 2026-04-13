"""
Sentence-transformer embeddings + FAISS vector store.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import EMBEDDING_MODEL, FAISS_DIR

logger = logging.getLogger(__name__)

_FAISS_INDEX_FILE = FAISS_DIR / "index.faiss"
_METADATA_FILE = FAISS_DIR / "metadata.pkl"


class EmbeddingStore:
    """Manages paper embeddings and FAISS similarity search."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._papers: list[dict] = []

    # ── Lazy-load model ───────────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ── Index management ──────────────────────────────────────────────────

    def _make_index(self, dim: int):
        import faiss
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine after L2 norm)
        return faiss.IndexIDMap(index)

    def build_index(self, papers: list[dict], batch_size: int = 64) -> None:
        """Encode paper abstracts and build the FAISS index."""
        import faiss

        texts = [
            f"{p.get('title', '')}. {p.get('abstract', '')}"
            for p in papers
        ]
        logger.info("Encoding %d papers …", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        self._index = self._make_index(dim)
        ids = np.arange(len(papers), dtype=np.int64)
        self._index.add_with_ids(embeddings, ids)
        self._papers = papers

        self._save()
        logger.info("Index built with %d vectors (dim=%d).", len(papers), dim)

    def add_papers(self, new_papers: list[dict]) -> int:
        """Add new papers to an existing index. Returns number added."""
        import faiss

        if not new_papers:
            return 0

        existing_ids = {p["id"] for p in self._papers}
        to_add = [p for p in new_papers if p["id"] not in existing_ids]
        if not to_add:
            return 0

        texts = [
            f"{p.get('title', '')}. {p.get('abstract', '')}"
            for p in to_add
        ]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        if self._index is None:
            dim = embeddings.shape[1]
            self._index = self._make_index(dim)
            start_id = 0
        else:
            start_id = len(self._papers)

        ids = np.arange(start_id, start_id + len(to_add), dtype=np.int64)
        self._index.add_with_ids(embeddings, ids)
        self._papers.extend(to_add)

        self._save()
        return len(to_add)

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """Return top-k papers most similar to query."""
        if self._index is None or self._index.ntotal == 0:
            return []

        q_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < score_threshold:
                continue
            paper = dict(self._papers[int(idx)])
            paper["relevance_score"] = float(score)
            results.append(paper)

        return results

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        import faiss
        if self._index is not None:
            faiss.write_index(self._index, str(_FAISS_INDEX_FILE))
        with _METADATA_FILE.open("wb") as fh:
            pickle.dump(self._papers, fh)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        import faiss
        if not _FAISS_INDEX_FILE.exists() or not _METADATA_FILE.exists():
            return False
        try:
            self._index = faiss.read_index(str(_FAISS_INDEX_FILE))
            with _METADATA_FILE.open("rb") as fh:
                self._papers = pickle.load(fh)
            logger.info(
                "Loaded FAISS index (%d vectors) from disk.", self._index.ntotal
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)
            return False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def paper_count(self) -> int:
        return len(self._papers)

    @property
    def all_papers(self) -> list[dict]:
        return list(self._papers)

    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0


# ── Singleton ─────────────────────────────────────────────────────────────

_store: Optional[EmbeddingStore] = None


def get_store() -> EmbeddingStore:
    global _store
    if _store is None:
        _store = EmbeddingStore()
    return _store
