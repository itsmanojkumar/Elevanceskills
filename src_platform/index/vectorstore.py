from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src_platform.common.utils import ensure_dir, read_json, write_json


@dataclass(frozen=True)
class VectorIndexMeta:
    model_name: str
    embedding_dim: int
    doc_count: int


class DynamicKBIndex:
    """
    Separate FAISS index for Task 1 (Dynamic KB).
    Stored under data_platform/dynamic_kb_index/ so it never touches MedQuAD indices.
    """

    def __init__(self, index_dir: Path, model_name: str):
        self.index_dir = index_dir
        self.model_name = model_name
        self.effective_model_name = model_name
        self.meta_file = self.index_dir / "meta.json"
        self.vectorstore = None

    def exists(self) -> bool:
        return self.index_dir.exists() and any(self.index_dir.iterdir())

    def _build_sentence_transformer_embeddings(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")

        class _STEmbeddings:
            def __call__(self, text: str) -> list[float]:
                return self.embed_query(text)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return [list(map(float, e)) for e in embs]

            def embed_query(self, text: str) -> list[float]:
                emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
                return list(map(float, emb))

        return _STEmbeddings()

    def _build_hash_embeddings(self, dim: int = 384):
        """
        Last-resort deterministic embedding fallback to avoid runtime crashes.
        This is lower quality but guarantees index build availability.
        """

        class _HashEmbeddings:
            def __init__(self, size: int):
                self.size = size

            def __call__(self, text: str) -> list[float]:
                return self.embed_query(text)

            def _vec(self, text: str) -> list[float]:
                v = [0.0] * self.size
                tokens = (text or "").lower().split()
                if not tokens:
                    return v
                for t in tokens:
                    h = int(hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest(), 16)
                    idx = h % self.size
                    sign = -1.0 if ((h >> 1) & 1) else 1.0
                    v[idx] += sign
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                return [x / norm for x in v]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self._vec(t) for t in texts]

            def embed_query(self, text: str) -> list[float]:
                return self._vec(text)

        return _HashEmbeddings(dim)

    def _get_embeddings(self):
        """
        Build embeddings model with fallback for environments that hit
        transformers "meta tensor" initialization issues.
        """
        primary = self.model_name
        fallback = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            embeddings = self._build_sentence_transformer_embeddings(primary)
            # Force model readiness early so meta-tensor errors are caught here.
            _ = embeddings.embed_query("embedding warmup")
            self.effective_model_name = primary
            return embeddings
        except Exception:
            try:
                embeddings = self._build_sentence_transformer_embeddings(fallback)
                _ = embeddings.embed_query("embedding warmup")
                self.effective_model_name = fallback
                return embeddings
            except Exception:
                embeddings = self._build_hash_embeddings()
                _ = embeddings.embed_query("embedding warmup")
                self.effective_model_name = "hash-fallback-384"
                return embeddings

    def load(self) -> bool:
        if not self.exists():
            return False
        try:
            from langchain_community.vectorstores import FAISS
            embeddings = self._get_embeddings()
            self.vectorstore = FAISS.load_local(
                str(self.index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            # Dimension check
            query_dim = len(embeddings.embed_query("dimension check"))
            index_dim = int(getattr(self.vectorstore.index, "d", -1))
            if index_dim != query_dim:
                raise RuntimeError(
                    f"DynamicKB index dimension mismatch: index.d={index_dim}, embed_dim={query_dim}. "
                    f"Rebuild the DynamicKB index."
                )
            return True
        except Exception:
            self.vectorstore = None
            return False

    def rebuild_from_documents(self, docs: list[dict[str, Any]]) -> VectorIndexMeta:
        """
        Rebuilds the entire index from docs (safe + simple for incremental ingestion).
        docs items should include:
          - text (chunk text)
          - metadata (dict)
        """
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS

        ensure_dir(self.index_dir)
        embeddings = self._get_embeddings()

        lc_docs = [
            Document(page_content=str(d["text"]), metadata=dict(d.get("metadata", {})))
            for d in docs
        ]
        if not lc_docs:
            raise RuntimeError("No documents to index.")

        vs = FAISS.from_documents(lc_docs, embeddings)
        vs.save_local(str(self.index_dir))
        self.vectorstore = vs

        dim = len(embeddings.embed_query("dimension check"))
        meta = VectorIndexMeta(model_name=self.effective_model_name, embedding_dim=dim, doc_count=len(lc_docs))
        write_json(self.meta_file, {"model_name": meta.model_name, "embedding_dim": meta.embedding_dim, "doc_count": meta.doc_count})
        return meta

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.vectorstore:
            raise RuntimeError("DynamicKB index not loaded.")
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        out: list[dict[str, Any]] = []
        for rank, (doc, score) in enumerate(results, start=1):
            out.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return out

