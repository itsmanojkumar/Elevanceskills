from __future__ import annotations

import json
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
        self.meta_file = self.index_dir / "meta.json"
        self.vectorstore = None

    def exists(self) -> bool:
        return self.index_dir.exists() and any(self.index_dir.iterdir())

    def _get_embeddings(self):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )

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
        meta = VectorIndexMeta(model_name=self.model_name, embedding_dim=dim, doc_count=len(lc_docs))
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

