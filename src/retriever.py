"""
Medical Retrieval System — Local HuggingFace Embeddings + FAISS + Re-ranking.

Two-stage pipeline (no TF-IDF fallback):

  Stage 1 (Recall): sentence-transformers (local) + FAISS vector store via LangChain
    - Embedding model: pritamdeka/S-PubMedBert-MS-MARCO (biomedical, higher accuracy)
    - Indexed text: question + focus + answer
    - FAISS: IndexFlatIP over L2-normalized vectors (exact cosine similarity)

  Stage 2 (Precision): Cross-encoder re-ranker over top-N candidates
    - Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, strong)
    - Scores (query, candidate_text) pairs and returns best top-k

Setup (one-time):
  python setup.py          <- downloads model + builds FAISS index (~5-10 min on CPU)

Runtime:
  Model loaded from disk cache (~/.cache/huggingface/)
  FAISS index loaded from data/faiss_index/
  Query embedding: ~50 ms  |  FAISS search: ~5 ms
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

DATA_DIR  = Path(__file__).parent.parent / "data"
FAISS_DIR = DATA_DIR / "faiss_index"
FAISS_META_FILE = FAISS_DIR / "meta.json"

# Models (all run 100% locally, no API):
#   Best medical embeddings: pritamdeka/S-PubMedBert-MS-MARCO (~420 MB, 768-dim)
#   Fast general           : sentence-transformers/all-MiniLM-L6-v2 (~90 MB, 384-dim)
DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval/re-ranking knobs
DEFAULT_CANDIDATE_POOL = 50  # FAISS candidates → re-ranker → final top_k


@dataclass(frozen=True)
class FaissIndexMeta:
    model_name: str
    embedding_dim: int
    indexed_text: str

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "indexed_text": self.indexed_text,
        }

    @staticmethod
    def from_dict(d: dict) -> "FaissIndexMeta":
        return FaissIndexMeta(
            model_name=str(d.get("model_name", "")),
            embedding_dim=int(d.get("embedding_dim", 0)),
            indexed_text=str(d.get("indexed_text", "")),
        )

# ---------------------------------------------------------------------------
# Embedding helper (lazy singleton)
# ---------------------------------------------------------------------------
_embed_cache: dict = {}


def _get_embeddings(model_name: str = DEFAULT_MODEL):
    """
    Return a LangChain HuggingFaceEmbeddings instance (lazy, cached).
    Model weights are downloaded once and cached in ~/.cache/huggingface/.
    """
    if model_name not in _embed_cache:
        try:
            # Newer, non-deprecated implementation
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        print(f"Loading local embedding model: {model_name}")
        _embed_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        print("Embedding model ready.")
    return _embed_cache[model_name]

_rerank_cache: dict = {}


def _get_reranker(model_name: str = DEFAULT_RERANK_MODEL):
    """Lazy-load a cross-encoder re-ranker (CPU)."""
    if model_name not in _rerank_cache:
        from sentence_transformers import CrossEncoder
        print(f"Loading cross-encoder re-ranker: {model_name}")
        _rerank_cache[model_name] = CrossEncoder(model_name, device="cpu")
        print("Re-ranker ready.")
    return _rerank_cache[model_name]


# ---------------------------------------------------------------------------
# FAISS retriever (primary)
# ---------------------------------------------------------------------------
class FAISSRetriever:
    """
    Semantic retriever using local sentence-transformers + LangChain FAISS.
    No API calls. Model is downloaded once (~90 MB) and cached locally.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name  = model_name
        self.vectorstore = None
        self._loaded     = False

    def build_index(self, records: list[dict]) -> None:
        """
        Embed all records locally and save a FAISS index to disk.
        Run once via setup.py — takes ~5-10 min on CPU for 47k docs.
        """
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        from tqdm import tqdm

        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        embeddings = _get_embeddings(self.model_name)

        print(f"Building FAISS index: {len(records):,} documents...")
        print(f"Model: {self.model_name}  |  Device: CPU")
        print("This runs fully locally — no API calls, no internet required.\n")

        docs = [
            Document(
                page_content=f"{r['question']} {r.get('focus', '')} {r.get('answer', '')}",
                metadata={
                    "question":       r["question"],
                    "answer":         r["answer"],
                    "focus":          r.get("focus", ""),
                    "source":         r.get("source", ""),
                    "url":            r.get("url", ""),
                    "question_type":  r.get("question_type", ""),
                    "semantic_types": json.dumps(r.get("semantic_types", [])),
                },
            )
            for r in records
        ]

        # Embed in batches with progress bar
        batch_size = 128
        all_texts  = [d.page_content for d in docs]
        all_vecs: list[list[float]] = []

        for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding"):
            batch = all_texts[i : i + batch_size]
            vecs  = embeddings.embed_documents(batch)
            all_vecs.extend(vecs)

        # Build FAISS index (inner product on L2-normalized = cosine similarity)
        import faiss
        dim    = len(all_vecs[0])
        matrix = np.array(all_vecs, dtype="float32")
        faiss.normalize_L2(matrix)
        idx = faiss.IndexFlatIP(dim)
        idx.add(matrix)

        # Save via LangChain FAISS wrapper
        self.vectorstore = FAISS.from_documents([docs[0]], embeddings)
        self.vectorstore.index = idx
        # Rebuild docstore with all documents
        from langchain_community.vectorstores.faiss import dependable_faiss_import
        from langchain_community.docstore.in_memory import InMemoryDocstore
        import uuid
        doc_ids  = [str(uuid.uuid4()) for _ in docs]
        docstore = InMemoryDocstore({did: doc for did, doc in zip(doc_ids, docs)})
        self.vectorstore.docstore           = docstore
        self.vectorstore.index_to_docstore_id = {i: did for i, did in enumerate(doc_ids)}

        self.vectorstore.save_local(str(FAISS_DIR))
        # Persist build metadata to detect dimension/model mismatches at runtime
        meta = FaissIndexMeta(
            model_name=self.model_name,
            embedding_dim=dim,
            indexed_text="question + focus + answer",
        )
        try:
            FAISS_META_FILE.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Warning: could not write FAISS meta file: {e}")
        self._loaded = True
        print(f"\nFAISS index saved → {FAISS_DIR}  (dim={dim}, docs={len(docs):,})")

    def load_index(self) -> bool:
        """Load pre-built FAISS index from disk."""
        if not FAISS_DIR.exists() or not any(FAISS_DIR.iterdir()):
            return False
        try:
            from langchain_community.vectorstores import FAISS
            embeddings = _get_embeddings(self.model_name)
            self.vectorstore = FAISS.load_local(
                str(FAISS_DIR), embeddings,
                allow_dangerous_deserialization=True,
            )
            # Validate embedding dimension matches FAISS index dimension to avoid:
            # faiss.class_wrappers.replacement_search: assert d == self.d
            query_dim = len(embeddings.embed_query("dimension check"))
            index_dim = int(getattr(self.vectorstore.index, "d", -1))
            if index_dim != query_dim:
                msg = (
                    f"FAISS index dimension mismatch: index.d={index_dim}, embed_dim={query_dim}. "
                    f"This usually happens when the index was built with a different embedding model. "
                    f"Delete '{FAISS_DIR}' and rebuild with: python setup.py"
                )
                raise RuntimeError(msg)
            self._loaded = True
            print("FAISS index loaded from disk.")
            return True
        except Exception as e:
            print(f"FAISS load failed: {e}")
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._loaded:
            raise RuntimeError("FAISS index not loaded.")
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        output  = []
        for rank, (doc, score) in enumerate(results, 1):
            m = doc.metadata
            output.append({
                "rank":           rank,
                "score":          float(score),
                "question":       m.get("question", ""),
                "answer":         m.get("answer", ""),
                "focus":          m.get("focus", ""),
                "source":         m.get("source", ""),
                "url":            m.get("url", ""),
                "question_type":  m.get("question_type", ""),
                "semantic_types": json.loads(m.get("semantic_types", "[]")),
                "retriever":      "faiss",
            })
        return output


# ---------------------------------------------------------------------------
# MedicalRetriever (FAISS only)
# ---------------------------------------------------------------------------
class MedicalRetriever:
    """
    FAISS-only retriever — requires a built FAISS index in data/faiss_index/.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ):
        self._model_name = model_name
        self._rerank_model = rerank_model
        self._candidate_pool = candidate_pool
        self._faiss  = FAISSRetriever(model_name)
        self._active = "none"  # "faiss" | "none"

    @property
    def active_retriever(self) -> str:
        return self._active

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def rerank_model(self) -> str:
        return self._rerank_model

    @property
    def documents(self) -> list[dict]:
        # Documents are stored inside the FAISS docstore; we don't expose the full
        # list here to avoid loading them eagerly into UI.
        return []

    def load_index(self) -> bool:
        if self._faiss.load_index():
            self._active = "faiss"
            print("Active retriever: FAISS (local HuggingFace embeddings)")
            return True
        return False

    def build_faiss(self, records: list[dict]) -> None:
        self._faiss.build_index(records)
        self._active = "faiss"

    def is_loaded(self) -> bool:
        return self._active != "none"

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self._active != "faiss":
            raise RuntimeError("FAISS index not loaded. Run setup.py first.")
        
        print(f"\n--- Processing Query: '{query}' ---")
        # Stage 1: recall more candidates from FAISS
        pool_k = max(top_k, self._candidate_pool)
        print(f"Stage 1: Fetching {pool_k} candidates from FAISS...")
        candidates = self._faiss.retrieve(query, top_k=pool_k)
        if not candidates:
            print("No candidates found.")
            return []

        # Stage 2: cross-encoder re-rank top-N
        print(f"Stage 2: Re-ranking top {len(candidates)} candidates with Cross-Encoder...")
        reranker = _get_reranker(self._rerank_model)
        topN = candidates[: self._candidate_pool]
        pairs = [
            (query, f"{c.get('question','')} {c.get('focus','')} {c.get('answer','')}")
            for c in topN
        ]
        scores = np.asarray(reranker.predict(pairs), dtype="float32")
        order = np.argsort(scores)[::-1][:top_k]

        reranked: list[dict] = []
        for rank, idx in enumerate(order, start=1):
            c = topN[int(idx)].copy()
            c["rank"] = rank
            c["score"] = float(scores[int(idx)])
            c["retriever"] = "faiss+rerank"
            reranked.append(c)
        
        print(f"Done. Top result score: {reranked[0]['score']:.4f}")
        return reranked

    def get_answer(self, query: str, top_k: int = 3) -> dict:
        results = self.retrieve(query, top_k=top_k)

        if not results or results[0]["score"] < 0.05:
            return {
                "found":     False,
                "message":   (
                    "I could not find a relevant answer in the MedQuAD dataset. "
                    "Please consult a qualified healthcare professional."
                ),
                "results":   [],
                "retriever": self._active,
            }

        best = results[0]
        return {
            "found":         True,
            "best_answer":   best["answer"],
            "best_question": best["question"],
            "focus":         best["focus"],
            "source":        best["source"],
            "url":           best["url"],
            "confidence":    best["score"],
            "question_type": best["question_type"],
            "related":       results[1:],
            "retriever":     self._active,
        }
