"""
RAG pipeline built with LangChain LCEL.

Architecture
────────────
  User query
      │
      ▼
  ArxivRetriever (BaseRetriever → FAISS)
      │ Documents
      ▼
  ChatPromptTemplate  ← MessagesPlaceholder (conversation memory)
      │ Formatted messages
      ▼
  BaseChatModel  (Ollama / Groq / HuggingFace)
      │ AIMessage
      ▼
  StrOutputParser  →  str response

Task variants (chat / summarize / explain) each get their own
PromptTemplate; the correct chain is selected at call time.
"""

from __future__ import annotations

import logging
import re
from typing import Generator, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import Field

from src.config import MAX_CONTEXT_PAPERS, TOP_K_RETRIEVAL
from src.embeddings import EmbeddingStore

logger = logging.getLogger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert AI research assistant specialising in scientific literature, "
    "with deep knowledge of computer science, machine learning, and related fields.\n\n"
    "Your role:\n"
    "- Answer questions accurately using the provided research paper context\n"
    "- Explain complex concepts clearly, using analogies when helpful\n"
    "- Summarise research findings concisely\n"
    "- Cite papers as [Author et al., Year] when referencing them\n"
    "- Acknowledge uncertainty when context is insufficient\n"
    "- Suggest related research directions when appropriate\n\n"
    "Guidelines:\n"
    "- Be precise and technical when needed, accessible when explaining fundamentals\n"
    "- Structure long responses with clear sections (## headings, bullet points)\n"
    "- Always ground answers in the provided paper context\n"
    "- Maintain continuity across the conversation history"
)

# ── Task-specific prompt templates ────────────────────────────────────────

_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "## Retrieved Research Papers\n{context}\n\n"
            "## Question\n{question}",
        ),
    ]
)

_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Please provide a comprehensive summary of the paper below, covering:\n"
            "1. **Main Contribution** – what problem does it solve?\n"
            "2. **Methodology** – key techniques and approaches\n"
            "3. **Results** – key findings and metrics\n"
            "4. **Limitations** – what are the shortcomings?\n"
            "5. **Impact** – why does it matter for the field?\n\n"
            "## Paper\n{context}",
        ),
    ]
)

_EXPLAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Please explain the concept: **{question}**\n\n"
            "Include:\n"
            "- A clear definition\n"
            "- Intuitive analogies or examples\n"
            "- Key technical details\n"
            "- How it is used in practice\n"
            "- References to relevant papers below\n\n"
            "## Relevant Papers\n{context}",
        ),
    ]
)

_TASK_PROMPTS: dict[str, ChatPromptTemplate] = {
    "chat": _CHAT_PROMPT,
    "summarize": _SUMMARIZE_PROMPT,
    "explain": _EXPLAIN_PROMPT,
}


# ── LangChain Retriever wrapping FAISS store ──────────────────────────────

class ArxivRetriever(BaseRetriever):
    """
    LangChain BaseRetriever that wraps our FAISS EmbeddingStore.
    Returns LangChain Documents so it plugs into any LCEL chain.
    """

    store: EmbeddingStore = Field(exclude=True)
    top_k: int = TOP_K_RETRIEVAL
    extra_papers: list[dict] = Field(default_factory=list, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        papers = self.store.search(query, top_k=self.top_k)
        # Merge any injected extra papers (e.g. fetched by arXiv ID)
        if self.extra_papers:
            existing_ids = {p["id"] for p in papers}
            for p in self.extra_papers:
                if p["id"] not in existing_ids:
                    papers.append(p)
        return [_paper_to_document(p) for p in papers[:MAX_CONTEXT_PAPERS]]

    # async variant (required by ABC; delegates to sync)
    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self._get_relevant_documents(query)


def _paper_to_document(paper: dict) -> Document:
    authors = paper.get("authors", [])
    author_str = (
        f"{authors[0].split()[-1]} et al." if len(authors) > 1
        else authors[0] if authors
        else "Unknown"
    )
    year = paper.get("published", "")[:4] or "?"
    cats = ", ".join(paper.get("categories", [])[:2])
    abstract = paper.get("abstract", "")
    if len(abstract) > 500:
        abstract = abstract[:500] + "…"

    content = (
        f"**{paper.get('title', 'Untitled')}**\n"
        f"Authors: {author_str} ({year}) | Categories: {cats}\n"
        f"Abstract: {abstract}\n"
        f"URL: {paper.get('url', '')}"
    )
    return Document(page_content=content, metadata=paper)


def _docs_to_context(docs: list[Document]) -> str:
    if not docs:
        return "No relevant papers found."
    return "\n\n---\n\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)
    )


# ── RAG Pipeline ──────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Conversational RAG pipeline using LangChain LCEL.

    Each call to stream_answer / answer:
      1. Retrieves relevant Documents via ArxivRetriever (FAISS)
      2. Formats context + chat_history into a ChatPromptTemplate
      3. Streams/invokes the LangChain BaseChatModel
      4. Updates the in-memory conversation history
    """

    def __init__(self, store: EmbeddingStore, llm: BaseChatModel) -> None:
        self.store = store
        self.llm = llm
        self.chat_history: list[HumanMessage | AIMessage] = []
        self._last_papers: list[dict] = []
        self._last_docs: list[Document] = []
        self._retriever = ArxivRetriever(store=store)
        self._output_parser = StrOutputParser()
        self._chains = self._build_chains()

    # ── Chain construction (LCEL) ─────────────────────────────────────────

    def _build_chains(self) -> dict[str, object]:
        """
        Build one LCEL chain per task type.

        Chain structure:
          RunnablePassthrough.assign(context=..., chat_history=...)
          | ChatPromptTemplate
          | BaseChatModel
          | StrOutputParser
        """
        chains: dict[str, object] = {}

        for task, prompt_tpl in _TASK_PROMPTS.items():
            retriever = self._retriever  # shared, mutated per-call for extra_papers

            if task == "summarize":
                # summarize doesn't use chat_history
                chain = (
                    RunnablePassthrough.assign(
                        context=RunnableLambda(
                            lambda inputs, r=retriever: _docs_to_context(
                                r._get_relevant_documents(inputs["question"])
                            )
                        )
                    )
                    | prompt_tpl
                    | self.llm
                    | self._output_parser
                )
            else:
                chain = (
                    RunnablePassthrough.assign(
                        context=RunnableLambda(
                            lambda inputs, r=retriever: _docs_to_context(
                                r._get_relevant_documents(inputs["question"])
                            )
                        ),
                        chat_history=RunnableLambda(
                            lambda _: self.chat_history
                        ),
                    )
                    | prompt_tpl
                    | self.llm
                    | self._output_parser
                )

            chains[task] = chain

        return chains

    # ── Retrieval helpers ─────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """Return raw paper dicts for a query (used by the Search tab)."""
        return self.store.search(query, top_k=top_k)

    def _retrieve_docs(
        self, query: str, extra_papers: list[dict] | None = None
    ) -> list[Document]:
        self._retriever.extra_papers = extra_papers or []
        docs = self._retriever._get_relevant_documents(query)
        self._last_docs = docs
        self._last_papers = [doc.metadata for doc in docs]
        return docs

    # ── Public answer methods ─────────────────────────────────────────────

    def answer(
        self,
        query: str,
        task: str = "chat",
        extra_papers: list[dict] | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Synchronous RAG answer.
        Returns (response_text, retrieved_paper_dicts).
        """
        self._retrieve_docs(query, extra_papers)

        chain = self._chains[task]
        response: str = chain.invoke({"question": query})

        self._update_history(query, response)
        return response, self._last_papers

    def stream_answer(
        self,
        query: str,
        task: str = "chat",
        extra_papers: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Streaming RAG answer via LangChain .stream().
        Yields text chunks; updates history when complete.
        """
        self._retrieve_docs(query, extra_papers)

        chain = self._chains[task]
        full_response = ""

        for chunk in chain.stream({"question": query}):
            full_response += chunk
            yield chunk

        self._update_history(query, full_response)

    # ── Convenience wrappers ──────────────────────────────────────────────

    def summarize_paper(self, paper: dict) -> tuple[str, list[dict]]:
        query = paper.get("title", "this paper")
        return self.answer(query, task="summarize", extra_papers=[paper])

    def explain_concept(self, concept: str) -> tuple[str, list[dict]]:
        return self.answer(concept, task="explain")

    def clear_history(self) -> None:
        self.chat_history.clear()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _update_history(self, query: str, response: str) -> None:
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response))
        # Keep only the last 8 messages (4 turns) to avoid context bloat
        if len(self.chat_history) > 8:
            self.chat_history = self.chat_history[-8:]

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def last_papers(self) -> list[dict]:
        return self._last_papers


# ── ArXiv ID extraction ───────────────────────────────────────────────────

def extract_arxiv_id(text: str) -> Optional[str]:
    """Extract an arXiv ID from a URL, DOI-style string, or bare ID."""
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"arxiv:([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"\b([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).split("v")[0]
    return None
