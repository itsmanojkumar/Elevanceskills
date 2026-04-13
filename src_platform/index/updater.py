from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src_platform.common.utils import ensure_dir, read_json, sha256_text, write_json
from src_platform.ingest.chunking import simple_chunk
from src_platform.ingest.fetchers import (
    FetchedDoc,
    fetch_pdf_file,
    fetch_pdf_url,
    fetch_rss_feed,
    fetch_sitemap_urls,
    fetch_web_page,
)


@dataclass(frozen=True)
class SourceSpec:
    kind: str  # "web" | "rss" | "sitemap" | "pdf" | "pdf_file"
    value: str


def _state_schema() -> dict:
    return {
        "sources": [],  # list[{kind, value}]
        "docs": [],     # list[{doc_id, source_id, uri, title, content_hash, chunks: [..]}]
        "chunks": [],   # list[{chunk_id, doc_id, text, metadata}]
    }


def load_state(state_file: Path) -> dict:
    return read_json(state_file, default=_state_schema())


def save_state(state_file: Path, state: dict) -> None:
    write_json(state_file, state)


def add_source(state: dict, spec: SourceSpec) -> dict:
    sources = list(state.get("sources") or [])
    entry = {"kind": spec.kind, "value": spec.value}
    if entry not in sources:
        sources.append(entry)
    state["sources"] = sources
    return state


def remove_source(state: dict, idx: int) -> dict:
    sources = list(state.get("sources") or [])
    if 0 <= idx < len(sources):
        sources.pop(idx)
    state["sources"] = sources
    return state


def update_from_sources(state_file: Path, *, max_chars: int = 900, overlap: int = 120) -> dict:
    """
    Fetch sources listed in state, generate chunks, and update the persisted state.
    Dedup is done by content hash per source URI.
    """
    state = load_state(state_file)
    sources = list(state.get("sources") or [])
    docs = list(state.get("docs") or [])
    chunks = list(state.get("chunks") or [])

    existing_doc_hashes = {d.get("content_hash") for d in docs}
    existing_chunk_ids = {c.get("chunk_id") for c in chunks}

    def _ingest_doc(fetched: FetchedDoc) -> None:
        nonlocal docs, chunks, existing_doc_hashes, existing_chunk_ids
        content_hash = sha256_text(fetched.text)
        if content_hash in existing_doc_hashes:
            return

        doc_id = sha256_text(f"{fetched.source_id}:{fetched.uri}:{content_hash}")
        doc_entry: dict[str, Any] = {
            "doc_id": doc_id,
            "source_id": fetched.source_id,
            "uri": fetched.uri,
            "title": fetched.title,
            "content_hash": content_hash,
        }
        docs.append(doc_entry)
        existing_doc_hashes.add(content_hash)

        parts = simple_chunk(fetched.text, max_chars=max_chars, overlap=overlap)
        for i, part in enumerate(parts):
            chunk_id = sha256_text(f"{doc_id}:{i}:{sha256_text(part)}")
            if chunk_id in existing_chunk_ids:
                continue
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": part,
                    "metadata": {
                        "doc_id": doc_id,
                        "source_id": fetched.source_id,
                        "uri": fetched.uri,
                        "title": fetched.title,
                        "chunk_index": i,
                    },
                }
            )
            existing_chunk_ids.add(chunk_id)

    for s in sources:
        kind = (s.get("kind") or "").strip()
        value = (s.get("value") or "").strip()
        if not kind or not value:
            continue

        try:
            if kind == "web":
                _ingest_doc(fetch_web_page(value))
            elif kind == "rss":
                for fd in fetch_rss_feed(value):
                    _ingest_doc(fd)
            elif kind == "pdf":
                _ingest_doc(fetch_pdf_url(value))
            elif kind == "pdf_file":
                _ingest_doc(fetch_pdf_file(value))
            elif kind == "sitemap":
                for page_url in fetch_sitemap_urls(value):
                    try:
                        _ingest_doc(fetch_web_page(page_url))
                    except Exception:
                        continue
        except Exception:
            # Keep updater resilient across mixed-quality sources.
            continue

    state["docs"] = docs
    state["chunks"] = chunks
    save_state(state_file, state)
    return state

