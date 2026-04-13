from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_id: str
    uri: str
    title: str
    text: str


def simple_chunk(text: str, *, max_chars: int = 900, overlap: int = 120) -> list[str]:
    """
    Simple character-based chunking with overlap.
    Good enough for web pages / notes without extra deps.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(text):
            break
        i = max(0, j - overlap)
    return chunks

