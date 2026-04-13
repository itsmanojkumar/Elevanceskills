"""
ArXiv paper loader – supports:
  1. arXiv Python API (default, no download required)
  2. Kaggle JSON snapshot (data/arxiv-metadata-oai-snapshot.json)
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import arxiv
import pandas as pd

from src.config import (
    CACHE_FILE,
    DATA_DIR,
    ARXIV_RATE_LIMIT_DELAY,
    INITIAL_PAPER_COUNT,
)

logger = logging.getLogger(__name__)

KAGGLE_SNAPSHOT = DATA_DIR / "arxiv-metadata-oai-snapshot.json"


# ── Data model ────────────────────────────────────────────────────────────

def paper_to_dict(result: arxiv.Result) -> dict:
    """Convert an arxiv.Result to a plain dict."""
    return {
        "id": result.entry_id.split("/abs/")[-1],
        "title": result.title.strip(),
        "abstract": result.summary.strip(),
        "authors": [a.name for a in result.authors],
        "categories": result.categories,
        "published": result.published.strftime("%Y-%m-%d") if result.published else "",
        "updated": result.updated.strftime("%Y-%m-%d") if result.updated else "",
        "pdf_url": result.pdf_url or "",
        "url": result.entry_id,
    }


# ── ArXiv API ────────────────────────────────────────────────────────────

def fetch_papers_from_api(
    query: str,
    max_results: int = 20,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
) -> list[dict]:
    """Search arXiv and return a list of paper dicts."""
    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=ARXIV_RATE_LIMIT_DELAY,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )
    papers = []
    try:
        for result in client.results(search):
            papers.append(paper_to_dict(result))
    except Exception as exc:
        logger.warning("ArXiv API error: %s", exc)
    return papers


def fetch_papers_by_category(
    category: str = "cs",
    max_results: int = INITIAL_PAPER_COUNT,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
) -> list[dict]:
    """Fetch the most recent papers for a top-level category."""
    query = f"cat:{category}.*" if "." not in category else f"cat:{category}"
    return fetch_papers_from_api(query, max_results=max_results, sort_by=sort_by)


def fetch_paper_by_id(arxiv_id: str) -> Optional[dict]:
    """Fetch a single paper by arXiv ID (e.g. '2303.08774')."""
    clean_id = arxiv_id.strip().split("abs/")[-1].split("v")[0]
    client = arxiv.Client()
    search = arxiv.Search(id_list=[clean_id])
    try:
        for result in client.results(search):
            return paper_to_dict(result)
    except Exception as exc:
        logger.warning("Could not fetch paper %s: %s", arxiv_id, exc)
    return None


# ── Kaggle snapshot ──────────────────────────────────────────────────────

def load_kaggle_snapshot(
    categories: list[str] | None = None,
    max_papers: int = 50_000,
) -> list[dict]:
    """
    Stream-load the Kaggle arXiv JSON snapshot.
    Each line is a JSON object (newline-delimited).
    """
    if not KAGGLE_SNAPSHOT.exists():
        return []

    papers: list[dict] = []
    cat_set = set(categories or [])
    logger.info("Loading Kaggle snapshot from %s …", KAGGLE_SNAPSHOT)

    with KAGGLE_SNAPSHOT.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(papers) >= max_papers:
                break
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            paper_cats = raw.get("categories", "").split()
            if cat_set and not any(
                any(pc.startswith(c) for c in cat_set) for pc in paper_cats
            ):
                continue

            # Extract the last version date
            versions = raw.get("versions", [])
            published = versions[0].get("created", "") if versions else ""
            try:
                published = datetime.strptime(
                    published, "%a, %d %b %Y %H:%M:%S %Z"
                ).strftime("%Y-%m-%d")
            except Exception:
                published = ""

            papers.append(
                {
                    "id": raw.get("id", ""),
                    "title": raw.get("title", "").replace("\n", " ").strip(),
                    "abstract": raw.get("abstract", "").replace("\n", " ").strip(),
                    "authors": [
                        a.strip()
                        for a in raw.get("authors", "").split(",")
                        if a.strip()
                    ],
                    "categories": paper_cats,
                    "published": published,
                    "updated": published,
                    "pdf_url": f"https://arxiv.org/pdf/{raw.get('id', '')}",
                    "url": f"https://arxiv.org/abs/{raw.get('id', '')}",
                }
            )

    logger.info("Loaded %d papers from Kaggle snapshot.", len(papers))
    return papers


# ── Cache helpers ─────────────────────────────────────────────────────────

def save_cache(papers: list[dict]) -> None:
    with CACHE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(papers, fh, ensure_ascii=False, indent=2)


def load_cache() -> list[dict]:
    if not CACHE_FILE.exists():
        return []
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def merge_papers(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge paper lists, deduplicating by arXiv ID."""
    seen = {p["id"] for p in existing}
    result = list(existing)
    for p in new:
        if p["id"] not in seen:
            seen.add(p["id"])
            result.append(p)
    return result


# ── Main loader ───────────────────────────────────────────────────────────

def load_papers(
    category: str = "cs",
    force_refresh: bool = False,
    use_kaggle: bool = True,
) -> list[dict]:
    """
    Load papers from cache or fetch fresh ones.
    Priority: Kaggle snapshot > cached API papers > fresh API fetch.
    """
    if use_kaggle and KAGGLE_SNAPSHOT.exists():
        cat_prefix = category.split(".")[0]
        papers = load_kaggle_snapshot(categories=[cat_prefix], max_papers=100_000)
        if papers:
            return papers

    cached = load_cache()
    if cached and not force_refresh:
        return cached

    logger.info("Fetching papers from arXiv API for category '%s' …", category)
    papers = fetch_papers_by_category(category, max_results=INITIAL_PAPER_COUNT)
    if papers:
        merged = merge_papers(cached, papers)
        save_cache(merged)
        return merged

    return cached


def papers_to_dataframe(papers: list[dict]) -> pd.DataFrame:
    """Convert paper dicts to a DataFrame for analytics."""
    df = pd.DataFrame(papers)
    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    if "authors" in df.columns:
        df["author_count"] = df["authors"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        df["first_author"] = df["authors"].apply(
            lambda x: x[0] if isinstance(x, list) and x else "Unknown"
        )
    if "categories" in df.columns:
        df["primary_category"] = df["categories"].apply(
            lambda x: x[0] if isinstance(x, list) and x else "Unknown"
        )
    return df
