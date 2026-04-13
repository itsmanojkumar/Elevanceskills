from __future__ import annotations

import re
from io import BytesIO
from dataclasses import dataclass
from typing import Optional

import feedparser
import requests
from lxml import html as lxml_html
from lxml import etree
from pypdf import PdfReader


@dataclass(frozen=True)
class FetchedDoc:
    source_id: str
    uri: str
    title: str
    text: str


def fetch_web_page(url: str, timeout_s: int = 30) -> FetchedDoc:
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "MedquadBot/1.0"})
    resp.raise_for_status()
    tree = lxml_html.fromstring(resp.content)

    # Extract title
    title = ""
    try:
        t = tree.xpath("//title/text()")
        if t:
            title = str(t[0]).strip()
    except Exception:
        title = ""

    # Remove scripts/styles
    for bad in tree.xpath("//script|//style|//noscript"):
        try:
            bad.getparent().remove(bad)
        except Exception:
            pass

    raw_text = tree.text_content()
    # Normalize whitespace
    raw_text = re.sub(r"[ \t]+", " ", raw_text)
    raw_text = re.sub(r"\n\s*\n+", "\n\n", raw_text).strip()

    source_id = f"web:{url}"
    return FetchedDoc(source_id=source_id, uri=url, title=title or url, text=raw_text)


def fetch_rss_feed(url: str, timeout_s: int = 30, max_items: int = 20) -> list[FetchedDoc]:
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "MedquadBot/1.0"})
    resp.raise_for_status()

    parsed = feedparser.parse(resp.content)
    docs: list[FetchedDoc] = []
    for i, entry in enumerate(parsed.entries[:max_items]):
        link = str(getattr(entry, "link", "") or "").strip()
        title = str(getattr(entry, "title", "") or "").strip() or f"RSS item {i+1}"
        summary = str(getattr(entry, "summary", "") or "").strip()
        content = ""
        if hasattr(entry, "content") and entry.content:
            first = entry.content[0]
            content = str(getattr(first, "value", "") or "").strip()
        text_raw = f"{title}\n\n{summary}\n\n{content}".strip()
        text_raw = re.sub(r"<[^>]+>", " ", text_raw)  # strip simple HTML tags
        text_raw = re.sub(r"[ \t]+", " ", text_raw)
        text_raw = re.sub(r"\n\s*\n+", "\n\n", text_raw).strip()
        if not text_raw:
            continue
        uri = link or f"{url}#item-{i+1}"
        docs.append(
            FetchedDoc(
                source_id=f"rss:{url}",
                uri=uri,
                title=title,
                text=text_raw,
            )
        )
    return docs


def fetch_pdf_url(url: str, timeout_s: int = 30) -> FetchedDoc:
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "MedquadBot/1.0"})
    resp.raise_for_status()
    reader = PdfReader(BytesIO(resp.content))
    pages_text: list[str] = []
    for p in reader.pages:
        try:
            pages_text.append(p.extract_text() or "")
        except Exception:
            continue
    text = "\n\n".join(pages_text).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if not text:
        text = "No extractable text found in PDF."
    return FetchedDoc(
        source_id=f"pdf:{url}",
        uri=url,
        title=url.split("/")[-1] or "PDF Document",
        text=text,
    )


def fetch_pdf_file(path: str) -> FetchedDoc:
    p = BytesIO()
    with open(path, "rb") as fh:
        p.write(fh.read())
    p.seek(0)
    reader = PdfReader(p)
    pages_text: list[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n\n".join(pages_text).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if not text:
        text = "No extractable text found in PDF."
    name = path.replace("\\", "/").split("/")[-1] or "uploaded.pdf"
    return FetchedDoc(
        source_id=f"pdf_file:{path}",
        uri=f"file://{path}",
        title=name,
        text=text,
    )


def fetch_sitemap_urls(url: str, timeout_s: int = 30, max_urls: int = 30) -> list[str]:
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "MedquadBot/1.0"})
    resp.raise_for_status()
    root = etree.fromstring(resp.content)
    locs = root.xpath("//*[local-name()='loc']/text()")
    out: list[str] = []
    for loc in locs:
        if not isinstance(loc, str):
            continue
        v = loc.strip()
        if v:
            out.append(v)
        if len(out) >= max_urls:
            break
    return out

