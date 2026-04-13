from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests
from lxml import html as lxml_html


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

