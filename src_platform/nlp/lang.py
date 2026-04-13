from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


SUPPORTED_LANGS = ["en", "es", "fr", "ar"]


@dataclass(frozen=True)
class LangDetectResult:
    lang: str
    confidence: float


def detect_language(text: str) -> LangDetectResult:
    """
    Detect language using langdetect (fast, lightweight).
    """
    try:
        from langdetect import detect_langs  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: langdetect. Install with `pip install langdetect`.") from e

    text = (text or "").strip()
    if not text:
        return LangDetectResult(lang="en", confidence=0.0)

    langs = detect_langs(text)
    if not langs:
        return LangDetectResult(lang="en", confidence=0.0)
    best = langs[0]
    code = str(getattr(best, "lang", "en"))
    prob = float(getattr(best, "prob", 0.0))
    if code not in SUPPORTED_LANGS:
        code = "en"
    return LangDetectResult(lang=code, confidence=prob)


def _opus_model_name(src: str, tgt: str) -> str:
    # Helsinki-NLP/opus-mt-{src}-{tgt}
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"


@lru_cache(maxsize=8)
def _get_translator(src: str, tgt: str):
    """
    Cached MarianMT translator pipeline. Downloads model once.
    """
    from transformers import MarianMTModel, MarianTokenizer  # type: ignore

    name = _opus_model_name(src, tgt)
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    return tok, model


def translate(text: str, *, src: str, tgt: str) -> str:
    """
    Local translation via MarianMT (Opus-MT).
    Note: This downloads models (one-time) on first use.
    """
    text = (text or "").strip()
    if not text or src == tgt:
        return text

    tok, model = _get_translator(src, tgt)
    batch = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    gen = model.generate(**batch, max_new_tokens=512)
    out = tok.batch_decode(gen, skip_special_tokens=True)
    return (out[0] if out else "").strip()

