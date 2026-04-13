from __future__ import annotations

import streamlit as st

from src_platform.config import Settings
from src_platform.index.vectorstore import DynamicKBIndex
from src_platform.llm.gemini_client import GeminiClient
from src_platform.nlp.lang import SUPPORTED_LANGS, detect_language, translate


def _show_translation_error(exc: Exception) -> None:
    msg = str(exc)
    if "SentencePiece" in msg or "sentencepiece" in msg:
        st.error(
            "Translation backend dependency is missing: `sentencepiece`.\n\n"
            "Install it and restart the app:\n"
            "`pip install sentencepiece` or `pip install -r requirements.txt`"
        )
        return
    if "meta tensor" in msg.lower() or "to_empty()" in msg:
        st.error(
            "Local translation model failed to initialize on this environment.\n\n"
            "Use GEMINI_API_KEY for LLM fallback translation, or restart after reinstalling dependencies."
        )
        return
    st.error(f"Translation failed: {msg}")


def _translate_with_fallback(text: str, *, src: str, tgt: str, settings: Settings) -> str:
    """
    Try local Marian translation first; fallback to Gemini if available.
    """
    try:
        return translate(text, src=src, tgt=tgt)
    except Exception as exc:
        msg = str(exc)
        meta_tensor_err = "meta tensor" in msg.lower() or "to_empty()" in msg.lower()
        if settings.gemini_api_key:
            client = GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
            prompt = (
                f"Translate the following text from {src} to {tgt}.\n"
                "Preserve meaning, tone, and intent. Return only translated text.\n\n"
                f"Text:\n{text}"
            )
            resp = client.generate_text(prompt)
            out = (resp.text or "").strip()
            if out:
                st.caption("Used LLM fallback translation for reliability.")
                return out
        if meta_tensor_err:
            raise RuntimeError(
                "Local translation model failed to load on this environment. "
                "Set GEMINI_API_KEY for LLM fallback translation."
            ) from exc
        raise


def render_multilingual_app(settings: Settings) -> None:
    st.header("Task 4 — Multilingual Support")

    st.markdown(
        "This module provides:\n"
        "- **Language detection** (local)\n"
        "- **Local translation** for supported languages via MarianMT (downloads models on first use)\n"
        "- Optional: multilingual search over Task 1 Dynamic KB\n"
    )

    st.subheader("Language detection")
    sample = st.text_area("Text", height=90, placeholder="Write in English, Spanish, French, or Arabic…")
    if st.button("Detect language", use_container_width=True):
        r = detect_language(sample)
        st.write(f"Detected: `{r.lang}` (confidence {r.confidence:.2f})")

    st.subheader("Translate")
    c1, c2 = st.columns(2)
    with c1:
        src = st.selectbox("Source language", options=SUPPORTED_LANGS, index=0)
    with c2:
        tgt = st.selectbox("Target language", options=SUPPORTED_LANGS, index=1)
    if st.button("Translate text", use_container_width=True):
        if not sample.strip():
            st.warning("Enter text above.")
        else:
            with st.spinner("Translating (may download model first time)..."):
                try:
                    out = _translate_with_fallback(sample, src=src, tgt=tgt, settings=settings)
                except Exception as exc:
                    _show_translation_error(exc)
                    return
            st.text_area("Translated output", value=out, height=120)

    st.subheader("Multilingual search over Dynamic KB (Task 1)")
    st.caption("Requires the Dynamic KB index to be built in `data_platform/dynamic_kb_index/`.")

    query = st.text_input("Query (any supported language)")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, key="ml_topk")
    out_lang = st.selectbox("Show results in", options=SUPPORTED_LANGS, index=0, key="ml_outlang")

    if st.button("Search Dynamic KB", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query.")
            return

        det = detect_language(query)
        q_lang = det.lang
        q_en = query.strip()
        if q_lang != "en":
            with st.spinner("Translating query to English..."):
                try:
                    q_en = _translate_with_fallback(query, src=q_lang, tgt="en", settings=settings)
                except Exception as exc:
                    _show_translation_error(exc)
                    return

        idx = DynamicKBIndex(settings.dynkb_index_dir, model_name="pritamdeka/S-PubMedBert-MS-MARCO")
        if not idx.load():
            st.error("Dynamic KB index not found. Build it from Task 1 page (Update now → Rebuild index).")
            return

        with st.spinner("Searching..."):
            results = idx.search(q_en, k=int(top_k))

        st.write(f"Query language: `{q_lang}` → searched in English")

        for r in results:
            md = r.get("metadata") or {}
            snippet = r["text"]
            if out_lang != "en":
                with st.spinner("Translating snippet..."):
                    try:
                        snippet = _translate_with_fallback(snippet, src="en", tgt=out_lang, settings=settings)
                    except Exception as exc:
                        _show_translation_error(exc)
                        return
            st.markdown(
                f"**#{r['rank']}** score={r['score']:.4f}  \n"
                f"- **title**: {md.get('title','')}  \n"
                f"- **uri**: {md.get('uri','')}  \n"
            )
            st.write(snippet)
            st.markdown("---")

    st.subheader("Culturally Appropriate Response (LLM)")
    st.caption(
        "Generates a culturally aware response in the user's detected language. "
        "Requires GEMINI_API_KEY."
    )
    user_msg = st.text_area("User message for culturally aware response", height=100, key="ml_cultural_input")
    if st.button("Generate culturally aware response", use_container_width=True):
        if not user_msg.strip():
            st.warning("Enter a message.")
            return
        det = detect_language(user_msg)
        if not settings.gemini_api_key:
            st.error("GEMINI_API_KEY is required for culturally aware LLM responses.")
            return
        client = GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
        prompt = (
            f"You are a multilingual support assistant. User language: {det.lang}.\n"
            "Respond in the same language as the user.\n"
            "Be culturally appropriate, polite, and clear for that language audience.\n"
            "Avoid stereotypes. Keep it practical and concise.\n\n"
            f"User message:\n{user_msg}"
        )
        with st.spinner("Generating response..."):
            resp = client.generate_text(prompt)
        st.write(resp.text or "(empty)")

