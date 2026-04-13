from __future__ import annotations

import streamlit as st

from src_platform.config import Settings
from src_platform.index.vectorstore import DynamicKBIndex
from src_platform.nlp.lang import SUPPORTED_LANGS, detect_language, translate


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
                out = translate(sample, src=src, tgt=tgt)
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
                q_en = translate(query, src=q_lang, tgt="en")

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
                        snippet = translate(snippet, src="en", tgt=out_lang)
                    except Exception:
                        pass
            st.markdown(
                f"**#{r['rank']}** score={r['score']:.4f}  \n"
                f"- **title**: {md.get('title','')}  \n"
                f"- **uri**: {md.get('uri','')}  \n"
            )
            st.write(snippet)
            st.markdown("---")

