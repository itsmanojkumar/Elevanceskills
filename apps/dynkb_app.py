from __future__ import annotations

import streamlit as st

from src_platform.config import Settings
from src_platform.common.utils import ensure_dir
from src_platform.index.updater import SourceSpec, add_source, load_state, remove_source, save_state, update_from_sources
from src_platform.index.vectorstore import DynamicKBIndex


DEFAULT_DYNKB_EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def render_dynkb_app(settings: Settings) -> None:
    st.header("Task 1 — Dynamic Knowledge Base")
    st.caption("User-provided sources → ingestion → chunking → separate FAISS index (no MedQuAD changes).")

    ensure_dir(settings.data_platform_dir)

    # Load persisted state
    state = load_state(settings.dynkb_state_file)

    st.subheader("Sources")
    with st.form("add_source_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 5])
        kind = c1.selectbox("Type", options=["web"], index=0)
        value = c2.text_input("URL", placeholder="https://example.com/page")
        submitted = st.form_submit_button("Add source")
        if submitted:
            if not value.strip():
                st.warning("Please enter a URL.")
            else:
                state = add_source(state, SourceSpec(kind=kind, value=value.strip()))
                save_state(settings.dynkb_state_file, state)
                st.success("Source added.")
                st.rerun()

    sources = list(state.get("sources") or [])
    if sources:
        for i, s in enumerate(sources):
            cols = st.columns([1, 6, 1])
            cols[0].write(f"`{s.get('kind','')}`")
            cols[1].write(s.get("value", ""))
            if cols[2].button("Remove", key=f"rm_{i}"):
                state = remove_source(state, i)
                save_state(settings.dynkb_state_file, state)
                st.rerun()
    else:
        st.info("No sources yet. Add at least one web URL.")

    st.subheader("Ingestion / Index")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        do_update = st.button("Update now", use_container_width=True)
    with c2:
        rebuild = st.button("Rebuild index", use_container_width=True)
    with c3:
        interval_min = st.number_input("Auto-update interval (minutes, 0=off)", min_value=0, max_value=240, value=0)

    if interval_min and interval_min > 0:
        # Streamlit auto-refresh while this page is open
        st.autorefresh(interval=int(interval_min * 60 * 1000), key="dynkb_autorefresh")

    if do_update or (interval_min and interval_min > 0):
        with st.spinner("Fetching sources and updating state..."):
            state = update_from_sources(settings.dynkb_state_file)
        st.success(f"Updated. Total docs: {len(state.get('docs', [])):,} | chunks: {len(state.get('chunks', [])):,}")

    # Index rebuild step
    chunks = list(state.get("chunks") or [])
    index = DynamicKBIndex(settings.dynkb_index_dir, model_name=DEFAULT_DYNKB_EMBED_MODEL)

    if rebuild:
        if not chunks:
            st.error("No chunks available. Add sources and click Update now first.")
        else:
            with st.spinner("Building FAISS index for Dynamic KB..."):
                meta = index.rebuild_from_documents(
                    [{"text": c["text"], "metadata": c.get("metadata", {})} for c in chunks]
                )
            st.success(f"Index built: docs={meta.doc_count:,} dim={meta.embedding_dim} model={meta.model_name}")

    loaded = index.load()
    st.write(f"**Index status**: {'✅ loaded' if loaded else '❌ not built yet'}")

    st.subheader("Chat over Dynamic KB")
    q = st.text_input("Ask a question about ingested sources")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

    if st.button("Search", use_container_width=True) and q.strip():
        if not index.vectorstore:
            st.error("Index not ready. Click Rebuild index.")
        else:
            with st.spinner("Searching..."):
                results = index.search(q.strip(), k=int(top_k))
            for r in results:
                md = r.get("metadata") or {}
                st.markdown(
                    f"**#{r['rank']}** score={r['score']:.4f}  \n"
                    f"- **title**: {md.get('title','')}  \n"
                    f"- **uri**: {md.get('uri','')}  \n"
                )
                st.write(r["text"])
                st.markdown("---")

