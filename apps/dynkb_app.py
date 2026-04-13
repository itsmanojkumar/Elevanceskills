from __future__ import annotations

import hashlib

import streamlit as st

from src_platform.config import Settings
from src_platform.common.utils import ensure_dir
from src_platform.index.updater import SourceSpec, add_source, load_state, remove_source, save_state, update_from_sources
from src_platform.index.vectorstore import DynamicKBIndex
from src_platform.llm.gemini_client import GeminiClient


DEFAULT_DYNKB_EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
RAG_TOP_K = 5


def _build_rag_prompt(question: str, results: list[dict]) -> str:
    context_blocks: list[str] = []
    for r in results[:RAG_TOP_K]:
        md = r.get("metadata") or {}
        context_blocks.append(
            "\n".join(
                [
                    f"[Source #{r['rank']}]",
                    f"title: {md.get('title', '')}",
                    f"uri: {md.get('uri', '')}",
                    f"content: {r.get('text', '')}",
                ]
            )
        )
    context = "\n\n---\n\n".join(context_blocks)
    return (
        "You are a factual assistant answering from retrieved Dynamic KB context.\n"
        "Use only the provided context. If insufficient, say what is missing.\n"
        "Cite supporting source numbers like [Source #2].\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context (top {RAG_TOP_K}):\n{context}\n\n"
        "Answer:"
    )


def render_dynkb_app(settings: Settings) -> None:
    st.header("Task 1 — Dynamic Knowledge Base")
    st.caption("User-provided sources → ingestion → chunking → separate FAISS index (no MedQuAD changes).")

    ensure_dir(settings.data_platform_dir)

    # Load persisted state
    state = load_state(settings.dynkb_state_file)

    st.subheader("Sources")
    with st.form("add_source_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 5])
        kind = c1.selectbox("Type", options=["web", "rss", "sitemap", "pdf"], index=0)
        placeholder_map = {
            "web": "https://example.com/page",
            "rss": "https://example.com/feed.xml",
            "sitemap": "https://example.com/sitemap.xml",
            "pdf": "https://example.com/file.pdf",
        }
        value = c2.text_input("Source URL", placeholder=placeholder_map.get(kind, "https://example.com"))
        submitted = st.form_submit_button("Add source")
        if submitted:
            if not value.strip():
                st.warning("Please enter a URL.")
            else:
                state = add_source(state, SourceSpec(kind=kind, value=value.strip()))
                save_state(settings.dynkb_state_file, state)
                st.success("Source added.")
                st.rerun()

    st.caption("Or upload a PDF file directly")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="dynkb_pdf_upload")
    if uploaded_pdf is not None:
        if st.button("Add uploaded PDF", use_container_width=True):
            uploads_dir = settings.data_platform_dir / "uploads"
            ensure_dir(uploads_dir)
            data = uploaded_pdf.getvalue()
            digest = hashlib.sha256(data).hexdigest()[:12]
            safe_name = uploaded_pdf.name.replace(" ", "_")
            out_path = uploads_dir / f"{digest}_{safe_name}"
            out_path.write_bytes(data)
            state = add_source(state, SourceSpec(kind="pdf_file", value=str(out_path)))
            save_state(settings.dynkb_state_file, state)
            st.success(f"Uploaded PDF added: {out_path.name}")
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

    # Index rebuild step
    chunks = list(state.get("chunks") or [])
    index = DynamicKBIndex(settings.dynkb_index_dir, model_name=DEFAULT_DYNKB_EMBED_MODEL)

    if do_update or (interval_min and interval_min > 0):
        with st.spinner("Fetching sources and updating state..."):
            state = update_from_sources(settings.dynkb_state_file)
        st.success(f"Updated. Total docs: {len(state.get('docs', [])):,} | chunks: {len(state.get('chunks', [])):,}")

        # Keep index in sync after ingestion so users can search immediately.
        updated_chunks = list(state.get("chunks") or [])
        chunks = updated_chunks
        if updated_chunks:
            with st.spinner("Syncing FAISS index with updated chunks..."):
                meta = index.rebuild_from_documents(
                    [{"text": c["text"], "metadata": c.get("metadata", {})} for c in updated_chunks]
                )
            st.success(
                f"Index synced: docs={meta.doc_count:,} dim={meta.embedding_dim} model={meta.model_name}"
            )

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
    if not loaded and chunks:
        st.caption("Tip: Click **Rebuild index** once after new ingestion to enable search.")

    st.subheader("Chat over Dynamic KB")
    q = st.text_input("Ask a question about ingested sources")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    use_llm = st.toggle(
        f"Use LLM answer synthesis from top {RAG_TOP_K}",
        value=bool(settings.gemini_api_key),
        help="If enabled and GEMINI_API_KEY exists, retrieves top 5 chunks and asks Gemini to answer from them.",
    )

    if st.button("Search", use_container_width=True) and q.strip():
        if not index.vectorstore:
            st.error("Index not ready. Click Rebuild index.")
        else:
            with st.spinner("Searching..."):
                requested_k = max(int(top_k), RAG_TOP_K if use_llm else int(top_k))
                results = index.search(q.strip(), k=requested_k)

            if use_llm:
                if not settings.gemini_api_key:
                    st.warning("LLM is enabled, but GEMINI_API_KEY is missing. Showing retrieval-only results.")
                else:
                    try:
                        prompt = _build_rag_prompt(q.strip(), results[:RAG_TOP_K])
                        client = GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
                        with st.spinner(f"Generating answer with {settings.gemini_model} from top {RAG_TOP_K}..."):
                            resp = client.generate_text(prompt)
                        st.markdown("### LLM Answer (RAG)")
                        st.write(resp.text or "(empty)")
                    except Exception as exc:
                        st.warning(f"LLM synthesis failed, showing retrieval-only results. Error: {exc}")

            st.markdown(f"### Retrieved Passages (Top {min(len(results), int(top_k))})")
            for r in results[: int(top_k)]:
                md = r.get("metadata") or {}
                st.markdown(
                    f"**#{r['rank']}** score={r['score']:.4f}  \n"
                    f"- **title**: {md.get('title','')}  \n"
                    f"- **uri**: {md.get('uri','')}  \n"
                )
                st.write(r["text"])
                st.markdown("---")

