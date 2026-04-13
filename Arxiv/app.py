"""
ArXiv Expert Chatbot — Streamlit Application
"""

from __future__ import annotations

import logging
import time

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="ArXiv Expert Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports after page config ─────────────────────────────────────────────
from src.config import (
    APP_TITLE,
    CS_SUBCATEGORIES,
    LLM_BACKEND,
    OLLAMA_MODEL,
    GROQ_MODEL,
    HF_MODEL,
    GROQ_API_KEY,
    HF_API_KEY,
)
from src.data_loader import (
    fetch_papers_from_api,
    fetch_paper_by_id,
    load_papers,
    papers_to_dataframe,
    KAGGLE_SNAPSHOT,
)
from src.embeddings import get_store
from src.llm_handler import get_llm, list_ollama_models
from src.rag_pipeline import RAGPipeline, extract_arxiv_id
from src.visualizations import (
    make_word_cloud,
    make_timeline_chart,
    make_category_chart,
    make_concept_network,
    make_topic_scatter,
    make_relevance_bar,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
:root {
    --app-bg: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 60%, #dbeafe 100%);
    --side-bg: linear-gradient(180deg, #e2e8f0 0%, #cbd5e1 100%);
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --card-bg: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    --card-border: #cbd5e1;
    --input-bg: #ffffff;
    --input-border: #94a3b8;
    --tab-bg: #e2e8f0;
    --tab-active: #4f46e5;
}
html[data-theme="dark"] {
    --app-bg: linear-gradient(135deg, #020817 0%, #0f172a 50%, #0d1f3c 100%);
    --side-bg: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    --text-primary: #e2e8f0;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --card-bg: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    --card-border: #334155;
    --input-bg: #1e293b;
    --input-border: #334155;
    --tab-bg: #0f172a;
    --tab-active: #6366f1;
}
[data-testid="stAppViewContainer"] {
    background: var(--app-bg);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: var(--side-bg);
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
.app-header {
    background: linear-gradient(90deg, #1e40af, #7c3aed, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -1px;
    margin-bottom: 0;
}
.app-subtitle { color: var(--text-muted); font-size: 0.95rem; margin-top: -4px; margin-bottom: 24px; }
.status-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.badge-green  { background: #14532d; color: #86efac; }
.badge-yellow { background: #713f12; color: #fde68a; }
.paper-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
}
.paper-card:hover { border-color: #6366f1; }
.paper-title { font-size: 1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
.paper-meta { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 8px; }
.paper-abstract { font-size: 0.85rem; color: var(--text-secondary); line-height: 1.55; }
.paper-score { display: inline-block; background: #1e3a5f; color: #60a5fa; border-radius: 6px; padding: 1px 8px; font-size: 0.75rem; font-weight: 600; }
.metric-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 10px; padding: 16px; text-align: center; }
.metric-value { font-size: 1.8rem; font-weight: 800; color: #6366f1; }
.metric-label { font-size: 0.8rem; color: var(--text-muted); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 4px; background: var(--tab-bg); padding: 4px; border-radius: 8px; border: 1px solid var(--card-border); }
[data-testid="stTabs"] [data-baseweb="tab"] { color: var(--text-muted) !important; border-radius: 6px !important; padding: 6px 16px !important; }
[data-testid="stTabs"] [aria-selected="true"] { background: var(--tab-active) !important; color: #fff !important; }
.stButton > button { background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { background: var(--input-bg) !important; border: 1px solid var(--input-border) !important; color: var(--text-primary) !important; border-radius: 8px !important; }
hr { border-color: var(--card-border) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────

def init_session() -> None:
    defaults = {
        "messages": [],
        "rag": None,
        "store_ready": False,
        "papers_loaded": 0,
        "llm_backend": LLM_BACKEND,
        "llm_model": OLLAMA_MODEL,
        "selected_category": "cs",
        "search_results": [],
        "last_retrieved": [],
        "init_done": False,
        "init_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Knowledge base init ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def initialise_knowledge_base(category: str, backend: str, model: str):
    try:
        store = get_store()
        if not store.load():
            papers = load_papers(category=category)
            if papers:
                store.build_index(papers)
        llm = get_llm(backend=backend, model=model)
        rag = RAGPipeline(store=store, llm=llm)
        return rag, store.paper_count, None
    except Exception as exc:
        logger.exception("Init error")
        return None, 0, str(exc)


# ── Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.markdown("## 🔬 ArXiv Expert")
        st.markdown("---")
        st.markdown("### ⚙️ LLM Settings")

        backend = st.selectbox(
            "Backend",
            ["ollama", "groq", "huggingface"],
            index=["ollama", "groq", "huggingface"].index(st.session_state.llm_backend),
            key="sb_backend",
        )
        st.session_state.llm_backend = backend

        if backend == "ollama":
            local_models = list_ollama_models()
            if local_models:
                model = st.selectbox("Model", local_models, key="sb_model_ol")
            else:
                model = st.text_input("Model name", OLLAMA_MODEL, key="sb_model_ol_txt")
                st.info("Run: `ollama pull llama3.2`")
        elif backend == "groq":
            groq_models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
            model = st.selectbox("Model", groq_models, key="sb_model_gr")
            if not GROQ_API_KEY:
                st.warning("Set GROQ_API_KEY in your .env")
        else:
            hf_models = [
                "meta-llama/Llama-3.3-70B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.1",
                "HuggingFaceH4/zephyr-7b-beta",
                "tiiuae/falcon-7b-instruct",
                "google/flan-ul2",
                "bigscience/bloomz-7b1",
            ]
            model = st.selectbox("Model", hf_models, index=0, key="sb_model_hf")
            st.caption("💡 Llama 3 models are gated — accept the license on HF first!")
            if not HF_API_KEY:
                st.warning("Set HF_API_KEY in your .env")

        st.session_state.llm_model = model
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")

        cat_label = st.selectbox("CS Subcategory", list(CS_SUBCATEGORIES.keys()), key="sb_cat")
        category = CS_SUBCATEGORIES[cat_label]
        st.session_state.selected_category = category

        if KAGGLE_SNAPSHOT.exists():
            st.markdown('<span class="status-badge badge-green">✓ Kaggle dataset found</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-yellow">⚡ Using arXiv API</span>', unsafe_allow_html=True)

        st.markdown("---")
        if st.session_state.store_ready:
            st.metric("Papers indexed", f"{st.session_state.papers_loaded:,}")
            st.metric("Backend", backend.capitalize())

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag:
                st.session_state.rag.clear_history()
            st.rerun()

        if st.button("🔄 Rebuild Index", use_container_width=True):
            initialise_knowledge_base.clear()
            st.session_state.init_done = False
            st.session_state.store_ready = False
            st.rerun()

        st.markdown("---")
        st.markdown("<small style='color:#475569'>Streamlit · FAISS · sentence-transformers</small>", unsafe_allow_html=True)

    return backend, model, category


# ── Paper card ────────────────────────────────────────────────────────────

def render_paper_card(paper: dict, show_score: bool = False) -> None:
    authors = paper.get("authors", [])
    author_str = (
        f"{authors[0]} et al." if len(authors) > 1
        else authors[0] if authors else "Unknown"
    )
    year = paper.get("published", "")[:4] or "?"
    cats = ", ".join(paper.get("categories", [])[:3])
    abstract = paper.get("abstract", "")
    abstract_short = abstract[:300] + "…" if len(abstract) > 300 else abstract
    score_html = ""
    if show_score and "relevance_score" in paper:
        score_html = f'<span class="paper-score">Score: {paper["relevance_score"]:.3f}</span>&nbsp;'

    st.markdown(
        f"""<div class="paper-card">
  <div class="paper-title">{paper.get('title', 'Untitled')}</div>
  <div class="paper-meta">{author_str} · {year} · {cats}</div>
  <div class="paper-abstract">{abstract_short}</div>
  <div style="margin-top:10px">
    {score_html}
    <a href="{paper.get('url','#')}" target="_blank" style="color:#60a5fa;font-size:0.8rem;text-decoration:none;">🔗 arXiv</a>
    &nbsp;·&nbsp;
    <a href="{paper.get('pdf_url','#')}" target="_blank" style="color:#f472b6;font-size:0.8rem;text-decoration:none;">📄 PDF</a>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Tab: Chat ─────────────────────────────────────────────────────────────

def tab_chat(rag: RAGPipeline) -> None:
    st.markdown("### 💬 Research Assistant Chat")
    st.markdown(
        "<small style='color:#64748b'>Ask about concepts, request paper summaries, or explore any CS topic. "
        "Paste an arXiv URL to summarise a specific paper.</small>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("papers"):
                with st.expander(f"📎 {len(msg['papers'])} source papers"):
                    for p in msg["papers"]:
                        render_paper_card(p, show_score=True)

    if prompt := st.chat_input("Ask a research question or paste an arXiv URL…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        arxiv_id = extract_arxiv_id(prompt)
        task = "chat"
        extra_papers: list[dict] = []

        if arxiv_id:
            with st.spinner(f"Fetching paper {arxiv_id}…"):
                paper = fetch_paper_by_id(arxiv_id)
            if paper:
                extra_papers = [paper]
                task = "summarize"
                st.toast(f"📄 Found: {paper['title'][:60]}…", icon="✅")
        elif any(kw in prompt.lower() for kw in ["explain", "what is", "define", "describe", "how does", "concept of"]):
            task = "explain"

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            retrieved: list[dict] = []

            try:
                gen = rag.stream_answer(prompt, task=task, extra_papers=extra_papers)
                for chunk in gen:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                retrieved = getattr(rag, "_last_papers", [])
            except RuntimeError as exc:
                full_response = f"⚠️ **LLM Error:** {exc}\n\nCheck your LLM settings in the sidebar."
                placeholder.error(full_response)
            except Exception as exc:
                full_response = f"⚠️ Error: {exc}"
                placeholder.error(full_response)

            if retrieved:
                with st.expander(f"📎 {len(retrieved)} source papers used"):
                    for p in retrieved:
                        render_paper_card(p, show_score=True)

        st.session_state.messages.append({"role": "assistant", "content": full_response, "papers": retrieved})
        st.session_state.last_retrieved = retrieved

    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "Explain the transformer architecture in detail",
            "What are the latest advances in diffusion models?",
            "Summarize recent work on LLM alignment",
            "How does RLHF work in language models?",
            "What is retrieval-augmented generation?",
            "Explain graph neural networks with examples",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(f"💬 {sug}", key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": sug})
                st.rerun()


# ── Tab: Search ───────────────────────────────────────────────────────────

def tab_search(rag: RAGPipeline) -> None:
    st.markdown("### 🔍 Paper Search")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search query", placeholder="e.g. attention mechanism, federated learning…", key="sq")
    with col2:
        top_k = st.slider("Results", 3, 20, 8, key="sk")

    c3, c4 = st.columns(2)
    use_semantic = c3.checkbox("Semantic search (FAISS)", value=True)
    use_live = c4.checkbox("Include live arXiv results", value=False)

    if st.button("🔍 Search", key="search_btn"):
        if not query.strip():
            st.warning("Enter a search query.")
            return

        results: list[dict] = []
        if use_semantic:
            with st.spinner("Searching indexed papers…"):
                results = rag.retrieve(query, top_k=top_k)
        if use_live:
            with st.spinner("Fetching live results from arXiv…"):
                live = fetch_papers_from_api(f"cat:cs.* AND ({query})", max_results=top_k)
                existing_ids = {p["id"] for p in results}
                for p in live:
                    if p["id"] not in existing_ids:
                        results.append(p)
                added = rag.store.add_papers(live)
                if added:
                    st.toast(f"Added {added} new papers to index", icon="📥")

        st.session_state.search_results = results
        if results:
            st.success(f"Found **{len(results)}** papers")
            if results and "relevance_score" in results[0]:
                st.plotly_chart(make_relevance_bar(results), use_container_width=True)
            for paper in results:
                render_paper_card(paper, show_score=True)
        else:
            st.info("No results. Try enabling live arXiv results.")

    st.markdown("---")
    st.markdown("#### 📄 Summarise a Specific Paper")
    paper_url = st.text_input("Paste arXiv URL or ID", placeholder="https://arxiv.org/abs/2310.06825", key="purl")
    if st.button("📝 Summarise", key="sum_btn"):
        arxiv_id = extract_arxiv_id(paper_url) if paper_url else None
        if not arxiv_id:
            st.error("Could not extract a valid arXiv ID.")
            return
        with st.spinner(f"Fetching {arxiv_id}…"):
            paper = fetch_paper_by_id(arxiv_id)
        if not paper:
            st.error(f"Paper {arxiv_id} not found.")
            return
        render_paper_card(paper)
        with st.spinner("Generating summary…"):
            summary, _ = rag.summarize_paper(paper)
        st.markdown("#### 📋 AI Summary")
        st.markdown(summary)


# ── Tab: Explore ──────────────────────────────────────────────────────────

def tab_explore(papers: list[dict]) -> None:
    st.markdown("### 📊 Concept Explorer")
    if not papers:
        st.info("No papers loaded yet.")
        return

    st.info(f"Analysing **{len(papers):,}** indexed papers.")
    viz_type = st.selectbox(
        "Visualisation",
        ["Word Cloud", "Concept Co-occurrence Network", "Topic Clusters (LDA + t-SNE)", "Publication Timeline", "Category Distribution"],
        key="viz_type",
    )
    sample = papers[:2000] if len(papers) > 2000 else papers

    if viz_type == "Word Cloud":
        field = st.radio("Field", ["abstract", "title"], horizontal=True)
        with st.spinner("Generating word cloud…"):
            img = make_word_cloud(sample, field=field)
        if img:
            st.image(img, use_container_width=True)
        else:
            st.error("Install wordcloud: `pip install wordcloud`")

    elif viz_type == "Concept Co-occurrence Network":
        c1, c2 = st.columns(2)
        n_kw = c1.slider("Keywords", 15, 60, 30)
        min_co = c2.slider("Min co-occurrences", 1, 10, 2)
        with st.spinner("Building network…"):
            fig = make_concept_network(sample, top_keywords=n_kw, min_cooccurrence=min_co)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Topic Clusters (LDA + t-SNE)":
        n_topics = st.slider("Topics", 3, 12, 6)
        with st.spinner("Running topic modelling (30–60s)…"):
            fig = make_topic_scatter(sample, n_topics=n_topics)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Publication Timeline":
        st.plotly_chart(make_timeline_chart(papers), use_container_width=True)

    else:
        st.plotly_chart(make_category_chart(papers), use_container_width=True)


# ── Tab: Analytics ────────────────────────────────────────────────────────

def tab_analytics(papers: list[dict]) -> None:
    import plotly.express as px

    st.markdown("### 📈 Dataset Analytics")
    if not papers:
        st.info("No papers loaded yet.")
        return

    df = papers_to_dataframe(papers)

    cols = st.columns(4)
    cols[0].markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Papers</div></div>', unsafe_allow_html=True)
    cols[1].markdown(f'<div class="metric-card"><div class="metric-value">{df["primary_category"].nunique() if "primary_category" in df else "—"}</div><div class="metric-label">Categories</div></div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="metric-card"><div class="metric-value">{df["first_author"].nunique() if "first_author" in df else "—"}</div><div class="metric-label">Unique Authors</div></div>', unsafe_allow_html=True)

    date_range = "—"
    if "published" in df.columns:
        valid = df["published"].dropna()
        if not valid.empty:
            date_range = f"{valid.min().year}–{valid.max().year}"
    cols[3].markdown(f'<div class="metric-card"><div class="metric-value">{date_range}</div><div class="metric-label">Date Range</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_timeline_chart(papers), use_container_width=True)
    with c2:
        st.plotly_chart(make_category_chart(papers, top_n=12), use_container_width=True)

    st.markdown("#### 👥 Most Prolific Authors (First Author)")
    if "first_author" in df.columns:
        top_authors = df["first_author"].value_counts().head(15).reset_index()
        top_authors.columns = ["Author", "Papers"]
        fig = px.bar(top_authors, x="Papers", y="Author", orientation="h", color="Papers",
                     color_continuous_scale="Cividis", title="Top 15 First Authors")
        fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                          yaxis=dict(autorange="reversed", gridcolor="#1e293b"),
                          xaxis=dict(gridcolor="#1e293b"), coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🗃️ Browse raw data"):
        disp_cols = [c for c in ["title", "first_author", "published", "primary_category"] if c in df.columns]
        st.dataframe(df[disp_cols].sort_values("published", ascending=False).head(200), use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    init_session()

    st.markdown(
        f'<h1 class="app-header">🔬 {APP_TITLE}</h1>'
        '<p class="app-subtitle">Expert AI assistant for scientific literature · RAG · open-source LLMs · arXiv</p>',
        unsafe_allow_html=True,
    )

    backend, model, category = render_sidebar()

    if not st.session_state.init_done:
        with st.spinner("⚙️ Initialising knowledge base…"):
            rag, paper_count, error = initialise_knowledge_base(category, backend, model)
        if error:
            st.session_state.init_error = error
            st.session_state.store_ready = False
        else:
            st.session_state.rag = rag
            st.session_state.papers_loaded = paper_count
            st.session_state.store_ready = True
            st.session_state.init_error = None
        st.session_state.init_done = True

    if st.session_state.init_error:
        st.error(
            f"**Init error:** {st.session_state.init_error}\n\n"
            "Check your LLM config in the sidebar and `.env` file."
        )

    if st.session_state.store_ready:
        count = st.session_state.papers_loaded
        rag: RAGPipeline = st.session_state.rag
        papers = rag.store.all_papers

        st.markdown(
            f'<div style="background:#0f2027;border:1px solid #1e40af;border-radius:8px;'
            f'padding:8px 16px;margin-bottom:16px;font-size:0.85rem;color:#93c5fd;">'
            f'✅ Knowledge base ready · <b>{count:,} papers</b> indexed · LLM: <b>{backend}/{model}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔍 Search Papers", "📊 Explore Concepts", "📈 Analytics"])
        with tab1:
            tab_chat(rag)
        with tab2:
            tab_search(rag)
        with tab3:
            tab_explore(papers)
        with tab4:
            tab_analytics(papers)
    else:
        if not st.session_state.init_error:
            st.info("⏳ Initialising…")


if __name__ == "__main__":
    main()
