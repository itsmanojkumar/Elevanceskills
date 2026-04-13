"""
MedQuAD Medical Q&A Chatbot — Streamlit Application.

Retrieval pipeline (fully local, zero API cost):
  Primary  : sentence-transformers (local) + FAISS vector store via LangChain
  Re-rank  : cross-encoder/ms-marco-MiniLM-L-6-v2

NER: spaCy en_core_web_sm + medical dictionary EntityRuler
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure local src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.retriever import MedicalRetriever
from src.medical_ner import get_ner, ENTITY_COLORS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedQuAD Medical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --text-primary: #1e293b;
    --text-muted: #64748b;
    --surface-soft: #f8fafc;
    --surface-border: #e2e8f0;
    --tag-bg: #dbeafe;
    --tag-text: #1e40af;
    --disclaimer-bg: #fff7ed;
    --disclaimer-text: #7c2d12;
}

html[data-theme="dark"] {
    --text-primary: #e2e8f0;
    --text-muted: #94a3b8;
    --surface-soft: #111827;
    --surface-border: #334155;
    --tag-bg: #1e3a8a;
    --tag-text: #bfdbfe;
    --disclaimer-bg: #3f1d0f;
    --disclaimer-text: #fed7aa;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
}
section[data-testid="stSidebar"] * { color: #e0f2fe !important; }
section[data-testid="stSidebar"] a { color: #7dd3fc !important; }

/* Entity highlight styling */
mark {
    border-radius: 4px;
    padding: 1px 3px;
    margin: 0 1px;
}
.ner-mark {
    color: var(--text-primary) !important;
    border: 1px solid var(--surface-border);
}

.meta-tag {
    display: inline-block;
    background: var(--tag-bg);
    color: var(--tag-text);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 4px 4px 0 0;
}
.entity-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    margin: 2px 3px;
    color: #1e293b;
}
.score-bar-wrap {
    background: #e2e8f0;
    border-radius: 999px;
    height: 6px;
    margin: 6px 0 10px;
    width: 100%;
}
.badge-faiss {
    display:inline-block;background:#d1fae5;color:#065f46;
    border-radius:999px;padding:2px 10px;font-size:0.7rem;font-weight:700;margin-left:8px;
}
.main-header { text-align: center; padding: 20px 0 10px; }
.main-header h1 {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.main-header p { color: #64748b; font-size: 0.95rem; }
.disclaimer {
    background: var(--disclaimer-bg); border-left: 4px solid #f97316;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.82rem; color: var(--disclaimer-text); margin-top: 10px;
}
.related-q {
    background: var(--surface-soft); border: 1px solid var(--surface-border);
    border-radius: 10px; padding: 10px 14px; margin: 6px 0;
    font-size: 0.85rem; color: var(--text-primary);
}
.answer-text {
    line-height: 1.6;
    color: var(--text-primary);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages"    not in st.session_state: st.session_state.messages    = []
if "index_ready" not in st.session_state: st.session_state.index_ready = False

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_retriever() -> MedicalRetriever:
    r = MedicalRetriever()
    if r.load_index():
        return r
    raise RuntimeError("FAISS index not found. Run `python setup.py` first.")


@st.cache_resource(show_spinner=False)
def get_ner_engine():
    return get_ner()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏥 MedQuAD Chatbot")
    st.markdown("---")

    st.markdown("### System Status")
    retriever_obj = None
    ner_obj       = None

    with st.spinner("Loading retrieval index..."):
        try:
            retriever_obj = get_retriever()
            st.session_state.index_ready = True
            st.success(
                f"✅ FAISS ready\n\n"
                f"🤗 `{retriever_obj.model_name.split('/')[-1]}`\n\n"
                f"Local embeddings"
            )
        except Exception as e:
            st.error("❌ FAISS index not ready.")
            st.code("python setup.py")
            st.caption(str(e))

    with st.spinner("Loading NER..."):
        try:
            ner_obj = get_ner_engine()
            st.success("✅ Medical NER ready")
        except Exception as e:
            st.warning(f"⚠️ NER unavailable: {e}")

    st.markdown("---")

    # Settings
    st.markdown("### Settings")
    top_k          = st.slider("Results to return", 1, 8, 3)
    show_entities  = st.toggle("Highlight medical entities", value=True)
    show_related   = st.toggle("Show related Q&As", value=True)
    show_scores    = st.toggle("Show similarity scores", value=False)
    conf_threshold = st.slider("Min confidence", 0.0, 1.0, 0.05, 0.01)

    st.markdown("---")

    # Entity legend
    st.markdown("### Entity Legend")
    for lbl, desc in {
        "DISEASE":    "Disease",
        "SYMPTOM":    "Symptom",
        "TREATMENT":  "Treatment",
        "MEDICATION": "Medication",
        "ANATOMY":    "Anatomy",
        "PROCEDURE":  "Procedure",
    }.items():
        color = ENTITY_COLORS.get(lbl, "#ADB5BD")
        st.markdown(
            f'<span class="entity-pill" style="background:{color};">{lbl}</span> {desc}',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Sample Questions")
    for sq in [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes asthma?",
        "What are side effects of ibuprofen?",
        "I feel tired all the time",
    ]:
        if st.button(sq, key=f"sq_{sq}", use_container_width=True):
            st.session_state["prefill_query"] = sq
            st.rerun()

    st.markdown("---")
    st.caption("Data: MedQuAD (NIH/NCI/NLM)")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
  <h1>🏥 MedQuAD Medical Q&A Chatbot</h1>
  <p>Local HuggingFace Embeddings + FAISS · Medical NER · Zero API cost</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Medical Disclaimer:</strong> This chatbot provides information from the MedQuAD dataset
  for <em>educational purposes only</em>. It is <strong>not</strong> a substitute for professional
  medical advice. Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------
def _color(s: float) -> str:
    return "#10b981" if s >= 0.6 else "#f59e0b" if s >= 0.3 else "#ef4444"

def _label(s: float) -> str:
    return "High" if s >= 0.6 else "Medium" if s >= 0.3 else "Low"


def build_response_html(query, result, ner, show_ents, show_rel, show_sc) -> str:
    if not result["found"]:
        return f'<div style="color:#ef4444;font-weight:600;margin-bottom:8px;">🤖 No relevant answer found</div><div class="answer-text">{result["message"]}</div>'

    # Clean the answer text to avoid Markdown triggering
    # Remove leading/trailing whitespace and normalize internal newlines to <br>
    answer_raw = result["best_answer"].strip()
    # Replace multiple spaces at start of lines and newlines with <br>
    answer_disp = "<br>".join([line.lstrip() for line in answer_raw.split("\n")])
    ent_html    = ""

    if show_ents and ner:
        # Highlight entities with <mark> tags
        answer_disp = ner.highlight_text(answer_raw)
        
        # Summary pills
        merged: dict = {}
        for src in (query, answer_raw):
            for lbl, txts in ner.get_entity_summary(src).items():
                merged.setdefault(lbl, [])
                for t in txts:
                    if t not in merged[lbl]:
                        merged[lbl].append(t)
        if merged:
            pills = "".join(
                f'<span class="entity-pill" style="background:{ENTITY_COLORS.get(lbl,"#ADB5BD")};">{t}</span>'
                for lbl, txts in merged.items() for t in txts[:5]
            )
            ent_html = (
                '<div style="margin-top:10px; border-top: 1px solid #e2e8f0; padding-top: 8px;">'
                '<strong style="font-size:0.8rem;color:#64748b;">Detected entities:</strong>'
                f'<br>{pills}</div>'
            )

    conf       = result.get("confidence", 0.0)
    bar_pct    = min(int(conf / 0.7 * 100), 100)
    conf_color = _color(conf)
    badge_lbl  = "🧬 FAISS"

    meta = ""
    for k, icon in [("focus", "🎯"), ("question_type", "❓"), ("source", "📚")]:
        if result.get(k):
            meta += f'<span class="meta-tag">{icon} {result[k]}</span>'
    
    if result.get("url"):
        meta += f'<span class="meta-tag"><a href="{result["url"]}" target="_blank" style="color:#1e40af;text-decoration:none;">🔗 Source</a></span>'

    sc_html = ""
    if show_sc:
        sc_html = f'<div style="font-size:0.75rem;color:#64748b;margin-top:4px;">Similarity score: <strong>{conf:.4f}</strong></div>'

    rel_html = ""
    if show_rel and result.get("related"):
        rel_html = '<div style="margin-top:14px;"><strong style="font-size:0.82rem;color:#475569;">Related Q&As:</strong>'
        for rel in result["related"][:2]:
            short = rel["answer"][:180].strip() + ("…" if len(rel["answer"]) > 180 else "")
            rel_html += (
                f'<div class="related-q">'
                f'<strong style="color:#1e40af;">{rel.get("focus","")}</strong><br>'
                f'<span style="font-size:0.85rem;">{rel["question"]}</span><br>'
                f'<span style="color:#64748b;font-size:0.8rem;">{short}</span>'
                f'</div>'
            )
        rel_html += "</div>"

    # Final HTML string construction (no leading whitespace/tabs)
    # Compact HTML to avoid markdown triggering code blocks
    html_out = (
        f'<div style="margin-bottom:6px;">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
        f'<strong style="font-size:0.95rem;color:#1e40af;">MedQuAD Local Intelligence</strong>'
        f'<span class="badge-faiss">{badge_lbl}</span>'
        f'<span style="font-size:0.78rem;color:{conf_color};font-weight:600;margin-left:auto;">'
        f'{_label(conf)} ({conf:.3f})'
        f'</span>'
        f'</div>'
        f'<div class="score-bar-wrap">'
        f'<div style="height:6px;border-radius:999px;width:{bar_pct}%;background:{conf_color};"></div>'
    )
    
    # Avoid newline before closing div of score-bar-wrap
    html_out += '</div>'
    
    if sc_html:
        html_out += sc_html
        
    html_out += f'<div style="margin-bottom:10px;">{meta}</div>'
    html_out += f'<div class="answer-text">{answer_disp}</div>'
    
    if ent_html:
        html_out += ent_html
    
    if rel_html:
        html_out += rel_html
        
    html_out += '</div>'
    
    # Replace all \n with <br> and strip to be 100% sure no markdown code block is triggered
    return html_out.replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------
# Initial welcome message
if not st.session_state.messages:
    welcome_text = (
        "Welcome! I am a specialized medical chatbot. I can answer questions about diseases, "
        "symptoms, treatments, and more using the MedQuAD dataset (47k Q&A pairs).\n\n"
        "**Everything runs locally** using biomedical BERT embeddings and FAISS search."
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_text})

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
# Handle prefilled queries from sidebar
query = st.chat_input("Ask a medical question...")
if "prefill_query" in st.session_state:
    query = st.session_state.pop("prefill_query")

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate response
    with st.chat_message("assistant"):
        if not st.session_state.index_ready:
            st.error("⏳ System still initializing. Please wait a moment.")
        else:
            with st.spinner("Analyzing MedQuAD dataset..."):
                try:
                    retriever = get_retriever()
                    result = retriever.get_answer(query, top_k=top_k)
                    
                    # Threshold check
                    if result["found"] and result["confidence"] < conf_threshold:
                         response_html = (
                            f"Best match confidence ({result['confidence']:.3f}) is below your "
                            f"threshold ({conf_threshold:.2f}). Try lowering the threshold in the sidebar."
                         )
                    else:
                        response_html = build_response_html(
                            query, result, ner_obj,
                            show_entities, show_related, show_scores,
                        )
                    
                    st.markdown(response_html.strip(), unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response_html.strip()})
                except Exception as e:
                    st.error(f"Error during retrieval: {e}")

    # No rerun needed with chat_input, it triggers refresh automatically
