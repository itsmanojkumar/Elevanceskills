"""
Visualization utilities: word clouds, topic networks, publication timelines,
category distributions, and concept graphs.
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Common stopwords for academic text ────────────────────────────────────

_STOP = frozenset(
    """a an the and or but in on at to for of with is are was were be been
    being have has had do does did will would could should may might shall
    this that these those it its we our they their we i our also such via
    both from as by using through based approach proposed method paper
    results show work using model models learning data training deep neural
    network networks proposed new further however also can using used
    presents present study studies analysis show shows recent previous
    state art research we propose present first two three multiple several""".split()
)


def _clean_text(texts: list[str]) -> list[str]:
    """Lowercase, remove punctuation, filter stopwords."""
    cleaned = []
    for t in texts:
        t = re.sub(r"[^a-zA-Z\s]", " ", t.lower())
        words = [w for w in t.split() if w not in _STOP and len(w) > 3]
        cleaned.append(" ".join(words))
    return cleaned


# ── Word Cloud ────────────────────────────────────────────────────────────

def make_word_cloud(
    papers: list[dict],
    field: str = "abstract",
    width: int = 800,
    height: int = 400,
) -> Optional[bytes]:
    """Return a PNG word cloud as bytes, or None if wordcloud not installed."""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("wordcloud / matplotlib not installed.")
        return None

    texts = [p.get(field, "") for p in papers if p.get(field)]
    combined = " ".join(_clean_text(texts))
    if not combined.strip():
        return None

    wc = WordCloud(
        width=width,
        height=height,
        background_color="#0f172a",
        colormap="cool",
        max_words=120,
        collocations=False,
        prefer_horizontal=0.7,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor="#0f172a")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Publication Timeline ──────────────────────────────────────────────────

def make_timeline_chart(papers: list[dict]) -> go.Figure:
    """Plotly bar chart: papers published per month."""
    rows = []
    for p in papers:
        date_str = p.get("published", "")
        if len(date_str) >= 7:
            rows.append(date_str[:7])

    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No date data available", showarrow=False)
        return fig

    counts = Counter(rows)
    months = sorted(counts.keys())
    values = [counts[m] for m in months]

    fig = px.bar(
        x=months,
        y=values,
        labels={"x": "Month", "y": "Papers"},
        title="📅 Publication Timeline",
        color=values,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        title_font_size=16,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    return fig


# ── Category Distribution ─────────────────────────────────────────────────

def make_category_chart(papers: list[dict], top_n: int = 15) -> go.Figure:
    """Plotly horizontal bar of top arXiv categories."""
    all_cats: list[str] = []
    for p in papers:
        cats = p.get("categories", [])
        if isinstance(cats, list):
            all_cats.extend(cats)
        elif isinstance(cats, str):
            all_cats.extend(cats.split())

    if not all_cats:
        fig = go.Figure()
        fig.add_annotation(text="No category data", showarrow=False)
        return fig

    counts = Counter(all_cats).most_common(top_n)
    cats, vals = zip(*counts)

    fig = px.bar(
        x=list(vals),
        y=list(cats),
        orientation="h",
        labels={"x": "Papers", "y": "Category"},
        title="📂 Top ArXiv Categories",
        color=list(vals),
        color_continuous_scale="Plasma",
    )
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        yaxis=dict(autorange="reversed", gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        coloraxis_showscale=False,
        title_font_size=16,
        height=max(350, top_n * 28),
    )
    return fig


# ── Keyword Co-occurrence Network ─────────────────────────────────────────

def _extract_keywords_tfidf(texts: list[str], top_n: int = 40) -> list[str]:
    """Extract top TF-IDF keywords from a list of texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    cleaned = _clean_text(texts)
    try:
        vec = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2))
        vec.fit(cleaned)
        return list(vec.get_feature_names_out())
    except Exception:
        return []


def make_concept_network(
    papers: list[dict],
    top_keywords: int = 30,
    min_cooccurrence: int = 2,
) -> go.Figure:
    """
    Build a keyword co-occurrence network from paper abstracts.
    Nodes = keywords, edges = co-occurrence in same abstract.
    """
    import networkx as nx

    texts = [p.get("abstract", "") for p in papers if p.get("abstract")]
    if len(texts) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 3 papers for a concept network", showarrow=False)
        return fig

    keywords = _extract_keywords_tfidf(texts, top_n=top_keywords)
    if not keywords:
        fig = go.Figure()
        fig.add_annotation(text="Could not extract keywords", showarrow=False)
        return fig

    kw_set = set(keywords)
    G = nx.Graph()
    G.add_nodes_from(keywords)

    for text in texts:
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        present = [kw for kw in kw_set if kw in cleaned]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_cooccurrence
    ]
    G.remove_edges_from(edges_to_remove)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    if len(G.nodes()) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough co-occurrences for network", showarrow=False)
        return fig

    pos = nx.spring_layout(G, seed=42, k=2 / (len(G.nodes()) ** 0.5))

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_sizes = [10 + 25 * (degrees[n] / max_deg) for n in G.nodes()]
    node_labels = list(G.nodes())

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#334155"),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        marker=dict(
            size=node_sizes,
            color=node_sizes,
            colorscale="Viridis",
            showscale=False,
            line=dict(width=1, color="#1e293b"),
        ),
        hovertemplate="<b>%{text}</b><br>Connections: %{marker.size:.0f}<extra></extra>",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="🕸️ Concept Co-occurrence Network",
            title_font_size=16,
            showlegend=False,
            hovermode="closest",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=520,
            margin=dict(t=50, b=20, l=20, r=20),
        ),
    )
    return fig


# ── LDA Topic Modeling ────────────────────────────────────────────────────

def make_topic_scatter(
    papers: list[dict],
    n_topics: int = 6,
    n_words_per_topic: int = 8,
) -> go.Figure:
    """
    LDA topic modeling on abstracts, visualised as a scatter via t-SNE.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.manifold import TSNE

    texts = [p.get("abstract", "") for p in papers if p.get("abstract")]
    if len(texts) < max(20, n_topics * 3):
        fig = go.Figure()
        fig.add_annotation(
            text=f"Need at least {max(20, n_topics * 3)} papers for topic modeling",
            showarrow=False,
            font=dict(color="#e2e8f0"),
        )
        fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
        return fig

    cleaned = _clean_text(texts)
    n_topics = min(n_topics, len(texts) // 3)

    vec = CountVectorizer(max_features=2000, min_df=2)
    dtm = vec.fit_transform(cleaned)

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=20
    )
    topic_dist = lda.fit_transform(dtm)
    doc_topics = topic_dist.argmax(axis=1)

    feature_names = vec.get_feature_names_out()
    topic_labels = []
    for i, comp in enumerate(lda.components_):
        top_words = [feature_names[j] for j in comp.argsort()[-n_words_per_topic:][::-1]]
        topic_labels.append(f"T{i}: {', '.join(top_words[:4])}")

    perplexity = min(30, len(texts) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(topic_dist)

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "topic": [topic_labels[t] for t in doc_topics],
            "title": [p.get("title", "")[:60] + "…" for p in papers if p.get("abstract")],
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="topic",
        hover_data={"title": True, "x": False, "y": False},
        title="🗂️ Topic Clusters (LDA + t-SNE)",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        title_font_size=16,
        legend=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(size=9),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    return fig


# ── Relevance bar ─────────────────────────────────────────────────────────

def make_relevance_bar(papers: list[dict]) -> go.Figure:
    """Horizontal bar showing relevance scores for retrieved papers."""
    if not papers:
        return go.Figure()

    titles = [p.get("title", "")[:55] + "…" for p in papers]
    scores = [p.get("relevance_score", 0.0) for p in papers]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=titles,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="📊 Retrieval Relevance Scores",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", size=11),
        yaxis=dict(autorange="reversed", gridcolor="#1e293b"),
        xaxis=dict(range=[0, 1.05], gridcolor="#1e293b"),
        height=max(200, len(papers) * 40 + 60),
        margin=dict(l=250, r=80, t=50, b=30),
    )
    return fig
