from __future__ import annotations

import streamlit as st

from src_platform.config import Settings
from src_platform.nlp.sentiment import SentimentAnalyzer


def render_sentiment_app(settings: Settings) -> None:
    _ = settings
    st.header("Task 3 — Sentiment Analysis")

    st.markdown(
        "Detects **positive / neutral / negative** sentiment locally (VADER).\n\n"
        "Use this as a drop-in signal to adapt responses (tone/pacing)."
    )

    text = st.text_area("Message to analyze", height=120, placeholder="Type a message…")
    if st.button("Analyze sentiment", use_container_width=True):
        analyzer = SentimentAnalyzer()
        r = analyzer.analyze(text)
        st.write(f"**label**: `{r.label}`")
        st.write(f"**compound**: `{r.compound:.3f}`")
        st.json(r.details)

