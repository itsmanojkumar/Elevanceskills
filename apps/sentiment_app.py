from __future__ import annotations

import streamlit as st

from src_platform.config import Settings
from src_platform.nlp.sentiment import SentimentAnalyzer, SentimentResult, build_sentiment_aware_reply


def render_sentiment_app(settings: Settings) -> None:
    _ = settings
    st.header("Task 3 — Sentiment Analysis")

    st.markdown(
        "Detects **positive / neutral / negative** sentiment locally (VADER) and "
        "adapts chatbot behavior to user emotion."
    )

    if "sentiment_chat" not in st.session_state:
        st.session_state.sentiment_chat = []
    if "sentiment_counts" not in st.session_state:
        st.session_state.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    analyzer = SentimentAnalyzer()

    st.subheader("Quick Sentiment Check")
    text = st.text_area("Message to analyze", height=110, placeholder="Type a customer message…")
    if st.button("Analyze sentiment", use_container_width=True):
        r = analyzer.analyze(text)
        st.write(f"**label**: `{r.label}`")
        st.write(f"**compound**: `{r.compound:.3f}`")
        st.json(r.details)

    st.markdown("---")
    st.subheader("Sentiment-Aware Chatbot")
    st.caption(
        "Each user message is scored, then the assistant response style is adapted "
        "(empathetic for negative, concise for neutral, reinforcing for positive)."
    )

    for turn in st.session_state.sentiment_chat:
        role = turn["role"]
        with st.chat_message(role):
            st.markdown(turn["content"])
            if role == "user":
                s: SentimentResult = turn["sentiment"]
                st.caption(f"Detected sentiment: {s.label} ({s.compound:.3f})")

    if prompt := st.chat_input("Type a customer message..."):
        user_sentiment = analyzer.analyze(prompt)
        st.session_state.sentiment_counts[user_sentiment.label] += 1
        user_turn = {
            "role": "user",
            "content": prompt,
            "sentiment": user_sentiment,
        }
        st.session_state.sentiment_chat.append(user_turn)

        # Render the new user message immediately.
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"Detected sentiment: {user_sentiment.label} ({user_sentiment.compound:.3f})")

        reply = build_sentiment_aware_reply(prompt, user_sentiment)
        assistant_turn = {
            "role": "assistant",
            "content": reply,
        }
        st.session_state.sentiment_chat.append(assistant_turn)

        # Render assistant response immediately so users don't miss it.
        with st.chat_message("assistant"):
            st.markdown(reply)

    st.markdown("---")
    st.subheader("Evaluation Snapshot (Session)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", st.session_state.sentiment_counts["positive"])
    c2.metric("Neutral", st.session_state.sentiment_counts["neutral"])
    c3.metric("Negative", st.session_state.sentiment_counts["negative"])

    st.caption(
        "Use these counts with user feedback (thumbs up/down or CSAT) to assess "
        "response appropriateness and customer satisfaction impact."
    )

