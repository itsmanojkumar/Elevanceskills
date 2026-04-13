from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentResult:
    label: str  # "positive" | "neutral" | "negative"
    compound: float
    details: dict


class SentimentAnalyzer:
    """
    Lightweight local sentiment analysis using VADER.
    Good for chat UX signals (not clinical sentiment).
    """

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: vaderSentiment. Install with `pip install vaderSentiment`."
            ) from e

        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        scores = self._analyzer.polarity_scores(text or "")
        c = float(scores.get("compound", 0.0))
        if c >= 0.25:
            label = "positive"
        elif c <= -0.25:
            label = "negative"
        else:
            label = "neutral"
        return SentimentResult(label=label, compound=c, details=scores)


def apply_sentiment_policy(user_text: str, sentiment: SentimentResult) -> str:
    """
    Returns a short system prefix (tone guidance) based on sentiment.
    Keep it small so it doesn't overpower retrieval content.
    """
    _ = user_text
    if sentiment.label == "negative":
        return (
            "Tone: empathetic and calm. Ask one clarifying question if needed. "
            "Keep steps small and actionable.\n"
        )
    if sentiment.label == "positive":
        return "Tone: upbeat and concise. Provide the best next step clearly.\n"
    return "Tone: professional and direct.\n"


def build_sentiment_aware_reply(user_text: str, sentiment: SentimentResult) -> str:
    """
    Build a customer-support style response adapted to detected sentiment.
    This keeps the demo local and deterministic (no external LLM required).
    """
    cleaned = (user_text or "").strip()
    if not cleaned:
        return "Please share your question, and I will help you right away."

    if sentiment.label == "negative":
        if sentiment.compound <= -0.6:
            return (
                "I am really sorry this has been frustrating. I want to help fix this quickly.\n\n"
                "Please share your order/account ID (if applicable) and what happened just before the issue. "
                "I will suggest the fastest recovery steps."
            )
        return (
            "I understand this is inconvenient, and I appreciate you sharing it.\n\n"
            "Let us resolve it together. Tell me the exact error or step where things failed, "
            "and I will guide you with clear next actions."
        )

    if sentiment.label == "positive":
        if sentiment.compound >= 0.6:
            return (
                "That is great to hear, thank you for the positive feedback!\n\n"
                "If you want, I can also share best-practice tips to get even better results."
            )
        return (
            "Glad to hear things are going well.\n\n"
            "If you share your next goal, I can suggest the most efficient next step."
        )

    return (
        "Thanks for your message.\n\n"
        "I can help with this. Please share one more detail about your goal or issue, "
        "and I will provide a precise step-by-step response."
    )

