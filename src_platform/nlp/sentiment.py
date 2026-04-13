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

