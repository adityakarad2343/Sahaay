"""Utility helpers for emotion detection and response generation."""

from textblob import TextBlob


def detect_emotion(text: str) -> str:
    """Convert TextBlob polarity into one of the project emotion labels."""
    try:
        polarity = TextBlob(text).sentiment.polarity
    except Exception:
        # If TextBlob fails for any reason, we fall back to a safe default.
        polarity = 0.0

    if polarity < -0.3:
        return "distressed"
    if polarity < 0:
        return "low"
    if polarity < 0.3:
        return "neutral"
    return "positive"


def generate_response(emotion: str, text: str) -> str:
    """Return a simple rule-based assistant reply based on detected emotion."""
    cleaned_text = text.strip()

    if emotion == "distressed":
        return (
            "I am really sorry you are going through this. "
            "You do not have to handle it alone, and I am here to listen. "
            "Would you like to talk about what feels hardest right now?"
        )
    if emotion == "low":
        return (
            "That sounds difficult, and it makes sense that you are feeling this way. "
            "Let us take it one step at a time. "
            "What would feel most helpful for you right now?"
        )
    if emotion == "positive":
        return (
            "That is wonderful to hear. "
            "I am glad things are feeling better, and I would love to help you keep that momentum going."
        )

    if cleaned_text.endswith("?"):
        return "I am here with you. Tell me a little more, and I will do my best to help."

    return "I am listening. Please go on whenever you are ready."
