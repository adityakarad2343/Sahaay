"""Utility helpers for emotion detection, context building, and AI responses."""

import os
import re
from typing import Any, Dict, List

from textblob import TextBlob


def detect_emotion(text: str) -> str:
    """Combine keyword refinement with sentiment to classify the user's emotion."""
    normalized_text = text.lower()
    distressed_keywords = [
        "anxious",
        "panic",
        "overwhelmed",
        "can't handle",
        "stressed",
        "pressure",
    ]
    low_keywords = [
        "tired",
        "sad",
        "lonely",
        "down",
        "not okay",
    ]
    positive_keywords = [
        "happy",
        "good",
        "great",
        "relieved",
        "better",
    ]

    if any(word in normalized_text for word in distressed_keywords):
        return "distressed"
    if any(word in normalized_text for word in low_keywords):
        return "low"
    if any(word in normalized_text for word in positive_keywords):
        return "positive"

    try:
        polarity = TextBlob(text).sentiment.polarity
    except Exception:
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


def fallback_response(emotion: str) -> str:
    """Return a minimal safe fallback response when Groq is unavailable."""
    if emotion == "positive":
        return "That is wonderful to hear. I am glad things are feeling better."
    if emotion in {"negative", "distressed", "low"}:
        return "I am really sorry you are going through this. I am here to listen."
    return "I am listening. Please go on whenever you are ready."


def gratitude_response(language: str) -> str:
    """Return a short closing response for simple thank-you messages."""
    if language == "hi":
        return "\u0906\u092a\u0915\u093e \u0927\u0928\u094d\u092f\u0935\u093e\u0926\u0964 \u0906\u092a\u0915\u093e \u0926\u093f\u0928 \u0905\u091a\u094d\u091b\u093e \u0930\u0939\u0947\u0964"
    if language == "mr":
        return "\u0927\u0928\u094d\u092f\u0935\u093e\u0926\u0964 \u0924\u0941\u092e\u091a\u093e \u0926\u093f\u0935\u0938 \u091b\u093e\u0928 \u091c\u093e\u0935\u094b\u0964"
    return "You are very welcome. Have a nice day."


def build_context(profile: Dict[str, Any], sessions: List[Dict[str, Any]]) -> str:
    """Build a bounded text context block from the user profile and relevant sessions."""
    bounded_sessions = sessions[:8]
    context_lines: List[str] = []

    if profile:
        background = profile.get("background") or ""
        stress_source = profile.get("stress_source") or ""
        age = profile.get("age")
        context_lines.append(f"User background: {background}".strip())
        context_lines.append(f"Stress source: {stress_source}".strip())
        context_lines.append(f"Age: {age}" if age else "Age: Not provided")

    context_lines.append("Relevant past conversations:")

    if not bounded_sessions:
        context_lines.append("- No relevant past conversations found.")
        return "\n".join(context_lines)

    for session in bounded_sessions:
        user_text = session.get("user_input") or session.get("text") or "No user text stored."
        assistant_text = session.get("response") or "No assistant response stored."
        context_lines.append(f"- User: {user_text}")
        context_lines.append(f"  Assistant: {assistant_text}")

    return "\n".join(context_lines)


def retrieve_relevant_sessions(
    user_text: str,
    sessions: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return the most relevant prior sessions using lightweight keyword overlap."""
    user_words = set(re.findall(r"\w+", user_text.lower()))
    scored_sessions = []

    for session in sessions:
        session_text = f"{session.get('user_input', '')} {session.get('response', '')}".lower()
        session_words = set(re.findall(r"\w+", session_text))
        overlap = len(user_words.intersection(session_words))
        emotion_bonus = 1 if session.get("emotion") else 0
        score = overlap + emotion_bonus
        scored_sessions.append((score, session))

    scored_sessions.sort(key=lambda item: item[0], reverse=True)
    limited_sessions = [session for score, session in scored_sessions[:top_k] if score > 0]
    return limited_sessions[:8]


def detect_language(text: str) -> str:
    """Detect whether the user is writing in English or Hindi/Marathi."""
    normalized_text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = set(normalized_text.split())

    hindi_markers = {
        "mujhe", "mera", "meri", "mere", "main", "mai", "nahi", "nhi", "hai", "hu",
        "hoon", "haan", "kya", "kyu", "kaise", "kaisa", "acha", "accha", "bahut",
        "thoda", "dard", "pareshan", "dukhi", "ghar", "dil", "yaar", "mujhko", "tum",
        "aap", "kyon", "soch", "tension",
    }
    marathi_markers = {
        "mala", "majha", "majhi", "majhe", "ahe", "aahe", "nako", "khup", "kasa",
        "kashi", "kaay", "kay", "vatat", "baray", "bare", "udya", "aata", "tumhi",
        "mi", "tula", "tumhala", "karan", "man", "manat", "kalji", "taan", "thakla",
        "thakli", "havay", "pahije", "jasta",
    }
    devanagari_hindi_markers = {
        "\u092e\u0941\u091d\u0947",
        "\u092e\u0947\u0930\u093e",
        "\u092e\u0947\u0930\u0940",
        "\u092e\u0948\u0902",
        "\u0928\u0939\u0940\u0902",
        "\u0939\u0948",
        "\u092c\u0939\u0941\u0924",
        "\u0926\u0930\u094d\u0926",
        "\u092a\u0930\u0947\u0936\u093e\u0928",
    }
    devanagari_marathi_markers = {
        "\u092e\u0932\u093e",
        "\u092e\u093e\u091d\u093e",
        "\u092e\u093e\u091d\u0940",
        "\u092e\u093e\u091d\u0947",
        "\u0906\u0939\u0947",
        "\u0916\u0942\u092a",
        "\u0924\u093e\u0923",
        "\u0915\u093e\u0933\u091c\u0940",
        "\u0935\u093e\u091f\u0924",
    }

    hindi_matches = len(words.intersection(hindi_markers))
    marathi_matches = len(words.intersection(marathi_markers))

    if marathi_matches > hindi_matches and marathi_matches > 0:
        return "mr"
    if hindi_matches > 0:
        return "hi"

    if any(marker in text for marker in devanagari_marathi_markers):
        return "mr"
    if any(marker in text for marker in devanagari_hindi_markers):
        return "hi"

    if any("\u0900" <= ch <= "\u097F" for ch in text):
        return "hi"

    return "en"


def is_devanagari_text(text: str) -> bool:
    """Check whether a response contains Devanagari characters."""
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def is_thank_you_message(text: str) -> bool:
    """Detect simple gratitude messages that should end gently without a follow-up question."""
    normalized_text = re.sub(r"\s+", " ", text.strip().lower())
    gratitude_markers = [
        "thank you",
        "thanks",
        "thanks a lot",
        "thank u",
        "thankyou",
        "ok thanks",
        "dhanyavaad",
        "dhanyavad",
        "shukriya",
    ]
    return any(marker in normalized_text for marker in gratitude_markers)


def is_closing_message(text: str) -> bool:
    """Detect short closing messages that usually end a conversation."""
    normalized_text = re.sub(r"\s+", " ", text.strip().lower())
    closing_markers = {
        "bye",
        "bye bye",
        "goodbye",
        "ok bye",
        "okay bye",
        "see you",
        "see you later",
        "take care",
    }
    return normalized_text in closing_markers or is_thank_you_message(text)


def extract_audio_features(file_path: str) -> Dict[str, float]:
    """Extract a couple of lightweight audio features for context building."""
    import librosa
    import numpy as np

    y, sr = librosa.load(file_path, duration=5)
    energy = np.mean(librosa.feature.rms(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return {
        "energy": float(energy),
        "tempo": float(tempo),
    }


def generate_journal_suggestions(text: str, emotion: str) -> List[str]:
    """Return practical journal suggestions based on the latest text and emotion."""
    normalized_text = text.lower()

    if "stress" in normalized_text or "anx" in normalized_text or emotion == "distressed":
        return [
            "Try a 5-minute breathing exercise before your next task.",
            "Eat something light and nourishing instead of skipping meals.",
            "Take a short walk to release some physical tension.",
        ]
    if "sleep" in normalized_text or "tired" in normalized_text or "thak" in normalized_text:
        return [
            "Avoid caffeine late in the day and hydrate well.",
            "Keep dinner light and plan a calm screen-free bedtime.",
            "Try a gentle stretch before sleeping tonight.",
        ]
    if "food" in normalized_text or "eat" in normalized_text or "meal" in normalized_text:
        return [
            "Notice whether regular meals change your mood through the day.",
            "Add one simple nourishing snack if energy feels low.",
            "Write down which foods make you feel steady versus heavy.",
        ]
    if emotion == "low":
        return [
            "Write down one small win from today.",
            "Drink some water and eat something simple before your next task.",
            "Note one thing that would make tonight feel lighter.",
        ]
    if emotion == "positive":
        return [
            "Capture what went well so you can revisit it later.",
            "Write down the habit or moment that helped today feel better.",
            "Pair the good mood with a short walk or stretch to reinforce it.",
        ]
    return [
        "Write one sentence about what you need most today.",
        "Notice where you feel this emotion in your body.",
        "End with one small habit that could support you next.",
    ]


def generate_ai_response(
    text: str,
    emotion: str,
    context: str,
    language: str = "en",
    conversation_turn_count: int = 1,
) -> str:
    """Generate a personalized reply with Groq and fall back safely if needed."""
    target_language = (language or "").strip().lower()
    if target_language not in {"en", "hi", "mr"}:
        target_language = detect_language(text)

    gratitude_message = is_thank_you_message(text)
    closing_message = is_closing_message(text)

    if target_language == "hi":
        language_instruction = "Respond in Hindi (Devanagari script only)."
    elif target_language == "mr":
        language_instruction = "Respond in Marathi (Devanagari script only)."
    else:
        language_instruction = "Respond in English."

    system_prompt = (
        "You are a calm, empathetic mental wellness companion. "
        "Prefer concise responses of 1 to 2 sentences unless more is truly needed. "
        "Respond naturally in 1 to 2 short lines for the early part of the conversation, and use 2 to 4 short lines only when a suggestion is genuinely helpful later. "
        "Keep responses warm, human, and conversational, not repetitive or overly polished. "
        "Avoid robotic, clinical, or template-like replies. Keep the conversation light and compassionate. "
        "For the first 3 to 4 turns, prefer gentle listening, reflection, and a soft follow-up question when it feels natural. "
        "Do not rush into advice early. After there has been a little back-and-forth, offer one simple practical suggestion in about 2 to 4 short lines when it feels helpful. "
        "Do not combine a follow-up question and advice in every reply. "
        "If the user is only thanking you, warmly wish them a nice day and do not ask a follow-up question. "
        "After around 6-7 turns, if the conversation already feels complete, add a gentle supportive closing line "
        "that reassures the user it is okay to pause here for now. "
        f"{language_instruction}"
    )

    prompt = f"""
Reply to the user in a calm, supportive, human way.

Rules:
 - Prefer concise responses in 1 to 2 sentences unless more is truly necessary
 - In the first 3 to 4 turns, prefer a brief reflective reply in about 1 to 2 short lines
 - After that, if helpful, give one simple practical suggestion in about 2 to 4 short lines
 - Keep it short, human, and conversational
 - Sound warm, grounded, and natural
 - Do not switch languages
 - Do not use Urdu
 - Do not over-explain
 - Use the context naturally without sounding scripted
 - Keep the tone light and compassionate
 - In the first 3 to 4 turns, focus more on listening and understanding than advising
 - In those early turns, ask a gentle follow-up question only when it feels natural and useful
 - After a little back-and-forth, it is okay to include one simple suggestion the user can work on
 - Do not give a suggestion in every reply
 - Do not ask a follow-up question in every reply
 - Avoid sounding like a counselor script or an interviewer
 - If the user is only saying thank you, reply warmly, wish them a nice day, and do not ask a question
 - Do not force a follow-up question; only ask if it feels helpful
 - If this conversation has reached 6 or more turns, it is okay to end with a calm, encouraging concluding statement that supports wrapping up the conversation gently
 - If the user is saying bye or closing the chat, respond supportively and do not try to continue the conversation

Context:
{context}

Detected emotion:
{emotion}

Conversation turn count:
{conversation_turn_count}

User said:
{text}
"""

    if gratitude_message:
        return gratitude_response(target_language)
    if closing_message:
        if target_language == "hi":
            return "\u0920\u0940\u0915 \u0939\u0948, \u0905\u092a\u0928\u093e \u0927\u094d\u092f\u093e\u0928 \u0930\u0916\u093f\u090f\u0964 \u091c\u092c \u092d\u0940 \u092e\u0928 \u0939\u094b, \u092b\u093f\u0930 \u092c\u093e\u0924 \u0915\u0930 \u0938\u0915\u0924\u0947 \u0939\u0948\u0902\u0964"
        if target_language == "mr":
            return "\u0920\u0940\u0915 \u0906\u0939\u0947, \u0938\u094d\u0935\u0924\u0903\u091a\u0940 \u0915\u093e\u0933\u091c\u0940 \u0918\u094d\u092f\u093e. \u092a\u0941\u0928\u094d\u0939\u093e \u092c\u094b\u0932\u093e\u092f\u091a\u0902 \u0935\u093e\u091f\u0932\u0902 \u0924\u0930 \u0928\u0915\u094d\u0915\u0940 \u092f\u093e."
        return "Of course. Take care, and feel free to come back whenever you want to talk again."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return fallback_response(emotion)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
        )

        response = (completion.choices[0].message.content or "").strip()
        if target_language in {"hi", "mr"} and response and not is_devanagari_text(response):
            retry_prompt = f"""
Rewrite the following response in {'Hindi' if target_language == 'hi' else 'Marathi'} using only Devanagari script.
Do not use English, Roman letters, or Urdu script.

Original response:
{response}
"""
            retry_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_prompt} Do not use English or Urdu.",
                    },
                    {"role": "user", "content": retry_prompt},
                ],
            )
            retry_response = (retry_completion.choices[0].message.content or "").strip()
            if retry_response and is_devanagari_text(retry_response):
                return retry_response

        if response:
            return response
    except Exception as error:
        print("GROQ FAILED:", error)

    return fallback_response(emotion)


async def generate_speech(text, output_file, language: str = "en"):
    """Generate speech audio for the assistant response."""
    import edge_tts

    normalized_language = (language or "en").strip().lower()
    primary_voice = {
        "hi": "hi-IN-SwaraNeural",
        "mr": "mr-IN-AarohiNeural",
        "en": "en-US-JennyNeural",
    }.get(normalized_language, "en-US-JennyNeural")
    fallback_voice = "hi-IN-SwaraNeural" if normalized_language in {"hi", "mr"} else "en-US-JennyNeural"

    try:
        communicate = edge_tts.Communicate(text=text, voice=primary_voice)
        await communicate.save(output_file)
    except Exception:
        communicate = edge_tts.Communicate(text=text, voice=fallback_voice)
        await communicate.save(output_file)
