"""Main Flask application for the Sahaay voice assistant backend."""

import asyncio
import io
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List

import bcrypt
import jwt
import sqlite3
from flask import Flask, g, jsonify, request, send_file
from flask_cors import CORS

try:
    import whisper
except Exception as error:
    whisper = None
    print("WHISPER IMPORT FAILED:", error)

from db import (
    create_journal_entry,
    create_user,
    get_all_sessions,
    delete_journal_entry,
    get_journal_entries,
    get_recent_sessions,
    get_total_session_count,
    get_user_journal_state,
    get_user_by_email,
    get_user_by_id,
    get_user_profile,
    init_db,
    save_session,
    update_journal_entry,
    update_user_profile,
)
from utils import (
    build_context,
    detect_emotion,
    detect_language,
    extract_audio_features,
    generate_ai_response,
    generate_journal_suggestions,
    generate_speech,
    retrieve_relevant_sessions,
)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "JWT_SECRET_KEY",
    "sahaay-dev-secret-key-change-this-for-production-12345",
)
app.config["JSON_SORT_KEYS"] = False
CORS(app)
TEMP_AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio")
try:
    WHISPER_MODEL = whisper.load_model("small") if whisper is not None else None
except Exception as error:
    WHISPER_MODEL = None
    print("WHISPER INIT FAILED:", error)


def create_jwt_token(user_id: int, email: str) -> str:
    """Create a signed JWT token for an authenticated user."""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=24),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")
    return token if isinstance(token, str) else token.decode("utf-8")


def token_required(route_function: Callable[..., Any]) -> Callable[..., Any]:
    """Protect a route by requiring a valid Bearer token."""

    @wraps(route_function)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        auth_header = request.headers.get("Authorization", "").strip()
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization token is missing."}), 401

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return jsonify({"error": "Authorization token is missing."}), 401

        try:
            payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            user = get_user_by_id(int(payload["user_id"]))
            if not user:
                return jsonify({"error": "User not found."}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired."}), 401
        except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
            return jsonify({"error": "Invalid token."}), 401

        g.current_user_id = user["id"]
        g.current_user = user
        return route_function(*args, **kwargs)

    return decorated_function


def validate_auth_payload(data: Dict[str, Any], require_name: bool = False) -> str:
    """Validate login/register payloads and return an error message when invalid."""
    required_fields = ["email", "password"]
    if require_name:
        required_fields.insert(0, "name")

    for field in required_fields:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"{field.capitalize()} is required."

    return ""


def validate_text_payload(data: Dict[str, Any]) -> str:
    """Validate the input text for the assistant processing route."""
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return "Text is required."
    return ""


def validate_journal_payload(data: Dict[str, Any]) -> str:
    """Validate the journal entry payload."""
    content = data.get("content")
    if not isinstance(content, str) or not content.strip():
        return "Content is required."
    return ""


def normalize_optional_text(value: Any) -> str | None:
    """Normalize optional string form fields."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned_value = value.strip()
    return cleaned_value or None


def normalize_optional_age(value: Any) -> int | None:
    """Convert an optional age field into an integer when possible."""
    cleaned_value = normalize_optional_text(value)
    if cleaned_value is None:
        return None

    try:
        age = int(cleaned_value)
    except (TypeError, ValueError):
        return None

    return age if age > 0 else None


def create_tts_audio(assistant_response: str, language: str = "en") -> str:
    """Generate an MP3 file for the assistant response and return its filename."""
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    filename = f"response_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(TEMP_AUDIO_DIR, filename)
    asyncio.run(generate_speech(assistant_response, filepath, language=language))
    return filename


def prepare_journal_update(user_id: int, user_text: str, emotion: str) -> tuple[int, bool, list[str] | None]:
    """Determine whether this response should refresh the stored journal suggestions."""
    journal_state = get_user_journal_state(user_id)
    message_count = int(journal_state["message_count"]) + 1
    journal_updated = message_count % 5 == 0
    suggestions = generate_journal_suggestions(user_text, emotion) if journal_updated else None
    return message_count, journal_updated, suggestions


def cleanup_audio_file(filepath: str) -> None:
    """Remove a generated audio file when explicit cleanup is requested."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass


def build_insights_summary(emotion_counts: Dict[str, int], recent_emotions: List[str]) -> str:
    """Create a short natural-language summary for the insights page."""
    if not emotion_counts:
        return "You have not had enough recent conversations yet to build an insights summary."

    sorted_emotions = sorted(
        emotion_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    dominant_emotion, dominant_count = sorted_emotions[0]
    recent_window = recent_emotions[-5:]
    low_count = sum(1 for emotion in recent_window if emotion in {"low", "distressed"})
    positive_count = sum(1 for emotion in recent_window if emotion == "positive")

    if positive_count >= 3:
        trend_note = "Your last few conversations look a little lighter than before."
    elif low_count >= 3:
        trend_note = "Your latest conversations suggest a stretch of heavier emotional load."
    else:
        trend_note = "Your recent pattern looks mixed, which can happen during busy or changing periods."

    if dominant_emotion == "distressed":
        suggestion = "Consider slowing the pace of your day and taking one grounding break when things feel intense."
    elif dominant_emotion == "low":
        suggestion = "A small reset like water, food, or a short walk could help create a little more steadiness."
    elif dominant_emotion == "positive":
        suggestion = "It may help to notice what has been supporting you so you can keep some of that momentum."
    else:
        suggestion = "A quick check-in with yourself each day could help you notice patterns before they build up."

    return (
        f"You have had {dominant_count} recent {dominant_emotion} check-in"
        f"{'' if dominant_count == 1 else 's'}, making it your most common emotion lately. "
        f"{trend_note} {suggestion}"
    )


def build_insights_payload(user_id: int) -> Dict[str, Any]:
    """Return the data needed to power the insights page."""
    sessions = get_recent_sessions(user_id, limit=15)
    total_sessions = get_total_session_count(user_id)
    emotion_counts: Dict[str, int] = {}

    for session in sessions:
        emotion = session.get("emotion") or "neutral"
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    recent_emotions = [
        session.get("emotion") or "neutral"
        for session in reversed(sessions[:5])
    ]

    return {
        "total_sessions": total_sessions,
        "emotion_counts": emotion_counts,
        "recent_emotions": recent_emotions,
        "summary": build_insights_summary(emotion_counts, recent_emotions),
        "sessions": sessions,
    }


@app.route("/", methods=["GET"])
def health_check():
    """Return a friendly message so it is easy to confirm the API is running."""
    return jsonify({"message": "Sahaay backend is running."}), 200


@app.route("/register", methods=["POST"])
def register():
    """Create a new user account with a bcrypt-hashed password."""
    data = request.get_json(silent=True) or {}
    validation_error = validate_auth_payload(data, require_name=True)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    name = data["name"].strip()
    email = data["email"].strip().lower()
    password = data["password"].strip()

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    try:
        create_user(name=name, email=email, password_hash=password_hash)
    except sqlite3.IntegrityError:
        return jsonify({"error": "A user with this email already exists."}), 409
    except sqlite3.Error:
        return jsonify({"error": "Failed to register user."}), 500

    return jsonify({"message": "User registered successfully."}), 201


@app.route("/login", methods=["POST"])
def login():
    """Authenticate a user and return a signed JWT token."""
    data = request.get_json(silent=True) or {}
    validation_error = validate_auth_payload(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    email = data["email"].strip().lower()
    password = data["password"].strip()
    user = get_user_by_email(email)

    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    password_matches = bcrypt.checkpw(
        password.encode("utf-8"),
        user["password_hash"].encode("utf-8"),
    )
    if not password_matches:
        return jsonify({"error": "Invalid email or password."}), 401

    token = create_jwt_token(user["id"], user["email"])
    return jsonify({"message": "Login successful.", "token": token}), 200


@app.route("/process_audio", methods=["POST"])
@token_required
def process_audio():
    """Analyze user text, retrieve context, create a response, and save the session."""
    data = request.get_json(silent=True) or {}
    validation_error = validate_text_payload(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    user_id = g.current_user_id
    user_name = g.current_user["name"]
    user_text = data["text"].strip()
    language = (data.get("language") or "").strip().lower()
    if language not in {"en", "hi", "mr"}:
        language = detect_language(user_text)

    emotion = detect_emotion(user_text)
    try:
        profile = get_user_profile(user_id)
        all_session_data = get_all_sessions(user_id)
        relevant_session_data = retrieve_relevant_sessions(
            user_text,
            all_session_data,
            top_k=8,
        )
        context = build_context(profile, relevant_session_data)
    except sqlite3.Error:
        return jsonify({"error": "Failed to load user context."}), 500

    assistant_response = generate_ai_response(
        user_text,
        emotion,
        context,
        language=language,
        conversation_turn_count=len(all_session_data) + 1,
    )
    print("LLM response ready")

    # Safety fallback (never allow None)
    if not assistant_response:
        assistant_response = "I'm here for you. Tell me more."

    try:
        message_count, _, journal_suggestions = prepare_journal_update(
            user_id,
            user_text,
            emotion,
        )
    except sqlite3.Error:
        return jsonify({"error": "Failed to prepare journal updates."}), 500

    try:
        filename = create_tts_audio(assistant_response, language=language)
    except Exception as error:
        print("TTS ERROR:", error)
        filename = None

    # Save session
    try:
        save_session(
            user_id=user_id,
            user_input=user_text,
            emotion=emotion,
            response=assistant_response,
            message_count=message_count,
            journal_suggestions=journal_suggestions,
        )
    except sqlite3.Error:
        return jsonify({"error": "Failed to save session."}), 500

    return jsonify({
        "response": assistant_response,
        "text": assistant_response,
        "emotion": emotion,
        "language": language,
        "name": user_name,
        "audio_url": f"/audio/{filename}" if filename else None,
    }), 200


@app.route("/sessions/recent", methods=["GET"])
@token_required
def recent_sessions():
    """Return the latest three saved sessions for the authenticated user."""
    try:
        sessions = get_recent_sessions(g.current_user_id, limit=3)
    except sqlite3.Error:
        return jsonify({"error": "Failed to fetch recent sessions."}), 500

    return jsonify({"sessions": sessions, "name": g.current_user["name"]}), 200


@app.route("/insights", methods=["GET"])
@token_required
def insights():
    """Return aggregated insights for the authenticated user's recent sessions."""
    try:
        payload = build_insights_payload(g.current_user_id)
    except sqlite3.Error:
        return jsonify({"error": "Failed to load insights."}), 500

    payload["name"] = g.current_user["name"]
    return jsonify(payload), 200


@app.route("/journal/suggestions", methods=["GET"])
@token_required
def journal_suggestions():
    """Return the latest stored journal suggestions for the authenticated user."""
    try:
        journal_state = get_user_journal_state(g.current_user_id)
    except sqlite3.Error:
        return jsonify({"error": "Failed to load suggestions."}), 500

    return jsonify(
        {
            "suggestions": journal_state["journal_suggestions"],
            "message_count": journal_state["message_count"],
            "name": g.current_user["name"],
        }
    ), 200


@app.route("/profile", methods=["GET"])
@token_required
def get_profile():
    """Return the authenticated user's profile information."""
    try:
        profile = get_user_profile(g.current_user_id)
    except sqlite3.Error:
        return jsonify({"error": "Failed to load profile."}), 500

    return jsonify(
        {
            "user_id": g.current_user_id,
            "name": g.current_user["name"],
            "age": profile.get("age"),
            "background": profile.get("background"),
            "stress_source": profile.get("stress_source"),
        }
    ), 200


@app.route("/profile/update", methods=["POST"])
@token_required
def save_profile():
    """Update the authenticated user's profile information."""
    data = request.get_json(silent=True) or {}
    current_user = g.current_user

    name = normalize_optional_text(data.get("name")) or current_user["name"]
    age = normalize_optional_age(data.get("age"))
    background = normalize_optional_text(data.get("background"))
    stress_source = normalize_optional_text(data.get("stress_source"))

    try:
        profile = update_user_profile(
            g.current_user_id,
            name=name,
            age=age,
            background=background,
            stress_source=stress_source,
        )
    except sqlite3.Error:
        return jsonify({"error": "Failed to update profile."}), 500

    g.current_user["name"] = name
    return jsonify(profile), 200


@app.route("/journal", methods=["POST"])
@token_required
def create_journal_note():
    """Create a journal entry for the authenticated user."""
    data = request.get_json(silent=True) or {}
    validation_error = validate_journal_payload(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    try:
        entry = create_journal_entry(g.current_user_id, data["content"].strip())
    except sqlite3.Error:
        return jsonify({"error": "Failed to save journal entry."}), 500

    return jsonify(entry), 201


@app.route("/journal/add", methods=["POST"])
@token_required
def create_journal_note_alias():
    """Backward-compatible alias for saving a journal entry."""
    return create_journal_note()


@app.route("/journal", methods=["GET"])
@token_required
def list_journal_notes():
    """Return all journal entries for the authenticated user."""
    try:
        entries = get_journal_entries(g.current_user_id)
    except sqlite3.Error:
        return jsonify({"error": "Failed to fetch journal entries."}), 500

    return jsonify(entries), 200


@app.route("/journal/<int:entry_id>", methods=["PUT"])
@token_required
def edit_journal_note(entry_id: int):
    """Update one journal entry for the authenticated user."""
    data = request.get_json(silent=True) or {}
    validation_error = validate_journal_payload(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    try:
        entry = update_journal_entry(g.current_user_id, entry_id, data["content"].strip())
    except sqlite3.Error:
        return jsonify({"error": "Failed to update journal entry."}), 500

    if not entry:
        return jsonify({"error": "Journal entry not found."}), 404

    return jsonify(entry), 200


@app.route("/journal/<int:entry_id>", methods=["DELETE"])
@token_required
def remove_journal_note(entry_id: int):
    """Delete one journal entry for the authenticated user."""
    try:
        deleted = delete_journal_entry(g.current_user_id, entry_id)
    except sqlite3.Error:
        return jsonify({"error": "Failed to delete journal entry."}), 500

    if not deleted:
        return jsonify({"error": "Journal entry not found."}), 404

    return jsonify({"message": "Journal entry deleted."}), 200


@app.route("/process_audio_file", methods=["POST"])
@token_required
def process_audio_file():
    """Transcribe an uploaded audio file and run it through the existing response pipeline."""
    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "Audio file is required."}), 400

    user_id = g.current_user_id
    file_extension = os.path.splitext(audio_file.filename)[1] or ".wav"
    language = (request.form.get("language") or "").strip().lower()
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            audio_file.save(temp_file_path)

        if WHISPER_MODEL is None:
            return jsonify({"error": "Speech transcription is unavailable."}), 500

        try:
            import librosa
            import soundfile as sf

            y, sr = librosa.load(temp_file_path, duration=6)
            y_trimmed, _ = librosa.effects.trim(y, top_db=40)
            if getattr(y_trimmed, "size", 0) == 0:
                y_trimmed = y
            sf.write(temp_file_path, y_trimmed, sr)
            print("Audio trimmed")
        except Exception as error:
            print("AUDIO PREPROCESS ERROR:", error)

        if language in {"hi", "mr"}:
            transcription_result = WHISPER_MODEL.transcribe(
                temp_file_path,
                language="hi",
                fp16=False,
            )
        else:
            transcription_result = WHISPER_MODEL.transcribe(temp_file_path)
        print("Transcription done")
        user_text = (transcription_result.get("text") or "").strip()

        if not user_text:
            return jsonify({"error": "Could not transcribe audio."}), 400

        if language not in {"en", "hi", "mr"}:
            language = detect_language(user_text)

        try:
            profile = get_user_profile(user_id)
            all_session_data = get_all_sessions(user_id)
        except sqlite3.Error:
            return jsonify({"error": "Failed to load user context."}), 500

        emotion = detect_emotion(user_text)
        relevant_session_data = retrieve_relevant_sessions(
            user_text,
            all_session_data,
            top_k=8,
        )
        context = build_context(profile, relevant_session_data)

        try:
            features = extract_audio_features(temp_file_path)
            context += f"\nAudio Energy: {features['energy']}, Tempo: {features['tempo']}"
        except Exception as error:
            print("LIBROSA ERROR:", error)

        assistant_response = generate_ai_response(
            user_text,
            emotion,
            context,
            language=language,
            conversation_turn_count=len(all_session_data) + 1,
        )
        print("LLM response ready")

        if not assistant_response:
            assistant_response = "I'm here for you. Tell me more."

        try:
            message_count, _, journal_suggestions = prepare_journal_update(
                user_id,
                user_text,
                emotion,
            )
        except sqlite3.Error:
            return jsonify({"error": "Failed to prepare journal updates."}), 500

        try:
            filename = create_tts_audio(assistant_response, language=language)
        except Exception as error:
            print("TTS ERROR:", error)
            filename = None

        try:
            save_session(
                user_id=user_id,
                user_input=user_text,
                emotion=emotion,
                response=assistant_response,
                message_count=message_count,
                journal_suggestions=journal_suggestions,
            )
        except sqlite3.Error:
            return jsonify({"error": "Failed to save session."}), 500

        return jsonify(
            {
                "text": user_text,
                "response": assistant_response,
                "emotion": emotion,
                "language": language,
                "name": g.current_user["name"],
                "audio_url": f"/audio/{filename}" if filename else None,
            }
        ), 200
    except Exception as error:
        print("WHISPER ERROR:", error)
        return jsonify({"error": "Failed to process audio file."}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


@app.route("/audio/<filename>")
def serve_audio(filename: str):
    """Serve generated TTS audio files."""
    filepath = os.path.join(TEMP_AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Audio file not found."}), 404

    with open(filepath, "rb") as audio_file:
        audio_bytes = audio_file.read()

    response = send_file(
        io.BytesIO(audio_bytes),
        mimetype="audio/mpeg",
        download_name=filename,
        max_age=3600,
    )
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@app.route("/audio/cleanup", methods=["POST"])
@token_required
def cleanup_audio():
    """Delete generated TTS files when the user explicitly logs out."""
    if not os.path.isdir(TEMP_AUDIO_DIR):
        return jsonify({"deleted": 0}), 200

    deleted_files = 0
    for filename in os.listdir(TEMP_AUDIO_DIR):
        filepath = os.path.join(TEMP_AUDIO_DIR, filename)
        if os.path.isfile(filepath):
            if filename.startswith("response_") and filename.endswith(".mp3"):
                before_delete = os.path.exists(filepath)
                cleanup_audio_file(filepath)
                if before_delete and not os.path.exists(filepath):
                    deleted_files += 1

    return jsonify({"deleted": deleted_files}), 200


@app.errorhandler(404)
def handle_not_found(_: Exception):
    """Return a JSON response for unknown routes."""
    return jsonify({"error": "Route not found."}), 404


@app.errorhandler(405)
def handle_method_not_allowed(_: Exception):
    """Return a JSON response for unsupported HTTP methods."""
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def handle_internal_error(_: Exception):
    """Return a JSON response for unexpected server failures."""
    return jsonify({"error": "Internal server error."}), 500


def create_app() -> Flask:
    """Factory-style helper that ensures the database exists before serving requests."""
    init_db()
    return app


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
