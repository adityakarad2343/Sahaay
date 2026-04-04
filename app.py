"""Main Flask application for the Sahaay voice assistant backend."""

import os
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict

import bcrypt
import jwt
import sqlite3
from flask import Flask, g, jsonify, request
from flask_cors import CORS

from db import create_user, get_recent_sessions, get_user_by_email, get_user_by_id, init_db, save_session
from utils import detect_emotion, generate_response


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "JWT_SECRET_KEY",
    "sahaay-dev-secret-key-change-this-for-production-12345",
)
app.config["JSON_SORT_KEYS"] = False
CORS(app)


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
    """Analyze user text, create a supportive response, and save the session."""
    data = request.get_json(silent=True) or {}
    validation_error = validate_text_payload(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    user_text = data["text"].strip()

    # These are the main assistant steps requested for the project.
    emotion = detect_emotion(user_text)
    assistant_response = generate_response(emotion, user_text)

    try:
        save_session(
            user_id=g.current_user_id,
            user_input=user_text,
            emotion=emotion,
            response=assistant_response,
        )
    except sqlite3.Error:
        return jsonify({"error": "Failed to save session."}), 500

    return jsonify({"text": assistant_response, "emotion": emotion}), 200


@app.route("/sessions/recent", methods=["GET"])
@token_required
def recent_sessions():
    """Return the latest three saved sessions for the authenticated user."""
    try:
        sessions = get_recent_sessions(g.current_user_id, limit=3)
    except sqlite3.Error:
        return jsonify({"error": "Failed to fetch recent sessions."}), 500

    return jsonify({"sessions": sessions}), 200


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
