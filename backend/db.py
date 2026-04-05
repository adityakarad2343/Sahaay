"""Database helpers for the Sahaay Flask backend."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.environ.get("SAHAAY_DB_PATH", os.path.join(BASE_DIR, "sahaay.db"))


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection with dictionary-style rows enabled."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    """Create all required tables if they do not already exist."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                emotion TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_profile (
                user_id INTEGER PRIMARY KEY,
                age INTEGER,
                background TEXT,
                stress_source TEXT,
                stress_factor TEXT,
                sleep_pattern TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
        _ensure_user_profile_columns(connection)


def _ensure_user_profile_columns(connection: sqlite3.Connection) -> None:
    """Add new profile columns to older databases without recreating tables."""
    required_columns = {
        "age": "INTEGER",
        "background": "TEXT",
        "stress_source": "TEXT",
        "message_count": "INTEGER DEFAULT 0",
        "journal_suggestions": "TEXT",
    }
    existing_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(user_profile)").fetchall()
    }

    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE user_profile ADD COLUMN {column_name} {column_type}"
            )

    connection.commit()


def create_user(name: str, email: str, password_hash: str) -> int:
    """Insert a new user and create an empty profile row for that user."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO users (name, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (name, email, password_hash),
        )
        user_id = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO user_profile (
                user_id,
                age,
                background,
                stress_source,
                stress_factor,
                sleep_pattern
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, None, None, None, None, None),
        )
        connection.commit()
        return int(user_id)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch a user record by email address."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, name, email, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a user record by its database id."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, name, email
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def get_user_profile(user_id: int) -> Dict[str, Any]:
    """Fetch lightweight profile fields used for context retrieval."""
    default_profile = {
        "user_id": user_id,
        "age": None,
        "background": None,
        "stress_source": None,
    }

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT user_id, age, background, stress_source
            FROM user_profile
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    if not row:
        return default_profile

    profile = dict(row)
    return {
        "user_id": profile.get("user_id", user_id),
        "age": profile.get("age"),
        "background": profile.get("background"),
        "stress_source": profile.get("stress_source"),
    }


def _deserialize_journal_suggestions(raw_value: Any) -> List[str]:
    """Convert stored JSON suggestions into a safe list of strings."""
    if not raw_value:
        return []

    try:
        suggestions = json.loads(raw_value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    if not isinstance(suggestions, list):
        return []

    return [str(item).strip() for item in suggestions if str(item).strip()]


def get_user_journal_state(user_id: int) -> Dict[str, Any]:
    """Return the saved message counter and latest stored suggestions for a user."""
    default_state = {
        "message_count": 0,
        "journal_suggestions": [],
    }

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT message_count, journal_suggestions
            FROM user_profile
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    if not row:
        return default_state

    return {
        "message_count": int(row["message_count"] or 0),
        "journal_suggestions": _deserialize_journal_suggestions(row["journal_suggestions"]),
    }


def update_user_profile(
    user_id: int,
    name: str,
    age: Optional[int],
    background: Optional[str],
    stress_source: Optional[str],
) -> Dict[str, Any]:
    """Update the basic user and profile fields used across the frontend."""
    normalized_background = background.strip() if isinstance(background, str) else None
    normalized_stress_source = stress_source.strip() if isinstance(stress_source, str) else None

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE users
            SET name = ?
            WHERE id = ?
            """,
            (name, user_id),
        )
        connection.execute(
            """
            INSERT INTO user_profile (user_id, age, background, stress_source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                age = excluded.age,
                background = excluded.background,
                stress_source = excluded.stress_source
            """,
            (user_id, age, normalized_background, normalized_stress_source),
        )
        connection.commit()

    return {
        "user_id": user_id,
        "name": name,
        "age": age,
        "background": normalized_background,
        "stress_source": normalized_stress_source,
    }


def save_session(
    user_id: int,
    user_input: str,
    emotion: str,
    response: str,
    message_count: int,
    journal_suggestions: Optional[List[str]] = None,
) -> int:
    """Store one assistant conversation turn and persist the user's journal state."""
    timestamp = datetime.now(timezone.utc).isoformat()
    normalized_message_count = max(int(message_count), 0)
    serialized_suggestions = (
        json.dumps(journal_suggestions) if journal_suggestions is not None else None
    )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO user_profile (user_id)
            VALUES (?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id,),
        )
        cursor = connection.execute(
            """
            INSERT INTO sessions (user_id, user_input, emotion, response, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, user_input, emotion, response, timestamp),
        )

        if serialized_suggestions is None:
            connection.execute(
                """
                UPDATE user_profile
                SET message_count = ?
                WHERE user_id = ?
                """,
                (normalized_message_count, user_id),
            )
        else:
            connection.execute(
                """
                UPDATE user_profile
                SET message_count = ?, journal_suggestions = ?
                WHERE user_id = ?
                """,
                (normalized_message_count, serialized_suggestions, user_id),
            )

        connection.commit()
        return int(cursor.lastrowid)


def get_recent_sessions(user_id: int, limit: int = 15) -> List[Dict[str, Any]]:
    """Return the most recent conversation entries for a user."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                user_input,
                user_input AS text,
                emotion,
                response,
                timestamp
            FROM sessions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_sessions(user_id: int) -> List[Dict[str, Any]]:
    """Return all saved sessions for retrieval-based context building."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT user_input, response, emotion
            FROM sessions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_total_session_count(user_id: int) -> int:
    """Return the total number of saved sessions for a user."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM sessions
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return int(row["total"] if row else 0)


def create_journal_entry(user_id: int, content: str) -> Dict[str, Any]:
    """Create one journal entry for a user and return the saved row."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO journal_entries (user_id, content)
            VALUES (?, ?)
            """,
            (user_id, content),
        )
        entry_id = int(cursor.lastrowid)
        row = connection.execute(
            """
            SELECT id, content, created_at
            FROM journal_entries
            WHERE id = ? AND user_id = ?
            """,
            (entry_id, user_id),
        ).fetchone()
        connection.commit()
    return dict(row) if row else {}


def get_journal_entries(user_id: int) -> List[Dict[str, Any]]:
    """Return all saved journal entries for a user, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, content, created_at
            FROM journal_entries
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_journal_entry(user_id: int, entry_id: int, content: str) -> Optional[Dict[str, Any]]:
    """Update one journal entry owned by the user and return the updated row."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE journal_entries
            SET content = ?
            WHERE id = ? AND user_id = ?
            """,
            (content, entry_id, user_id),
        )
        if cursor.rowcount == 0:
            return None

        row = connection.execute(
            """
            SELECT id, content, created_at
            FROM journal_entries
            WHERE id = ? AND user_id = ?
            """,
            (entry_id, user_id),
        ).fetchone()
        connection.commit()
    return dict(row) if row else None


def delete_journal_entry(user_id: int, entry_id: int) -> bool:
    """Delete one journal entry owned by the user."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM journal_entries
            WHERE id = ? AND user_id = ?
            """,
            (entry_id, user_id),
        )
        connection.commit()
    return cursor.rowcount > 0
