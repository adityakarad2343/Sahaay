"""Database helpers for the Sahaay Flask backend."""

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
                stress_factor TEXT,
                sleep_pattern TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )


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
            INSERT INTO user_profile (user_id, stress_factor, sleep_pattern)
            VALUES (?, ?, ?)
            """,
            (user_id, None, None),
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


def save_session(user_id: int, user_input: str, emotion: str, response: str) -> int:
    """Store one assistant conversation turn for a user."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO sessions (user_id, user_input, emotion, response, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, user_input, emotion, response, timestamp),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_recent_sessions(user_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    """Return the most recent conversation entries for a user."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, user_input, emotion, response, timestamp
            FROM sessions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]
