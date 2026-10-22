import sqlite3
from pathlib import Path
from datetime import datetime


DB_PATH = Path(__file__).resolve().parent.parent / "users.db"


def initialize_history_table():
    """Create the analysis history table if it does not exist."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                real_probability REAL NOT NULL,
                fake_probability REAL NOT NULL,
                duration REAL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def save_analysis(
    username,
    filename,
    prediction,
    confidence,
    real_probability,
    fake_probability,
    duration=None
):
    """Validate and save one successful audio analysis."""

    import math

    if not isinstance(username, str) or not username.strip():
        raise ValueError("A logged-in username is required.")

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A valid filename is required.")

    if prediction not in ("Real", "Fake"):
        raise ValueError("Prediction must be Real or Fake.")

    try:
        confidence = float(confidence)
        real_probability = float(real_probability)
        fake_probability = float(fake_probability)

        if not all(math.isfinite(value) for value in (
            confidence, real_probability, fake_probability
        )):
            raise ValueError("Probabilities must be finite numbers.")

        if not all(0.0 <= value <= 1.0 for value in (
            confidence, real_probability, fake_probability
        )):
            raise ValueError("Probabilities must be between 0 and 1.")

        if duration is not None:
            duration = float(duration)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Duration must be a non-negative finite number.")

    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid analysis values.") from error

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO analysis_history (
                username,
                filename,
                prediction,
                confidence,
                real_probability,
                fake_probability,
                duration,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username.strip(),
            filename.strip(),
            prediction,
            confidence,
            real_probability,
            fake_probability,
            duration,
            datetime.now().astimezone().isoformat(timespec="seconds")
        ))
        conn.commit()


def get_user_history(username, limit=100):
    """Return the latest analysis records for one user."""

    if not username:
        return []

    initialize_history_table()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT
                id,
                filename,
                prediction,
                confidence,
                real_probability,
                fake_probability,
                duration,
                created_at
            FROM analysis_history
            WHERE username = ? COLLATE NOCASE
            ORDER BY id DESC
            LIMIT ?
        """, (username, max(1, min(int(limit), 500)))).fetchall()

        return [dict(row) for row in rows]


def delete_user_history(username):
    """Delete only the specified user's analysis history."""

    if not username:
        raise ValueError("A username is required.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            DELETE FROM analysis_history
            WHERE username = ? COLLATE NOCASE
        """, (username,))
        conn.commit()


initialize_history_table()
