import sqlite3
import hashlib
import hmac
import secrets
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "users.db"


def get_connection():
    connection = sqlite3.connect(str(DB_PATH))
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        600000,
    ).hex()

    return password_hash, salt


def register_user(username, password):
    username = username.strip()

    if len(username) < 3 or len(username) > 30:
        return False, "Username must be 3–30 characters long."

    if not all(char.isalnum() or char in "_-" for char in username):
        return False, "Use only letters, numbers, underscores, or hyphens."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if len(password) > 128:
        return False, "Password must not exceed 128 characters."

    password_hash, salt = hash_password(password)

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (username, password_hash, salt)
                VALUES (?, ?, ?)
                """,
                (username, password_hash, salt),
            )

        return True, "Account created successfully. You can now log in."

    except sqlite3.IntegrityError:
        return False, "This username is already registered."


def verify_user(username, password):
    if not isinstance(username, str) or not isinstance(password, str):
        return False

    username = username.strip()

    if not username or not password:
        return False

    if len(username) > 30 or len(password) > 128:
        return False

    try:
        with get_connection() as connection:
            user = connection.execute(
                """
                SELECT username, password_hash, salt
                FROM users
                WHERE username = ? COLLATE NOCASE
                """,
                (username,),
            ).fetchone()

        if user is None:
            return False

        stored_username, stored_hash, salt = user
        candidate_hash, _ = hash_password(password, salt)

        return hmac.compare_digest(candidate_hash, stored_hash)

    except sqlite3.Error:
        return False


initialize_database()
