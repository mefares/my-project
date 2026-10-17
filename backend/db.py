"""The only module that touches SQLite. One table: runs (one row per send)."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "runs.db"

# Columns the caller provides. id and created_at are filled in by the database / insert_run.
COLUMNS = [
    "provider", "model", "system_prompt", "temperature", "top_p", "max_tokens",
    "question", "answer", "input_tokens", "output_tokens", "finish_reason",
    "latency_ms", "error", "experiment", "setting_label",
]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,   -- UTC ISO timestamp
    provider      TEXT NOT NULL,   -- 'mistral' or 'gemini'
    model         TEXT NOT NULL,
    system_prompt TEXT,            -- NULL or empty = no system prompt
    temperature   REAL,            -- NULL = not sent (provider default)
    top_p         REAL,            -- NULL = not sent
    max_tokens    INTEGER,         -- NULL = not sent
    question      TEXT NOT NULL,
    answer        TEXT,            -- NULL when the call failed
    input_tokens  INTEGER,
    output_tokens INTEGER,         -- Gemini: answer tokens + thinking tokens
    finish_reason TEXT,            -- 'stop', 'length' or 'other'
    latency_ms    INTEGER,
    error         TEXT,            -- redacted error message; failed sends are saved too
    experiment    TEXT,            -- NULL for sends from the UI; set by replay.py
    setting_label TEXT             -- e.g. 'temp=0.0'; set by replay.py
)
"""


@contextmanager
def connect():
    """Open a connection for one `with` block: save the changes at the end, then close it.

    Rows can be read like dicts. One short connection per call keeps it simple.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db():
    """Create the table if it does not exist yet."""
    with connect() as connection:
        connection.execute(CREATE_TABLE)


def insert_run(**fields):
    """Save one run. Pass the columns by name; missing ones are stored as NULL. Returns the saved row."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown column(s): {sorted(unknown)}")
    values = {name: fields.get(name) for name in COLUMNS}
    values["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    names = ["created_at"] + COLUMNS
    placeholders = ", ".join("?" for _ in names)
    sql = f"INSERT INTO runs ({', '.join(names)}) VALUES ({placeholders})"
    with connect() as connection:
        cursor = connection.execute(sql, [values[name] for name in names])
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def list_runs(limit=50):
    """Latest runs first, as a list of dicts."""
    with connect() as connection:
        rows = connection.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
