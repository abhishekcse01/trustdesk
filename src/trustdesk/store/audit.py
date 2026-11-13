"""Append-only audit log of reviewed answers (G2), in SQLite.

Each row records the question, the final answer, its citations and verification scores, the
model and prompt version, who decided and when, plus a SHA-256 of the payload so later edits to
a row are detectable. The table has no UPDATE path in code.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    verification_json TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
);
"""


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(SCHEMA)
    return conn


def record(db: Path, entry: dict[str, Any]) -> str:
    """Insert one decision; returns the payload hash."""
    row = {
        "question_id": entry["question_id"],
        "question": entry["question"],
        "status": entry["status"],
        "decision": entry["decision"],
        "reviewer": entry["reviewer"],
        "answer_text": entry["answer_text"],
        "citations_json": json.dumps(entry.get("citations", []), ensure_ascii=False),
        "verification_json": json.dumps(entry.get("verification", {}), ensure_ascii=False),
        "model": entry.get("model", ""),
        "prompt_version": entry.get("prompt_version", ""),
        "trace_id": entry.get("trace_id", ""),
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()
    with connect(db) as conn:
        conn.execute(
            f"INSERT INTO answers ({', '.join(row)}, payload_sha256) "
            f"VALUES ({', '.join('?' for _ in row)}, ?)",
            [*row.values(), digest],
        )
    return digest


def recent(db: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not db.exists():
        return []
    with connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM answers ORDER BY id DESC LIMIT ?", [limit]).fetchall()
    return [dict(r) for r in rows]
