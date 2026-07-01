"""
SQLite Run History Database
Saves every pipeline run so results persist across page refreshes.
Users can click any past run to restore its full results instantly.

Schema:
  runs   — one row per run (metadata + compressed JSON blob)
  papers — one row per paper per run (for quick listing)
"""
import sqlite3
import json
import gzip
import os
import hashlib
from datetime import datetime
from pathlib import Path

# Database lives next to the app so it persists across restarts
DB_DIR  = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "run_history.db"


def _get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            domain      TEXT,
            interest    TEXT,
            n_papers    INTEGER,
            n_gaps      INTEGER,
            n_proposals INTEGER,
            has_comp    INTEGER DEFAULT 0,
            results_gz  BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS papers (
            run_id    TEXT NOT NULL,
            filename  TEXT,
            title     TEXT,
            tier      INTEGER DEFAULT 1,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_papers_run   ON papers(run_id);
    """)
    conn.commit()
    conn.close()


def _compress(data: dict) -> bytes:
    return gzip.compress(json.dumps(data, default=str).encode())


def _decompress(blob: bytes) -> dict:
    return json.loads(gzip.decompress(blob).decode())


def _run_id(name: str, ts: str) -> str:
    return hashlib.md5(f"{name}{ts}".encode()).hexdigest()[:12]


def save_run(
    name: str,
    results: dict,
    comp_results: dict | None = None,
) -> str:
    """
    Persist a full pipeline run to the database.
    Returns the run_id for later retrieval.
    """
    init_db()
    ts  = datetime.now().isoformat(timespec="seconds")
    rid = _run_id(name, ts)

    context     = results.get("context", {})
    gaps        = results.get("gaps", [])
    proposals   = results.get("proposals", [])
    papers_list = results.get("papers", [])

    # Pack everything into one blob
    blob_data = {"results": results}
    if comp_results:
        blob_data["comp_results"] = comp_results
    compressed = _compress(blob_data)

    conn = _get_conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (id, name, created_at, domain, interest, n_papers, n_gaps,
                n_proposals, has_comp, results_gz)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                rid, name, ts,
                context.get("domain", ""),
                context.get("interest", "")[:120],
                len(papers_list),
                len(gaps),
                len(proposals),
                1 if comp_results else 0,
                compressed,
            ),
        )
        for p in papers_list:
            wiki  = p.get("wiki", {})
            title = wiki.get("title") or p.get("filename", "")
            conn.execute(
                "INSERT INTO papers (run_id, filename, title, tier) VALUES (?,?,?,?)",
                (rid, p.get("filename", ""), title, p.get("tier", 1)),
            )
        conn.commit()
    finally:
        conn.close()

    return rid


def list_runs(limit: int = 20) -> list[dict]:
    """Return recent runs (metadata only, no blob)."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, name, created_at, domain, n_papers, n_gaps,
                  n_proposals, has_comp
           FROM runs ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_run(run_id: str) -> dict:
    """
    Load full results for one run.
    Returns {'results': {...}, 'comp_results': {...} | None}
    """
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT results_gz FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {}
    return _decompress(row["results_gz"])


def delete_run(run_id: str) -> None:
    init_db()
    conn = _get_conn()
    conn.execute("DELETE FROM papers WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs    WHERE id     = ?", (run_id,))
    conn.commit()
    conn.close()


def get_run_papers(run_id: str) -> list[dict]:
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT filename, title, tier FROM papers WHERE run_id = ?", (run_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
