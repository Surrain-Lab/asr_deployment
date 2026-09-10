"""SQLite job store.

Three processes touch this database: the web app, the scheduler, and one
runner per active job. WAL mode plus short-lived connections keeps them from
blocking each other.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .config import cfg

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    module       TEXT    NOT NULL,
    label        TEXT    NOT NULL DEFAULT '',
    params       TEXT    NOT NULL DEFAULT '{}',
    steps        TEXT    NOT NULL DEFAULT '[]',
    step_index   INTEGER NOT NULL DEFAULT 0,
    step_name    TEXT    NOT NULL DEFAULT '',
    status       TEXT    NOT NULL DEFAULT 'queued',
    needs_gpu    INTEGER NOT NULL DEFAULT 0,
    gpu          INTEGER,
    pid          INTEGER,
    run_dir      TEXT    NOT NULL DEFAULT '',
    log_path     TEXT    NOT NULL DEFAULT '',
    error        TEXT    NOT NULL DEFAULT '',
    depends_on   INTEGER,
    created_at   REAL    NOT NULL,
    started_at   REAL,
    finished_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

TERMINAL = ("done", "failed", "cancelled")


@contextmanager
def conn():
    c = sqlite3.connect(cfg.db_path, timeout=30.0)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
    with conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)


def create_job(module: str, label: str, params: dict, steps: list,
               needs_gpu: bool, run_dir: str, log_path: str,
               depends_on: int | None = None) -> int:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO jobs (module, label, params, steps, status, needs_gpu,
                                 run_dir, log_path, depends_on, created_at)
               VALUES (?,?,?,?,'queued',?,?,?,?,?)""",
            (module, label, json.dumps(params), json.dumps(steps),
             1 if needs_gpu else 0, run_dir, log_path, depends_on, time.time()),
        )
        return int(cur.lastrowid)


def get(job_id: int) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 100, status: str | None = None) -> list[dict]:
    q = "SELECT * FROM jobs"
    args: tuple = ()
    if status:
        q += " WHERE status=?"
        args = (status,)
    q += " ORDER BY id DESC LIMIT ?"
    args += (limit,)
    with conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def update(job_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE jobs SET {sets} WHERE id=?",
                  (*fields.values(), job_id))


def claim_next(gpu_free: list[int], cpu_slots: int) -> dict | None:
    """Atomically pick and mark one runnable queued job.

    A job is runnable when its dependency (if any) finished successfully and a
    slot of the right kind is free. GPU jobs are preferred so the L40s do not
    sit idle behind a queue of cheap CPU work.
    """
    with conn() as c:
        c.execute("BEGIN IMMEDIATE")
        rows = c.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY id ASC"
        ).fetchall()
        for row in rows:
            job = dict(row)
            dep = job["depends_on"]
            if dep is not None:
                d = c.execute("SELECT status FROM jobs WHERE id=?", (dep,)).fetchone()
                if d is None or d["status"] != "done":
                    # Dependency failed or was cancelled: this job can never run.
                    if d is not None and d["status"] in ("failed", "cancelled"):
                        c.execute(
                            "UPDATE jobs SET status='cancelled', error=?, finished_at=? WHERE id=?",
                            (f"dependency job {dep} did not succeed", time.time(), job["id"]),
                        )
                    continue
            if job["needs_gpu"]:
                if not gpu_free:
                    continue
                gpu = gpu_free[0]
                c.execute(
                    "UPDATE jobs SET status='running', gpu=?, started_at=? WHERE id=?",
                    (gpu, time.time(), job["id"]),
                )
                job["status"], job["gpu"] = "running", gpu
                return job
            if cpu_slots <= 0:
                continue
            c.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?",
                      (time.time(), job["id"]))
            job["status"] = "running"
            return job
    return None


def running_jobs() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in
                c.execute("SELECT * FROM jobs WHERE status='running'").fetchall()]
