"""Scheduler daemon: hands queued jobs to detached runners as slots free up.

Deliberately thin. It only decides *when* a job starts; the runner it spawns
owns everything after that. So restarting the scheduler - or killing it
outright - leaves running jobs untouched, and the queue simply resumes.

Slots:
  * one GPU job per L40, pinned via CUDA_VISIBLE_DEVICES
  * a separate pool for CPU-only work (WER, vocab, human cleanup) so cheap jobs
    never queue behind a six-hour transcription
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import db
from .config import ROOT, cfg

BOOT_LOG = ROOT / "logs" / "worker.log"


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reconcile() -> None:
    """Fail jobs whose runner vanished (machine reboot, OOM kill, kill -9)."""
    for job in db.running_jobs():
        pid = job["pid"]
        # A just-claimed job has no pid yet; give the runner time to record one.
        if pid is None:
            age = time.time() - (job["started_at"] or time.time())
            if age > 120:
                db.update(job["id"], status="failed", finished_at=time.time(),
                          error="runner never started")
            continue
        if not pid_alive(pid):
            db.update(job["id"], status="failed", finished_at=time.time(),
                      error=f"runner process {pid} disappeared")


def free_slots() -> tuple[list[int], int]:
    running = db.running_jobs()
    used_gpus = {j["gpu"] for j in running if j["gpu"] is not None}
    gpu_free = [g for g in cfg.gpus if g not in used_gpus]
    cpu_used = sum(1 for j in running if j["gpu"] is None)
    return gpu_free, max(0, cfg.max_cpu_jobs - cpu_used)


def spawn(job: dict) -> None:
    BOOT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BOOT_LOG, "a") as bl:
        subprocess.Popen(
            [sys.executable, "-m", "asrbench.runner", str(job["id"])],
            cwd=str(ROOT), stdout=bl, stderr=subprocess.STDOUT,
            start_new_session=True,   # detached: survives scheduler restart
        )
    where = f"gpu {job['gpu']}" if job["gpu"] is not None else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] started job {job['id']} "
          f"({job['module']}) on {where}", flush=True)


def loop() -> None:
    db.init()
    print(f"[{time.strftime('%H:%M:%S')}] scheduler up "
          f"(gpus={cfg.gpus}, cpu slots={cfg.max_cpu_jobs})", flush=True)

    stop = {"now": False}

    def on_term(signum, frame):
        stop["now"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    while not stop["now"]:
        try:
            reconcile()
            gpu_free, cpu_slots = free_slots()
            while gpu_free or cpu_slots > 0:
                job = db.claim_next(gpu_free, cpu_slots)
                if job is None:
                    break
                spawn(job)
                if job["gpu"] is not None:
                    gpu_free = [g for g in gpu_free if g != job["gpu"]]
                else:
                    cpu_slots -= 1
        except Exception as e:  # noqa: BLE001 - the scheduler must not die
            print(f"[{time.strftime('%H:%M:%S')}] scheduler error: {e!r}", flush=True)
        time.sleep(cfg.poll_seconds)

    print(f"[{time.strftime('%H:%M:%S')}] scheduler stopping "
          f"(running jobs are detached and continue)", flush=True)


if __name__ == "__main__":
    loop()
