"""Execute one job's steps, in order, writing a single combined log.

Runs as its own detached process (`python -m asrbench.runner <job_id>`), so a
scheduler restart - or the web app going away entirely - never interrupts work
already in flight. The runner owns the job row from the moment it starts until
it reaches a terminal state.
"""
from __future__ import annotations

import fcntl
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import db
from .config import ROOT, cfg
from .plan import run_layout

LOCK_DIR = ROOT / "locks"


class Lock:
    """Cross-process lock so VTC1's shared model directory is used serially."""

    def __init__(self, name: str, log):
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOCK_DIR / f"{name}.lock"
        self.log = log
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.log(f"    waiting for '{self.path.stem}' lock "
                     f"(another job is using the shared VTC1 directory)...")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
            self.log(f"    acquired '{self.path.stem}' lock")
        return self

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
        return False


# ─────────────────────────────────────────────────────────────────
# cuDNN discovery
# ─────────────────────────────────────────────────────────────────

_ENV_PREFIX: dict[str, Path] = {}
_NVIDIA_LIBS: dict[str, str] = {}


def env_prefix(name: str) -> Path | None:
    """Locate a conda environment's prefix (cached; one `conda env list` call)."""
    if not _ENV_PREFIX:
        try:
            out = subprocess.run([str(cfg.conda), "env", "list", "--json"],
                                 capture_output=True, text=True, timeout=60)
            for p in json.loads(out.stdout).get("envs", []):
                _ENV_PREFIX[Path(p).name] = Path(p)
        except Exception:  # noqa: BLE001 - absence just disables the fix below
            pass
    return _ENV_PREFIX.get(name)


def nvidia_lib_dirs(env_name: str) -> str:
    """The pip-installed NVIDIA library directories inside a conda env.

    ctranslate2 dlopens libcudnn_cnn.so.9 by name at inference time, and without
    these on LD_LIBRARY_PATH it aborts with "Unable to load any of
    {libcudnn_cnn.so.9...}" mid-transcription - reproducible outside this system
    too, so it is an environment issue rather than one this wrapper introduced.

    Pipeline 3.0 hit the same wall and fixed it by prepending the whole conda
    lib/ directory, which shadowed the system libstdc++ with an older one and
    broke every ffmpeg and torchcodec subprocess; a note in run_pipeline.py
    records that it was disabled for exactly that reason. Adding only
    site-packages/nvidia/*/lib avoids the trap - those directories contain no
    libstdc++ at all - and is applied only to GPU steps.
    """
    if env_name in _NVIDIA_LIBS:
        return _NVIDIA_LIBS[env_name]
    prefix = env_prefix(env_name)
    dirs: list[str] = []
    if prefix:
        dirs = sorted(glob.glob(str(prefix / "lib" / "python*" / "site-packages"
                                    / "nvidia" / "*" / "lib")))
    _NVIDIA_LIBS[env_name] = ":".join(dirs)
    return _NVIDIA_LIBS[env_name]


def step_env(step: dict, base: dict) -> dict:
    """Environment for one step: base, plus cuDNN paths for GPU work."""
    if not step["gpu"] or step["env"] == "none":
        return base
    name = cfg.env_main if step["env"] == "main" else cfg.env_vtc1
    libs = nvidia_lib_dirs(name)
    if not libs:
        return base
    env = dict(base)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{libs}:{existing}".rstrip(":")
    return env


def build_argv(step: dict) -> list[str]:
    """Wrap a step's command in `conda run` for the environment it needs.

    VTC1 needs the python-3.8 `pyannote` env; everything else needs `whisperx`.
    VTC2 is invoked through `uv`, which manages its own venv, so it takes no
    conda wrapper at all.
    """
    env = step["env"]
    cmd = list(step["cmd"])
    if env == "none":
        return cmd
    name = cfg.env_main if env == "main" else cfg.env_vtc1
    return [str(cfg.conda), "run", "-n", name, "--no-capture-output", *cmd]


def run_job(job_id: int) -> int:
    job = db.get(job_id)
    if job is None:
        print(f"no such job: {job_id}", file=sys.stderr)
        return 2

    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    steps = json.loads(job["steps"])
    params = json.loads(job["params"])
    gpu = job["gpu"]

    logf = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")

    def log(msg: str = "") -> None:
        logf.write(msg + "\n")

    def stamp() -> str:
        return time.strftime("%H:%M:%S")

    # Provenance: enough to reproduce this run by hand later.
    (run_dir / "job.json").write_text(json.dumps({
        "id": job_id, "module": job["module"], "label": job["label"],
        "params": params, "steps": steps, "gpu": gpu,
        "created_at": job["created_at"],
    }, indent=2))

    log("=" * 78)
    log(f"  JOB {job_id}  -  {job['module']}  -  {job['label']}")
    log("=" * 78)
    log(f"  run dir : {run_dir}")
    log(f"  gpu     : {gpu if gpu is not None else 'none (CPU job)'}")
    log(f"  steps   : {len(steps)}")
    for k, v in params.items():
        log(f"  {k:<14}: {v}")
    log("=" * 78)
    log()

    base_env = os.environ.copy()
    if gpu is not None:
        # Pin the whole job to one L40 so two GPU jobs never contend.
        base_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    base_env["PYTHONUNBUFFERED"] = "1"

    for idx, step in enumerate(steps):
        db.update(job_id, step_index=idx, step_name=step["name"])
        log(f"[{stamp()}] STEP {idx + 1}/{len(steps)}  {step['name']}")

        argv = build_argv(step)
        log(f"    $ {' '.join(argv)}")
        if step["cwd"]:
            log(f"    (cwd: {step['cwd']})")

        started = time.time()
        try:
            lock_ctx = Lock(step["lock"], log) if step["lock"] else None
            if lock_ctx:
                lock_ctx.__enter__()
            try:
                proc = subprocess.run(
                    argv, cwd=step["cwd"] or None, env=step_env(step, base_env),
                    stdout=logf, stderr=subprocess.STDOUT, text=True,
                )
                rc = proc.returncode
            finally:
                if lock_ctx:
                    lock_ctx.__exit__(None, None, None)
        except FileNotFoundError as e:
            log(f"    !! command not found: {e}")
            rc = 127
        except Exception as e:  # noqa: BLE001 - any failure must land in the log
            log(f"    !! runner error: {e!r}")
            rc = 1

        elapsed = time.time() - started
        if rc != 0:
            log()
            log(f"[{stamp()}] FAILED at step {idx + 1} ({step['name']}) "
                f"with exit code {rc} after {elapsed:.1f}s")
            db.update(job_id, status="failed", finished_at=time.time(),
                      error=f"step {idx + 1} '{step['name']}' exited {rc}")
            logf.close()
            return rc

        log(f"[{stamp()}] ok ({elapsed:.1f}s)")
        log()

    db.update(job_id, status="done", step_index=len(steps),
              step_name="complete", finished_at=time.time())
    log("=" * 78)
    log(f"[{stamp()}] JOB {job_id} COMPLETE  -  {len(steps)} step(s)")
    log(f"  outputs: {run_dir}")
    log("=" * 78)
    logf.close()
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m asrbench.runner <job_id>", file=sys.stderr)
        return 2
    job_id = int(sys.argv[1])
    db.init()
    db.update(job_id, pid=os.getpid())

    def on_term(signum, frame):
        db.update(job_id, status="cancelled", finished_at=time.time(),
                  error="cancelled by user")
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)
    return run_job(job_id)


if __name__ == "__main__":
    sys.exit(main())
