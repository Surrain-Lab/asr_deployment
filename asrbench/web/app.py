"""ASR_Benchmark web UI.

Bound to 127.0.0.1 and reached over an SSH tunnel, so there is no auth layer
and no route may ever touch a path outside the configured browse roots.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import signal
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, PlainTextResponse,
                               RedirectResponse, StreamingResponse)
from fastapi.templating import Jinja2Templates

from .. import db, plan
from ..config import ROOT, cfg
from ..discover import scan_audio, scan_text

app = FastAPI(title="ASR Benchmark")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
# Shown in the sidebar so it is obvious which server a tab is pointed at
# when several are tunnelled at once.
templates.env.globals["HOSTNAME"] = os.uname().nodename

# Directories excluded from the results zip: large, regenerable, not results.
BULKY = {"audio_16k", "chunks", "raw_transcripts"}

# Checkbox convention: a browser omits an unchecked box from the POST body
# entirely, so every checkbox parameter defaults to "" (off) here and the
# template's `checked` attribute decides what is on by default. Defaulting to
# "on" would make the box impossible to untick.

MODULE_INFO = {
    "human": {
        "title": "Human Transcript Cleanup",
        "blurb": "CHAT transcripts to clean, per-speaker and combined reference text.",
        "icon": "H",
    },
    "asr": {
        "title": "Audio Transcription",
        "blurb": "Audio to transcripts through any of the five pipeline variants.",
        "icon": "A",
    },
    "wer": {
        "title": "WER Evaluation",
        "blurb": "Score a hypothesis folder against a reference, overall or split by English vs Spanish.",
        "icon": "W",
    },
    "langvocab": {
        "title": "Language & Vocabulary",
        "blurb": "EN/ES detection, word counts, vocabulary tables, WER by language.",
        "icon": "L",
    },
    "full": {
        "title": "Full Benchmark",
        "blurb": "Steps 0-5 end to end across every selected pipeline variant.",
        "icon": "F",
    },
}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def safe_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not cfg.is_browsable(p):
        raise HTTPException(400, f"path is outside the allowed roots: {raw}")
    return p


def run_path(job: dict, rel: str) -> Path:
    base = Path(job["run_dir"]).resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(400, "path escapes the run directory")
    return target


def fmt_duration(job: dict) -> str:
    start = job["started_at"]
    if not start:
        return "-"
    end = job["finished_at"] or time.time()
    s = int(end - start)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


def decorate(job: dict) -> dict:
    steps = json.loads(job["steps"] or "[]")
    job["n_steps"] = len(steps)
    job["duration"] = fmt_duration(job)
    job["params_d"] = json.loads(job["params"] or "{}")
    job["pct"] = int(100 * job["step_index"] / len(steps)) if steps else 0
    if job["status"] == "done":
        job["pct"] = 100
    job["info"] = MODULE_INFO.get(job["module"], {"title": job["module"]})
    return job


# ─────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup() -> None:
    db.init()
    cfg.runs_root.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    jobs = [decorate(j) for j in db.list_jobs(50)]
    return templates.TemplateResponse(request, "index.html", {
        "request": request, "jobs": jobs, "modules": MODULE_INFO,
    })


@app.get("/jobs/fragment", response_class=HTMLResponse)
def jobs_fragment(request: Request):
    jobs = [decorate(j) for j in db.list_jobs(50)]
    return templates.TemplateResponse(request, "_joblist.html",
                                      {"request": request, "jobs": jobs})


# ─────────────────────────────────────────────────────────────────
# Directory browsing and corpus preview
# ─────────────────────────────────────────────────────────────────

@app.get("/browse", response_class=HTMLResponse)
def browse(request: Request, path: str = "", field: str = "", kind: str = "audio"):
    start = Path(path).expanduser() if path else cfg.browse_roots[0]
    if not cfg.is_browsable(start):
        start = cfg.browse_roots[0]
    if not start.is_dir():
        start = start.parent if start.parent.is_dir() else cfg.browse_roots[0]

    try:
        entries = sorted(
            [d for d in start.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda p: p.name.lower(),
        )
    except PermissionError:
        entries = []

    parent = start.parent if cfg.is_browsable(start.parent) and start != start.parent else None
    corpus = (scan_audio(start) if kind == "audio" else scan_text(start))

    return templates.TemplateResponse(request, "_browser.html", {
        "request": request, "cwd": start, "entries": entries, "parent": parent,
        "field": field, "kind": kind, "corpus": corpus.as_dict(),
        "roots": cfg.browse_roots,
    })


@app.get("/preview", response_class=HTMLResponse)
def preview(request: Request, path: str = "", kind: str = "audio"):
    """Structure check shown before launch, so a wrong folder level is caught
    in seconds rather than after a six-hour job produces nothing."""
    if not path:
        return HTMLResponse("")
    p = Path(path).expanduser()
    if not cfg.is_browsable(p) or not p.is_dir():
        return templates.TemplateResponse(request, "_preview.html", {
            "request": request, "corpus": None, "path": path})
    corpus = (scan_audio(p) if kind == "audio" else scan_text(p))
    return templates.TemplateResponse(request, "_preview.html", {
        "request": request, "corpus": corpus.as_dict(), "path": path})


# ─────────────────────────────────────────────────────────────────
# Job creation
# ─────────────────────────────────────────────────────────────────

@app.get("/new/{module}", response_class=HTMLResponse)
def new_form(request: Request, module: str):
    if module not in MODULE_INFO:
        raise HTTPException(404, "unknown module")
    return templates.TemplateResponse(request, f"form_{module}.html", {
        "request": request, "module": module, "info": MODULE_INFO[module],
        "variants": plan.VARIANTS, "modules": MODULE_INFO,
    })


def launch(module: str, label: str, params: dict) -> int:
    needs_gpu = module in plan.GPU_MODULES
    job_id = db.create_job(module, label, params, [], needs_gpu, "", "")

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.runs_root / f"{job_id:05d}_{module}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "job.log"

    try:
        steps = plan.build(module, params, run_dir)
    except Exception as e:  # noqa: BLE001 - surface planning errors in the UI
        db.update(job_id, status="failed", error=f"could not plan job: {e}",
                  run_dir=str(run_dir), log_path=str(log_path),
                  finished_at=time.time())
        return job_id

    db.update(job_id, steps=json.dumps(steps), run_dir=str(run_dir),
              log_path=str(log_path))
    return job_id


@app.post("/new/human")
def create_human(annotated_dir: str = Form(...), label: str = Form(""),
                 word_count: str = Form("")):
    safe_path(annotated_dir)
    jid = launch("human", label or Path(annotated_dir).name,
                 {"annotated_dir": annotated_dir, "word_count": word_count == "on"})
    return RedirectResponse(f"/job/{jid}", status_code=303)


@app.post("/new/asr")
def create_asr(audio_dir: str = Form(...), variant: str = Form(...),
               device: str = Form("cuda"), rttm_txt_dir: str = Form(""),
               label: str = Form("")):
    safe_path(audio_dir)
    if rttm_txt_dir.strip():
        safe_path(rttm_txt_dir)
    if variant not in plan.VARIANTS:
        raise HTTPException(400, "unknown variant")
    jid = launch("asr", label or f"{variant} / {Path(audio_dir).name}",
                 {"audio_dir": audio_dir, "variant": variant, "device": device,
                  "rttm_txt_dir": rttm_txt_dir})
    return RedirectResponse(f"/job/{jid}", status_code=303)


@app.post("/new/wer")
def create_wer(ref_dir: str = Form(...), hyp_dir: str = Form(...),
               language: str = Form("Spanish"), aggregate: str = Form(""),
               by_language: str = Form(""), flavor: str = Form("vtc2"),
               label: str = Form("")):
    safe_path(ref_dir)
    safe_path(hyp_dir)
    jid = launch("wer", label or f"{Path(hyp_dir).name} vs {Path(ref_dir).name}",
                 {"ref_dir": ref_dir, "hyp_dir": hyp_dir, "language": language,
                  "aggregate": aggregate == "on",
                  "by_language": by_language == "on", "flavor": flavor})
    return RedirectResponse(f"/job/{jid}", status_code=303)


@app.post("/new/langvocab")
def create_langvocab(input_dir: str = Form(...), flavor: str = Form("vtc2"),
                     ref_dir: str = Form(""), label: str = Form("")):
    safe_path(input_dir)
    if ref_dir.strip():
        safe_path(ref_dir)
    jid = launch("langvocab", label or f"{flavor} / {Path(input_dir).name}",
                 {"input_dir": input_dir, "flavor": flavor, "ref_dir": ref_dir})
    return RedirectResponse(f"/job/{jid}", status_code=303)


@app.post("/new/full")
async def create_full(request: Request):
    form = await request.form()
    audio_dir = form["audio_dir"]
    annotated_dir = form["annotated_dir"]
    safe_path(audio_dir)
    safe_path(annotated_dir)
    variants = form.getlist("variants") or ["vtc1_whisperx", "vtc2_whisperx"]
    jid = launch("full", form.get("label") or Path(audio_dir).name, {
        "audio_dir": audio_dir, "annotated_dir": annotated_dir,
        "variants": list(variants), "language": form.get("language", "Spanish"),
        "device": form.get("device", "cuda"),
    })
    return RedirectResponse(f"/job/{jid}", status_code=303)


# ─────────────────────────────────────────────────────────────────
# Job detail
# ─────────────────────────────────────────────────────────────────

@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    job = decorate(job)
    return templates.TemplateResponse(request, "job.html", {
        "request": request, "job": job, "steps": json.loads(job["steps"] or "[]"),
        "modules": MODULE_INFO,
    })


@app.get("/job/{job_id}/status", response_class=HTMLResponse)
def job_status(request: Request, job_id: int):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    job = decorate(job)
    return templates.TemplateResponse(request, "_jobstatus.html", {
        "request": request, "job": job, "steps": json.loads(job["steps"] or "[]"),
    })


@app.get("/job/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: int, tail: int = 400):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    p = Path(job["log_path"] or "")
    if not p.exists():
        return "Waiting for the job to start...\n"
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-tail:])


@app.post("/job/{job_id}/cancel")
def job_cancel(job_id: int):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] == "queued":
        db.update(job_id, status="cancelled", finished_at=time.time(),
                  error="cancelled before start")
    elif job["status"] == "running" and job["pid"]:
        try:
            # The runner starts its own session, so this reaches its children too.
            os.killpg(os.getpgid(job["pid"]), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            db.update(job_id, status="cancelled", finished_at=time.time(),
                      error="process already gone")
    return RedirectResponse(f"/job/{job_id}", status_code=303)


# ─────────────────────────────────────────────────────────────────
# Results browsing and download
# ─────────────────────────────────────────────────────────────────

@app.get("/job/{job_id}/files", response_class=HTMLResponse)
def job_files(request: Request, job_id: int, rel: str = ""):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    base = Path(job["run_dir"])
    here = run_path(job, rel)
    if not here.is_dir():
        raise HTTPException(404, "not a directory")

    dirs, files = [], []
    for e in sorted(here.iterdir(), key=lambda p: p.name.lower()):
        r = str(e.relative_to(base))
        if e.is_dir():
            dirs.append({"name": e.name, "rel": r})
        else:
            files.append({"name": e.name, "rel": r, "size": e.stat().st_size})

    parent = str(Path(rel).parent) if rel not in ("", ".") else None
    if parent == ".":
        parent = ""
    return templates.TemplateResponse(request, "_files.html", {
        "request": request, "job": job, "rel": rel, "dirs": dirs,
        "files": files, "parent": parent,
    })


@app.get("/job/{job_id}/raw")
def job_raw(job_id: int, rel: str):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    p = run_path(job, rel)
    if not p.is_file():
        raise HTTPException(404, "not a file")
    return FileResponse(p, filename=p.name)


@app.get("/job/{job_id}/view", response_class=PlainTextResponse)
def job_view(job_id: int, rel: str, limit: int = 400_000):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    p = run_path(job, rel)
    if not p.is_file():
        raise HTTPException(404, "not a file")
    if p.suffix.lower() not in {".txt", ".csv", ".json", ".log", ".rttm", ".tex", ".md"}:
        return "(binary file - use Download)"
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read(limit)


@app.get("/job/{job_id}/zip")
def job_zip(job_id: int):
    """Zip the results, skipping normalised audio and chunk intermediates -
    those are large, regenerable, and not what anyone wants to download."""
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    base = Path(job["run_dir"])
    if not base.is_dir():
        raise HTTPException(404, "run directory is gone")

    def gen():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in BULKY]
                for fn in filenames:
                    full = Path(dirpath) / fn
                    if full.is_symlink():
                        continue
                    z.write(full, full.relative_to(base))
        buf.seek(0)
        yield from buf

    name = f"{base.name}_results.zip"
    return StreamingResponse(gen(), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{name}"'})
