"""Preflight check: is this machine actually able to run the pipeline?

ASR_Benchmark wraps the Audio_Transcription_Pipeline3.0 scripts and the VTC1 /
VTC2 model checkouts rather than containing them. None of that travels with a
git clone, so on a new server the usual failure is a job that dies several steps
in with a FileNotFoundError buried in a log.

This checks everything up front and says exactly what is missing.

    python -m asrbench.doctor            # report, exit 1 if anything is broken
    python -m asrbench.doctor --suggest  # print detected values for config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}

# Scripts wrapped from Pipeline 3.0, relative to its root. A missing one breaks
# the module named alongside it.
REQUIRED_SCRIPTS = {
    "Human_Evaluation/Human_Evaluation2.0/transcript_converter2.py": "human cleanup",
    "Human_Evaluation/Human_Evaluation2.0/combine_clips2.py": "human cleanup",
    "Human_Evaluation/Human_Evaluation2.0/count_eng_esp_final.py": "human word counts",
    "VTC1_Pipeline/rttm_txt_unmerged.py": "VTC1",
    "VTC1_Pipeline/whisperx_vtc_transcribe_unmerged.py": "VTC1",
    "VTC1_Pipeline/split_transcript_step2_unmerged.py": "VTC1",
    "VTC1_Pipeline/langdetect_fallback/langdetect_tags_unmerged.py": "VTC1 language",
    "VTC1_Pipeline/langdetect_fallback/count_words_unmerged.py": "VTC1 language",
    "VTC2_Pipeline/rttm_txt_updated.py": "VTC2",
    "VTC2_Pipeline/whisperx_vtc2_transcrib.py": "VTC2",
    "VTC2_Pipeline/labels_seperator.py": "VTC2",
    "VTC2_Pipeline/evaluation.py": "WER evaluation",
    "VTC2_Pipeline/combine_calculation_overall_final_eval.py": "WER aggregation",
    "VTC2_Pipeline/Langdetect_fallback/langdetect_tags.py": "VTC2 language",
    "VTC2_Pipeline/Langdetect_fallback/count_words.py": "VTC2 language",
    "Only_Whisperx/whisperx_detect_lang.py": "WhisperX",
    "Only_Whisperx/merge_whisperx_generated_chunk_step2.py": "WhisperX",
    "Only_Whisperx/transcript_based_speaker_tagger_step3.py": "WhisperX speaker labels",
    "Only_Whisperx/langdetect_fallback/langdetect_tags.py": "WhisperX language",
    "Only_Whisperx/langdetect_fallback/count_words.py": "WhisperX language",
    "transcript_tags.py": "vocabulary / language tables",
    "wer_by_language.py": "WER by language",
    "make_vocab_table.py": "vocabulary tables",
    "vocab_by_speaker.py": "vocabulary tables",
    "wer_per_clip_14clips_excel.py": "comparison workbooks",
    "wer_per_recording_14clips_excel.py": "comparison workbooks",
    "word_count_comparison_14clips_excel.py": "comparison workbooks",
}

SYSTEM_TOOLS = {
    "ffmpeg": ("audio conversion", True),
    "ffprobe": ("audio format probing", True),
    "sox": ("required by VTC1's apply.sh", True),
    "uv": ("runs VTC2 inference", True),
    "git-lfs": ("VTC2 model weights", False),
}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, section: str, status: str, name: str, detail: str = "") -> None:
        self.rows.append((section, status, name, detail))

    def counts(self) -> dict[str, int]:
        out = {OK: 0, WARN: 0, FAIL: 0}
        for _, st, _, _ in self.rows:
            out[st] += 1
        return out

    def render(self) -> str:
        lines = []
        section = None
        for sec, st, name, detail in self.rows:
            if sec != section:
                lines.append("")
                lines.append(sec)
                lines.append("-" * 74)
                section = sec
            line = f"[{MARK[st]}] {name}"
            if detail:
                line += f"\n           {detail}"
            lines.append(line)
        c = self.counts()
        lines.append("")
        lines.append("=" * 74)
        lines.append(f"  {c[OK]} ok, {c[WARN]} warning(s), {c[FAIL]} failure(s)")
        lines.append("=" * 74)
        return "\n".join(lines)


def conda_envs(conda: Path) -> dict[str, Path]:
    try:
        out = subprocess.run([str(conda), "env", "list", "--json"],
                             capture_output=True, text=True, timeout=90)
        return {Path(p).name: Path(p) for p in json.loads(out.stdout).get("envs", [])}
    except Exception:  # noqa: BLE001
        return {}


def check(cfg) -> Report:
    r = Report()

    # --- system tools -------------------------------------------------
    for tool, (why, required) in SYSTEM_TOOLS.items():
        path = shutil.which(tool)
        if path:
            r.add("System tools", OK, tool, path)
        else:
            r.add("System tools", FAIL if required else WARN, tool,
                  f"not on PATH - needed for {why}")

    # --- conda and environments ---------------------------------------
    if not cfg.conda.exists():
        r.add("Conda", FAIL, "conda", f"not found at {cfg.conda} (paths.conda)")
        envs = {}
    else:
        r.add("Conda", OK, "conda", str(cfg.conda))
        envs = conda_envs(cfg.conda)

    for label, name in (("main (WhisperX, eval, Excel)", cfg.env_main),
                        ("vtc1 (pyannote)", cfg.env_vtc1)):
        if name in envs:
            r.add("Conda", OK, f"env '{name}'  - {label}", str(envs[name]))
        else:
            r.add("Conda", FAIL, f"env '{name}'  - {label}",
                  "environment not found; create it or fix envs: in config.yaml")

    # --- Pipeline 3.0 scripts -----------------------------------------
    if not cfg.pipeline3.is_dir():
        r.add("Pipeline 3.0", FAIL, "pipeline root",
              f"{cfg.pipeline3} does not exist (paths.pipeline3)")
    else:
        r.add("Pipeline 3.0", OK, "pipeline root", str(cfg.pipeline3))
        missing: dict[str, list[str]] = {}
        for rel, feature in REQUIRED_SCRIPTS.items():
            if not (cfg.pipeline3 / rel).exists():
                missing.setdefault(feature, []).append(rel)
        if not missing:
            r.add("Pipeline 3.0", OK,
                  f"all {len(REQUIRED_SCRIPTS)} wrapped scripts present")
        else:
            for feature, rels in sorted(missing.items()):
                r.add("Pipeline 3.0", FAIL, f"missing - breaks {feature}",
                      "; ".join(rels))

    # --- models --------------------------------------------------------
    apply_sh = cfg.vtc1_model / "apply.sh"
    if apply_sh.exists():
        r.add("Models", OK, "VTC1", str(cfg.vtc1_model))
    else:
        r.add("Models", FAIL, "VTC1",
              f"apply.sh not found under {cfg.vtc1_model}\n"
              f"           clone https://github.com/MarvinLvn/voice-type-classifier")

    infer = cfg.vtc2_model / "scripts" / "infer.py"
    ckpt = cfg.vtc2_model / "VTC-2.0" / "model" / "best.ckpt"
    conf = cfg.vtc2_model / "VTC-2.0" / "model" / "config.yml"
    if infer.exists() and ckpt.exists() and conf.exists():
        size = ckpt.stat().st_size
        if size < 1_000_000:
            r.add("Models", FAIL, "VTC2",
                  f"best.ckpt is only {size:,} bytes - looks like an unfetched "
                  f"git-lfs pointer. Run: git lfs pull")
        else:
            r.add("Models", OK, "VTC2", f"{cfg.vtc2_model} (ckpt {size / 1e6:.0f} MB)")
    else:
        gone = [str(p) for p in (infer, conf, ckpt) if not p.exists()]
        r.add("Models", FAIL, "VTC2", "missing: " + "; ".join(gone))

    # --- storage -------------------------------------------------------
    try:
        cfg.runs_root.mkdir(parents=True, exist_ok=True)
        probe = cfg.runs_root / ".asrbench_write_test"
        probe.write_text("ok")
        probe.unlink()
        usage = shutil.disk_usage(cfg.runs_root)
        free_gb = usage.free / 1e9
        status = OK if free_gb > 20 else WARN
        r.add("Storage", status, "runs directory writable",
              f"{cfg.runs_root} ({free_gb:,.0f} GB free)")
    except Exception as e:  # noqa: BLE001
        r.add("Storage", FAIL, "runs directory", f"{cfg.runs_root}: {e}")

    for root in cfg.browse_roots:
        if root.is_dir():
            r.add("Storage", OK, f"browse root {root}")
        else:
            r.add("Storage", WARN, f"browse root {root}", "does not exist")

    # --- GPU -----------------------------------------------------------
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            gpus = [l for l in out.splitlines() if l.strip()]
            r.add("GPU", OK, f"{len(gpus)} GPU(s) detected", "; ".join(gpus))
            configured = set(cfg.gpus)
            available = {int(l.split(",")[0]) for l in gpus}
            bad = configured - available
            if bad:
                r.add("GPU", FAIL, "scheduler.gpus mismatch",
                      f"config lists {sorted(configured)} but only "
                      f"{sorted(available)} exist")
        except Exception as e:  # noqa: BLE001
            r.add("GPU", WARN, "nvidia-smi failed", str(e))
    else:
        r.add("GPU", WARN, "no nvidia-smi",
              "CPU-only machine; transcription will be very slow")

    return r


def suggest() -> int:
    """Print values detected on this machine, for filling in config.yaml."""
    print("# Detected on this machine - copy into config.yaml\n")
    conda = shutil.which("conda") or ""
    if not conda:
        for c in (Path.home() / "miniconda3/bin/conda",
                  Path.home() / "anaconda3/bin/conda",
                  Path.home() / "miniforge3/bin/conda"):
            if c.exists():
                conda = str(c)
                break
    print("paths:")
    print(f"  conda:      {conda or '<conda not found>'}")

    guesses = [Path.home() / "Audio_Transcription_Pipeline3.0",
               Path.cwd().parent / "Audio_Transcription_Pipeline3.0"]
    found = next((g for g in guesses if g.is_dir()), None)
    print(f"  pipeline3:  {found or '<path to Audio_Transcription_Pipeline3.0>'}")

    vtc1 = vtc2 = None
    if found:
        for base in (found, found.parent):
            c1 = base / "VTC1_Pipeline" / "voice_type_classifier"
            if (c1 / "apply.sh").exists():
                vtc1 = c1
            for c2 in base.glob("**/VTC/VTC"):
                if (c2 / "scripts" / "infer.py").exists():
                    vtc2 = c2
                    break
    print(f"  vtc1_model: {vtc1 or '<path to voice_type_classifier>'}")
    print(f"  vtc2_model: {vtc2 or '<path to VTC checkout>'}")
    print(f"  runs_root:  {Path.home() / 'asrbench_runs'}")

    if conda:
        envs = conda_envs(Path(conda))
        print("\nenvs:")
        main = "whisperx" if "whisperx" in envs else "<env with whisperx installed>"
        v1 = "pyannote" if "pyannote" in envs else "<env with pyannote-audio>"
        print(f"  main: {main}")
        print(f"  vtc1: {v1}")
        print(f"\n# environments present: {', '.join(sorted(envs)) or 'none'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Check this machine can run ASR_Benchmark")
    ap.add_argument("--suggest", action="store_true",
                    help="print detected values for config.yaml instead of checking")
    args = ap.parse_args()

    if args.suggest:
        return suggest()

    try:
        from .config import cfg
    except FileNotFoundError:
        print("No config.yaml found.\n\n"
              "  cp config.example.yaml config.yaml\n"
              "  python -m asrbench.doctor --suggest   # detects paths for you\n",
              file=sys.stderr)
        return 2

    print("=" * 74)
    print("  ASR_Benchmark preflight")
    print("=" * 74)
    print(f"  host   : {os.uname().nodename}")
    print(f"  config : {Path(os.environ.get('ASRBENCH_CONFIG', 'config.yaml')).resolve()}")

    r = check(cfg)
    print(r.render())

    c = r.counts()
    if c[FAIL]:
        print("\nNot ready. Fix the FAIL lines above, then run this again.")
        print("Tip: `python -m asrbench.doctor --suggest` prints detected paths.")
        return 1
    if c[WARN]:
        print("\nUsable, with warnings noted above.")
    else:
        print("\nAll checks passed. Start with: bin/start.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
