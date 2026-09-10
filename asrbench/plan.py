"""Translate a module + parameters into an ordered list of executable steps.

This is the only place that knows how Pipeline 3.0's scripts fit together.
Every step is a dict:

    name  : human-readable label shown in the UI
    env   : 'main' (whisperx) | 'vtc1' (pyannote) | 'none' (no conda wrapper)
    cmd   : argv list
    cwd   : working directory, or None
    gpu   : True if the step should get a CUDA device pinned
    lock  : name of a global lock to hold, or None

Findings that shaped this file:
  * evaluation.py / combine_calculation*.py / count_words*.py are duplicated
    three times with identical code, so any copy may be used generically.
  * langdetect_tags*.py genuinely differs per pipeline, so each pipeline keeps
    calling its own copy.
  * VTC1's apply.sh always writes into the model directory, a single global
    location, so those steps take an exclusive lock and their output is
    harvested into the run directory immediately afterwards.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .config import cfg

MODULES_DIR = cfg.modules_dir
PY_MAIN = "python"  # resolved inside the conda env by `conda run`

# Pipeline variants offered by the ASR module.
VARIANTS = {
    "whisperx_only": "WhisperX only (5-min chunks, no speaker labels)",
    "whisperx_vtc1": "WhisperX + VTC1 speaker labels",
    "whisperx_vtc2": "WhisperX + VTC2 speaker labels",
    "vtc1_whisperx": "VTC1 segments -> WhisperX",
    "vtc2_whisperx": "VTC2 segments -> WhisperX",
}

SPEAKERS = ["OVERALL", "ADULT", "CHILDREN", "KCHI", "OCH", "MAL", "FEM"]


# The three language-detection scripts are NOT interchangeable: VTC1's handles
# the [SPEECH] class the others lack, and WhisperX's reads only <clip>_tagged.txt.
TAGGERS = {
    "vtc1": lambda: cfg.vtc1_dir / "langdetect_fallback" / "langdetect_tags_unmerged.py",
    "vtc2": lambda: cfg.vtc2_dir / "Langdetect_fallback" / "langdetect_tags.py",
    "whisperx": lambda: cfg.whisperx_dir / "langdetect_fallback" / "langdetect_tags.py",
}
COUNTERS = {
    "vtc1": lambda: cfg.vtc1_dir / "langdetect_fallback" / "count_words_unmerged.py",
    "vtc2": lambda: cfg.vtc2_dir / "Langdetect_fallback" / "count_words.py",
    "whisperx": lambda: cfg.whisperx_dir / "langdetect_fallback" / "count_words.py",
}


def has_langdetect(d) -> bool:
    """True if a folder already holds language-tagged output.

    wer_by_language.py reads hypotheses from *_tagged_langdetect.txt, so a raw
    ASR folder must be run through langdetect first. Checking lets the WER
    module insert that step only when it is actually needed.
    """
    path = Path(d)
    if not path.is_dir():
        return False
    return next(path.rglob("*_tagged_langdetect.txt"), None) is not None


def _step(name, cmd, env="main", cwd=None, gpu=False, lock=None, key=None):
    """One executable step.

    `key` names shared work that must happen at most once in a combined run
    (audio preparation, each VTC inference, the WhisperX chunk pass). It is an
    explicit label rather than something inferred from `name`, because matching
    on names silently skips a step the moment a name is reworded.
    """
    return {
        "name": name,
        "env": env,
        "cmd": [str(c) for c in cmd],
        "cwd": str(cwd) if cwd else None,
        "gpu": bool(gpu),
        "lock": lock,
        "key": key,
    }


def run_layout(run_dir: Path) -> dict:
    """All output paths for a run, mirroring Pipeline 3.0's outputs/ layout."""
    r = Path(run_dir)
    return {
        "audio16k": r / "audio_16k",
        "human": {
            "benchmark": r / "human_eval" / "five_mins_benchmark",
            "combined": r / "human_eval" / "combined",
            "word_count": r / "human_eval" / "word_count",
        },
        "vtc1": {
            "rttm_raw": r / "vtc1" / "rttm_raw",
            "rttm_txt": r / "vtc1" / "rttm_txt",
            "tagged": r / "vtc1" / "tagged_transcripts",
            "speaker_separated": r / "vtc1" / "speaker_separated",
            "eval_report": r / "vtc1" / "eval_report",
            "langdetect": r / "vtc1" / "langdetect",
            "word_count": r / "vtc1" / "word_count",
        },
        "vtc2": {
            "rttm_raw": r / "vtc2" / "rttm_raw",
            "rttm_txt": r / "vtc2" / "rttm_txt",
            "tagged": r / "vtc2" / "tagged_transcripts",
            "speaker_separated": r / "vtc2" / "speaker_separated",
            "eval_report": r / "vtc2" / "eval_report",
            "langdetect": r / "vtc2" / "langdetect",
            "word_count": r / "vtc2" / "word_count",
        },
        # The chunk transcription is shared, but the speaker labels are not:
        # tagging with VTC1 and with VTC2 must land in separate folders or the
        # second overwrites the first and both variants score identically.
        "whisperx": {
            "chunks": r / "whisperx_only" / "chunks",
            "raw": r / "whisperx_only" / "raw_transcripts",
            "merged": r / "whisperx_only" / "merged",
            "tagged_vtc1": r / "whisperx_only" / "tagged_vtc1",
            "tagged_vtc2": r / "whisperx_only" / "tagged_vtc2",
            "eval_report": r / "whisperx_only" / "eval_report",
            "langdetect_vtc1": r / "whisperx_only" / "langdetect_vtc1",
            "langdetect_vtc2": r / "whisperx_only" / "langdetect_vtc2",
            "word_count_vtc1": r / "whisperx_only" / "word_count_vtc1",
            "word_count_vtc2": r / "whisperx_only" / "word_count_vtc2",
        },
        "reports": r / "reports",
    }


# ─────────────────────────────────────────────────────────────────
# Shared fragments
# ─────────────────────────────────────────────────────────────────

def prepare_audio(src: Path, dest: Path) -> list:
    """Non-destructive 16 kHz mono normalisation.

    Pipeline 3.0 converted the user's WAVs in place. Here the normalised copy
    goes into the run directory instead, so pointing the UI at a shared corpus
    can never alter it. Files already 16 kHz mono are symlinked, not copied.
    """
    return [_step(
        "Prepare audio (16 kHz mono)",
        [PY_MAIN, MODULES_DIR / "prepare_audio.py", "--src", src, "--dest", dest],
        key="audio",
    )]


def vtc1_rttm(audio: Path, L: dict, device: str) -> list:
    """Run VTC1 and land its RTTM inside the run directory."""
    model = cfg.vtc1_model
    vtc1_device = "gpu" if device in ("cuda", "gpu") else "cpu"
    return [
        _step(
            "VTC1 inference (apply.sh)",
            [PY_MAIN, MODULES_DIR / "run_vtc1.py",
             "--model", model, "--audio", audio, "--device", vtc1_device],
            env="vtc1", cwd=model, gpu=(vtc1_device == "gpu"), lock="vtc1",
            key="vtc1",
        ),
        _step(
            "Harvest VTC1 RTTM into run directory",
            [PY_MAIN, MODULES_DIR / "harvest_vtc1.py",
             "--model", model, "--audio", audio, "--dest", L["vtc1"]["rttm_raw"]],
            lock="vtc1", key="vtc1",
        ),
        _step(
            "VTC1 RTTM -> timestamped text",
            [PY_MAIN, cfg.vtc1_dir / "rttm_txt_unmerged.py",
             "-i", L["vtc1"]["rttm_raw"], "-o", L["vtc1"]["rttm_txt"]],
            key="vtc1",
        ),
    ]


def vtc2_rttm(audio: Path, L: dict, device: str) -> list:
    """Run VTC2 and convert its RTTM. VTC2 writes where we tell it, so no lock."""
    model = cfg.vtc2_model
    return [
        _step(
            "VTC2 inference (infer.py)",
            ["uv", "run", model / "scripts" / "infer.py",
             "--wavs", audio,
             "--output", L["vtc2"]["rttm_raw"],
             "--config", model / "VTC-2.0" / "model" / "config.yml",
             "--checkpoint", model / "VTC-2.0" / "model" / "best.ckpt",
             "--device", device,
             "--keep_raw", "--recursive_search"],
            env="none", cwd=model, gpu=(device in ("cuda", "gpu")), key="vtc2",
        ),
        _step(
            "VTC2 RTTM -> timestamped text",
            [PY_MAIN, cfg.vtc2_dir / "rttm_txt_updated.py",
             "-i", L["vtc2"]["rttm_raw"] / "raw_rttm", "-o", L["vtc2"]["rttm_txt"]],
            key="vtc2",
        ),
    ]


# ─────────────────────────────────────────────────────────────────
# Modules
# ─────────────────────────────────────────────────────────────────

def plan_human(params: dict, run_dir: Path) -> list:
    """CHAT transcripts -> clean / per-speaker / combined reference text."""
    L = run_layout(run_dir)
    src = Path(params["annotated_dir"])
    flat = Path(run_dir) / "annotated_flat"
    H = cfg.human_dir
    steps = [
        # transcript_converter2.py treats immediate subdirectories as recordings,
        # so the source is flattened to <recording>/<clip>.txt first. That also
        # lets one job cover a corpus split across several wrapper folders.
        _step("Prepare transcripts (flatten to recording/clip)",
              [PY_MAIN, MODULES_DIR / "prepare_transcripts.py",
               "--src", src, "--dest", flat], key="annot"),
        _step("Human CHAT -> clean + per-speaker text",
              [PY_MAIN, H / "transcript_converter2.py",
               "-i", flat, "-o", L["human"]["benchmark"]]),
        _step("Combine clips -> per-recording reference",
              [PY_MAIN, H / "combine_clips2.py",
               "-i", L["human"]["benchmark"], "-o", L["human"]["combined"]]),
    ]
    if params.get("word_count", True):
        steps.append(_step(
            "Count English/Spanish words",
            [PY_MAIN, H / "count_eng_esp_final.py",
             "-i", flat, "-o", L["human"]["word_count"]]))
    return steps


def plan_asr(params: dict, run_dir: Path) -> list:
    """Audio -> transcripts, for one of the five pipeline variants."""
    L = run_layout(run_dir)
    src_audio = Path(params["audio_dir"])
    audio = L["audio16k"]
    device = params.get("device", "cuda")
    variant = params["variant"]

    steps = prepare_audio(src_audio, audio)

    if variant == "vtc1_whisperx":
        steps += vtc1_rttm(audio, L, device)
        steps += [
            _step("WhisperX on VTC1 segments",
                  [PY_MAIN, cfg.vtc1_dir / "whisperx_vtc_transcribe_unmerged.py",
                   "--audio", audio, "--rttm", L["vtc1"]["rttm_txt"],
                   "--output", L["vtc1"]["tagged"]], gpu=True),
            _step("Split VTC1 transcripts by speaker",
                  [PY_MAIN, cfg.vtc1_dir / "split_transcript_step2_unmerged.py",
                   "-i", L["vtc1"]["tagged"], "-o", L["vtc1"]["speaker_separated"]]),
        ]
        return steps

    if variant == "vtc2_whisperx":
        steps += vtc2_rttm(audio, L, device)
        steps += [
            _step("WhisperX on VTC2 segments",
                  [PY_MAIN, cfg.vtc2_dir / "whisperx_vtc2_transcrib.py",
                   "--audio", audio, "--rttm", L["vtc2"]["rttm_txt"],
                   "--output", L["vtc2"]["tagged"]], gpu=True),
            _step("Split VTC2 transcripts by speaker",
                  [PY_MAIN, cfg.vtc2_dir / "labels_seperator.py",
                   "-i", L["vtc2"]["tagged"], "-o", L["vtc2"]["speaker_separated"]]),
        ]
        return steps

    # WhisperX-chunk variants: transcribe whole chunks, then attach labels.
    steps += [
        _step("WhisperX chunk + transcribe (large-v3)",
              [PY_MAIN, cfg.whisperx_dir / "whisperx_detect_lang.py",
               "--audio", audio,
               "--chunks", L["whisperx"]["chunks"],
               "--output", L["whisperx"]["raw"]], gpu=True, key="wx_chunks"),
        _step("Merge chunk transcripts",
              [PY_MAIN, cfg.whisperx_dir / "merge_whisperx_generated_chunk_step2.py",
               "-i", L["whisperx"]["raw"], "-o", L["whisperx"]["merged"]],
              key="wx_chunks"),
    ]

    if variant == "whisperx_only":
        return steps

    # Speaker labels come from a VTC RTTM. Either reuse one the user points at,
    # or produce it here. VTC1 and VTC2 rttm_txt share an identical format, so
    # the tagger accepts either.
    src = "vtc1" if variant == "whisperx_vtc1" else "vtc2"
    reuse = params.get("rttm_txt_dir", "").strip()
    if reuse:
        speaker_dir = Path(reuse)
    elif src == "vtc1":
        steps = vtc1_rttm(audio, L, device) + steps
        speaker_dir = L["vtc1"]["rttm_txt"]
    else:
        steps = vtc2_rttm(audio, L, device) + steps
        speaker_dir = L["vtc2"]["rttm_txt"]

    steps.append(_step(
        f"Assign speaker labels from {src.upper()} RTTM timestamps",
        [PY_MAIN, cfg.whisperx_dir / "transcript_based_speaker_tagger_step3.py",
         "--speaker-dir", speaker_dir,
         "--source-dir", L["whisperx"]["merged"],
         "--output-dir", L["whisperx"][f"tagged_{src}"]],
        key=f"wx_tag_{src}"))
    return steps


def plan_wer(params: dict, run_dir: Path) -> list:
    """Reference dir + hypothesis dir -> per-clip and per-recording metrics.

    Uses VTC2_Pipeline/evaluation.py as the canonical copy: all three
    per-pipeline evaluation scripts are identical apart from trailing comments.
    """
    L = run_layout(run_dir)
    out = L["reports"] / "eval_report"
    aligned = Path(run_dir) / "hyp_aligned"
    steps = [
        # evaluation.py finds a hypothesis by joining the REFERENCE's recording
        # folder name, and the two trees do not always agree on it (a batch
        # suffix present on one side and not the other, say). Unaligned, those
        # recordings score a silent 100% WER, so this builds a symlinked view
        # of the hypothesis under the reference's names first.
        _step("Align hypothesis folder names to the reference",
              [PY_MAIN, MODULES_DIR / "align_trees.py",
               "--ref", params["ref_dir"], "--hyp", params["hyp_dir"],
               "--dest", aligned]),
        _step("Per-clip WER / CER / MER / WIL",
              [PY_MAIN, cfg.vtc2_dir / "evaluation.py",
               "--ground-truth-dir", params["ref_dir"],
               "--whisperx-dir", aligned,
               "--output-dir", out,
               "--language", params.get("language", "Spanish")]),
    ]
    if params.get("aggregate", True):
        steps.append(_step(
            "Aggregate per recording (sum counts, recompute WER)",
            [PY_MAIN, cfg.vtc2_dir / "combine_calculation_overall_final_eval.py",
             "--folder", out]))

    if params.get("by_language"):
        steps += wer_by_language_steps(
            ref_dir=Path(params["ref_dir"]),
            hyp_dir=Path(params["hyp_dir"]),
            flavor=params.get("flavor", "vtc2"),
            L=L,
        )
    return steps


def wer_by_language_steps(ref_dir: Path, hyp_dir: Path, flavor: str, L: dict) -> list:
    """English vs Spanish WER, plus the langdetect pass it depends on.

    The comparison is only meaningful on language-tagged text: the reference
    carries a (xx) tag per line and the hypothesis a trailing [xx] tag that
    langdetect adds. If the chosen hypothesis folder has already been through
    langdetect its files are used directly; otherwise the pass is inserted here
    so this works straight from a raw ASR output folder.
    """
    steps = []
    if has_langdetect(hyp_dir):
        lang_dir = hyp_dir
    else:
        lang_dir = L["reports"] / "langdetect"
        steps.append(_step(
            f"Language tag correction ({flavor}) - needed for the breakdown",
            [PY_MAIN, TAGGERS[flavor](), "-i", hyp_dir, "-o", lang_dir]))

    csv_path = L["reports"] / "wer_by_language.csv"
    steps += [
        _step("WER by language, per clip",
              [PY_MAIN, cfg.pipeline3 / "wer_by_language.py",
               "--ref-dir", ref_dir, "--hyp-dir", lang_dir,
               "--ref-tag", "paren", "--hyp-tag", "trailing",
               "--out", csv_path]),
        _step("English vs Spanish summary (corpus level)",
              [PY_MAIN, MODULES_DIR / "summarize_wer_by_language.py",
               "--in", csv_path,
               "--out", L["reports"] / "wer_by_language_summary.csv",
               "--by-recording", L["reports"] / "wer_by_language_per_recording.csv"]),
    ]
    return steps


def plan_langvocab(params: dict, run_dir: Path,
                   out_lang: Path | None = None,
                   out_count: Path | None = None) -> list:
    """Language re-detection, EN/ES word counts, and vocabulary tables."""
    L = run_layout(run_dir)
    flavor = params.get("flavor", "vtc2")
    src = Path(params["input_dir"])
    out_lang = out_lang or L["reports"] / "langdetect"
    out_count = out_count or L["reports"] / "word_count"

    steps = [
        _step(f"Language tag correction ({flavor})",
              [PY_MAIN, TAGGERS[flavor](), "-i", src, "-o", out_lang]),
        _step(f"EN/ES word counts ({flavor})",
              [PY_MAIN, COUNTERS[flavor](), "-i", out_lang, "-o", out_count]),
    ]

    # Vocabulary and per-language WER need a human reference to compare against.
    ref = params.get("ref_dir", "").strip()
    if ref:
        P = cfg.pipeline3
        steps += wer_by_language_steps(Path(ref), out_lang, flavor, L)
        steps += [
            _step("Vocabulary by clip",
                  [PY_MAIN, P / "make_vocab_table.py",
                   "--ref-dir", ref, "--hyp-dir", out_lang,
                   "--ref-tag", "paren", "--hyp-tag", "trailing",
                   "--out", L["reports"] / "vocab_by_clip.csv"]),
            _step("Vocabulary by speaker group",
                  [PY_MAIN, P / "vocab_by_speaker.py",
                   "--ref-dir", ref, "--hyp-dir", out_lang,
                   "--ref-tag", "paren", "--hyp-tag", "trailing",
                   "--out", L["reports"] / "vocab_by_speaker.csv"]),
        ]
    return steps


def plan_full(params: dict, run_dir: Path) -> list:
    """Steps 0-5 end to end for every selected variant."""
    L = run_layout(run_dir)
    variants = params.get("variants") or ["vtc1_whisperx", "vtc2_whisperx", "whisperx_vtc1"]
    language = params.get("language", "Spanish")
    device = params.get("device", "cuda")

    steps = plan_human({"annotated_dir": params["annotated_dir"]}, run_dir)

    # Shared work runs once. Variants that need the same VTC pass, the same
    # normalised audio, or the same WhisperX chunk transcription reuse it.
    emitted: set[str] = set()
    for v in variants:
        sub = plan_asr({"audio_dir": params["audio_dir"], "variant": v,
                        "device": device}, run_dir)
        for st in sub:
            key = st.get("key")
            if key and key in emitted:
                continue
            steps.append(st)
        emitted.update(st["key"] for st in sub if st.get("key"))

    # Evaluation for each variant against the human reference produced above.
    # whisperx_only stops at merged chunk text - one file per clip, with no
    # speaker separation - so there is nothing for a per-speaker evaluation to
    # read. It is transcribed but not scored.
    hyp_for = {
        "vtc1_whisperx": L["vtc1"]["speaker_separated"],
        "vtc2_whisperx": L["vtc2"]["speaker_separated"],
        "whisperx_vtc1": L["whisperx"]["tagged_vtc1"],
        "whisperx_vtc2": L["whisperx"]["tagged_vtc2"],
    }
    for v in variants:
        if v not in hyp_for:
            continue
        out = L["reports"] / f"eval_{v}"
        aligned = Path(run_dir) / "hyp_aligned" / v
        steps += [
            _step(f"Align hypothesis names - {v}",
                  [PY_MAIN, MODULES_DIR / "align_trees.py",
                   "--ref", L["human"]["benchmark"], "--hyp", hyp_for[v],
                   "--dest", aligned]),
            _step(f"WER per clip - {v}",
                  [PY_MAIN, cfg.vtc2_dir / "evaluation.py",
                   "--ground-truth-dir", L["human"]["benchmark"],
                   "--whisperx-dir", aligned,
                   "--output-dir", out, "--language", language]),
            _step(f"Aggregate per recording - {v}",
                  [PY_MAIN, cfg.vtc2_dir / "combine_calculation_overall_final_eval.py",
                   "--folder", out]),
        ]

    # Step 4: language re-detection and EN/ES word counts, per variant. The
    # comparison workbook needs these, and each pipeline needs its own tagger.
    for v in variants:
        spec = VARIANT_SOURCE.get(v)
        if spec is None:
            continue
        flavor, in_key, lang_key, count_key = spec
        tree = "whisperx" if v.startswith("whisperx_") else flavor
        steps += plan_langvocab(
            {"input_dir": str(L[tree][in_key]), "flavor": flavor}, run_dir,
            out_lang=L[tree][lang_key], out_count=L[tree][count_key])

    # Cross-pipeline comparison workbooks.
    steps.append(_step(
        "Comparison workbooks (WER + word count)",
        [PY_MAIN, MODULES_DIR / "build_reports.py",
         "--run-dir", run_dir, "--variants", *variants]))
    return steps


# Which transcripts each variant produces, and which language-detection script
# understands them. The three taggers are NOT interchangeable: VTC1's handles
# the [SPEECH] class the others lack, and WhisperX's reads only <clip>_tagged.txt.
# variant -> (langdetect flavour, transcripts key, langdetect out, counts out).
# whisperx_only is absent: it has no speaker-separated text to analyse.
VARIANT_SOURCE = {
    "vtc1_whisperx": ("vtc1", "speaker_separated", "langdetect", "word_count"),
    "vtc2_whisperx": ("vtc2", "speaker_separated", "langdetect", "word_count"),
    "whisperx_vtc1": ("whisperx", "tagged_vtc1", "langdetect_vtc1", "word_count_vtc1"),
    "whisperx_vtc2": ("whisperx", "tagged_vtc2", "langdetect_vtc2", "word_count_vtc2"),
}


PLANNERS = {
    "human": plan_human,
    "asr": plan_asr,
    "wer": plan_wer,
    "langvocab": plan_langvocab,
    "full": plan_full,
}

GPU_MODULES = {"asr", "full"}


def build(module: str, params: dict, run_dir: Path) -> list:
    if module not in PLANNERS:
        raise ValueError(f"unknown module: {module}")
    return PLANNERS[module](params, run_dir)
