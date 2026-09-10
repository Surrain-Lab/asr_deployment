#!/usr/bin/env python3
"""Generate the cross-pipeline comparison workbooks for one run.

The three Excel scripts in Pipeline 3.0 take no arguments and hardcode
/home/subedi/Audio_Transcription_Pipeline3.0/output_14clips. Pointed at a run
directory they would silently read that old corpus and emit workbooks that look
right but describe different data - so they cannot be called as they are.

Rather than reimplement them (and risk the numbers drifting from published
results), this rewrites only their CONFIG assignments. The `PIPELINES` and
`OUTPUT_FILE` statements are located with `ast`, so exact line spans are
replaced and every line of real logic is preserved byte for byte.
"""
from __future__ import annotations

import argparse
import ast
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asrbench.config import cfg  # noqa: E402
from asrbench.plan import run_layout  # noqa: E402

VARIANT_LABEL = {
    "vtc1_whisperx": "VTC1 + Whisper",
    "vtc2_whisperx": "VTC2 + Whisper",
    "whisperx_only": "Only Whisper",
    "whisperx_vtc1": "Whisper + VTC1 labels",
    "whisperx_vtc2": "Whisper + VTC2 labels",
}
# Where each variant's word counts land in the run layout.
VARIANT_WC = {
    "vtc1_whisperx": ("vtc1", "word_count"),
    "vtc2_whisperx": ("vtc2", "word_count"),
    "whisperx_only": ("whisperx", "word_count"),
    "whisperx_vtc1": ("whisperx", "word_count"),
    "whisperx_vtc2": ("whisperx", "word_count"),
}


def replace_assignments(src: str, repl: dict[str, str]) -> str:
    """Swap top-level assignments by name, keeping everything else untouched."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    spans = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in repl:
                spans.append((node.lineno - 1, node.end_lineno, tgt.id))
    if not spans:
        return src
    for start, end, name in sorted(spans, reverse=True):
        lines[start:end] = [repl[name] + "\n"]
    return "".join(lines)


def build(script: Path, out_py: Path, pipelines: list[tuple[str, str]],
          out_file: Path) -> None:
    src = script.read_text()
    patched = replace_assignments(src, {
        "PIPELINES": "PIPELINES = " + repr([(n, str(p)) for n, p in pipelines]),
        "OUTPUT_FILE": "OUTPUT_FILE = " + repr(str(out_file)),
    })
    out_py.parent.mkdir(parents=True, exist_ok=True)
    out_py.write_text(patched)


def run_patched(out_py: Path, label: str) -> bool:
    print(f"\n  [{label}] running {out_py.name}")
    try:
        runpy.run_path(str(out_py), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"  !! {label} exited {e.code}", file=sys.stderr)
            return False
    except Exception as e:  # noqa: BLE001 - report and continue to the next book
        print(f"  !! {label} failed: {e!r}", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    L = run_layout(run_dir)
    reports = L["reports"]
    gen = reports / "_generated"
    reports.mkdir(parents=True, exist_ok=True)

    P = cfg.pipeline3
    wer_pipelines = [
        (VARIANT_LABEL.get(v, v), reports / f"eval_{v}")
        for v in args.variants
        if (reports / f"eval_{v}").is_dir()
    ]
    if not wer_pipelines:
        print("ERROR: no eval_report folders found for the requested variants",
              file=sys.stderr)
        return 1

    wc_pipelines = [("Human Eval", L["human"]["word_count"])]
    seen = set()
    for v in args.variants:
        key = VARIANT_WC[v]
        if key in seen:
            continue
        seen.add(key)
        d = L[key[0]][key[1]]
        if Path(d).is_dir():
            wc_pipelines.append((VARIANT_LABEL.get(v, v), d))

    print(f"Run directory : {run_dir}")
    print(f"Variants      : {', '.join(args.variants)}")
    print(f"WER sources   : {len(wer_pipelines)}")
    print(f"Word count    : {len(wc_pipelines)}")

    jobs = [
        (P / "wer_per_clip_14clips_excel.py", "wer_per_clip.py",
         wer_pipelines, reports / "wer_per_clip.xlsx", "WER per clip"),
        (P / "wer_per_recording_14clips_excel.py", "wer_per_recording.py",
         wer_pipelines, reports / "wer_per_recording.xlsx", "WER per recording"),
        (P / "word_count_comparison_14clips_excel.py", "word_count_comparison.py",
         wc_pipelines, reports / "word_count_comparison.xlsx", "Word count comparison"),
    ]

    ok = True
    for script, name, pipes, out_file, label in jobs:
        if not script.exists():
            print(f"  !! source script missing: {script}", file=sys.stderr)
            ok = False
            continue
        target = gen / name
        build(script, target, pipes, out_file)
        if not run_patched(target, label):
            ok = False
        elif out_file.exists():
            print(f"  -> {out_file}  ({out_file.stat().st_size:,} bytes)")

    print()
    print("Workbooks written to", reports)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
