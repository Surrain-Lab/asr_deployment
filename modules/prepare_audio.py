#!/usr/bin/env python3
"""Normalise a corpus to <dest>/<recording>/<clip>.wav at 16 kHz mono.

Two jobs in one:

1. Flatten the nesting. The corpus is not uniformly deep - audio sits at
   audio/<recording>/<clip>.wav but annotations at
   annotated-text/<wrapper>/<recording>/<clip>.txt. Pipeline 3.0's scripts all
   assume recordings are immediate subdirectories, so we normalise to that shape
   here and everything downstream is spared the problem.

2. Guarantee 16 kHz mono. VTC1 hard-fails otherwise.

Pipeline 3.0 converted the user's WAVs in place. This writes into the run
directory instead, so a shared corpus is never modified. Files already in the
right format are symlinked rather than copied, which costs no disk and no time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asrbench.discover import scan_audio  # noqa: E402


def probe(wav: Path) -> tuple[str, str] | None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", str(wav)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    parts = r.stdout.strip().split(",")
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    src, dest = Path(args.src), Path(args.dest)
    corpus = scan_audio(src)

    if not corpus.ok:
        print(f"ERROR: no .wav files found under {src}", file=sys.stderr)
        return 1

    print(f"Source            : {src}")
    print(f"Detected layout   : {corpus.note()}")
    print(f"Recording level   : {corpus.effective_root}")
    print(f"Normalised output : {dest}")
    print()

    dest.mkdir(parents=True, exist_ok=True)
    linked = converted = failed = 0

    for rec in corpus.recordings:
        out_rec = dest / rec.name
        out_rec.mkdir(parents=True, exist_ok=True)
        for clip in rec.clips:
            src_wav = rec.path / clip
            out_wav = out_rec / clip
            if out_wav.exists() or out_wav.is_symlink():
                out_wav.unlink()

            fmt = probe(src_wav)
            if fmt is None:
                print(f"  !! could not probe {src_wav} - skipped")
                failed += 1
                continue

            if fmt == ("16000", "1"):
                out_wav.symlink_to(src_wav.resolve())
                linked += 1
                continue

            print(f"  converting {rec.name}/{clip}  ({fmt[0]} Hz, {fmt[1]}ch -> 16000 Hz, 1ch)")
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(src_wav), "-ar", "16000", "-ac", "1", str(out_wav)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"  !! ffmpeg failed for {src_wav}:\n{r.stderr}", file=sys.stderr)
                out_wav.unlink(missing_ok=True)
                failed += 1
                continue
            converted += 1

    print()
    print(f"Done: {linked} already 16 kHz mono (symlinked), "
          f"{converted} converted, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
