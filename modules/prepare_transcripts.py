#!/usr/bin/env python3
"""Normalise an annotation corpus to <dest>/<recording>/<clip>.txt.

Same job as prepare_audio.py, for the text side. The annotation tree is not
necessarily laid out like the audio tree:

    <audio root>/<recording>/<clip>.wav                     recordings 1 deep
    <annotation root>/<wrapper>/<recording>/<clip>.txt      recordings 2 deep

and an annotation root may hold more than one wrapper folder, when a corpus was
collected or delivered in batches. transcript_converter2.py treats immediate
subdirectories as recordings, so handed such a root it would take the wrapper
names for recording names and produce a couple of mis-nested recordings instead
of the real ones.

Flattening here with symlinks means one job can cover the whole corpus however
the source happens to be wrapped. Nothing is copied and nothing is modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asrbench.discover import scan_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    src, dest = Path(args.src), Path(args.dest)
    corpus = scan_text(src)

    if not corpus.ok:
        print(f"ERROR: no .txt files found under {src}", file=sys.stderr)
        return 1

    print(f"Source          : {src}")
    print(f"Detected layout : {corpus.note()}")
    if corpus.mixed:
        print(f"Combining {len(corpus.groups)} parent folder(s) into one corpus:")
        for g in corpus.groups:
            print(f"    {g}")
    print(f"Flattened to    : {dest}")
    print()

    dest.mkdir(parents=True, exist_ok=True)
    linked = 0
    clashes: list[str] = []
    seen: dict[str, Path] = {}

    for rec in corpus.recordings:
        # A recording name appearing under two wrappers would silently overwrite
        # itself once flattened, so refuse rather than lose clips.
        if rec.name in seen and seen[rec.name] != rec.path:
            clashes.append(f"{rec.name}: {seen[rec.name]} and {rec.path}")
            continue
        seen[rec.name] = rec.path

        out_rec = dest / rec.name
        out_rec.mkdir(parents=True, exist_ok=True)
        for clip in rec.clips:
            link = out_rec / clip
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to((rec.path / clip).resolve())
            linked += 1
        print(f"  {rec.name:<28} {len(rec.clips):>3} clips")

    print()
    if clashes:
        print("ERROR: the same recording name appears under more than one parent "
              "folder, so flattening them would lose clips:", file=sys.stderr)
        for c in clashes:
            print(f"    {c}", file=sys.stderr)
        return 1

    print(f"Done: {len(seen)} recording(s), {linked} clip(s) linked into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
