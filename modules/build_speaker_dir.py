#!/usr/bin/env python3
"""Reshape VTC rttm_txt into the layout the WhisperX speaker tagger expects.

transcript_based_speaker_tagger_step3.py resolves both of its inputs the same
way:

    speaker_folder = <speaker-dir> / <recording> / <clip>
    source_folder  = <source-dir>  / <recording> / <clip>

The merged chunk transcripts are already laid out that way, but rttm_txt is not:
rttm_txt_unmerged.py keys its output by clip (<clip>/<clip>/) and
rttm_txt_updated.py by clip at one level (<clip>/). Handed either directly, the
tagger reports "Source folder not found" for every clip, writes nothing, and
exits 0 - so the WhisperX-with-labels variants come out empty with no error.

Pipeline 3.0 carries a hand-built `..._speakerdir` folder that is exactly this
reshape, so the step was known; it was just never automated. This builds the
same view with symlinks, deriving each clip's recording from the audio corpus.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asrbench.discover import scan_audio  # noqa: E402

MARKER = "_all_speakers.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm-txt", required=True, help="VTC rttm_txt directory")
    ap.add_argument("--audio", required=True, help="normalised audio, for clip -> recording")
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    rttm = Path(args.rttm_txt)
    dest = Path(args.dest)

    corpus = scan_audio(args.audio)
    if not corpus.ok:
        print(f"ERROR: no audio under {args.audio}", file=sys.stderr)
        return 1
    clip_to_rec = {
        Path(clip).stem: rec.name
        for rec in corpus.recordings for clip in rec.clips
    }

    found = sorted(rttm.rglob("*" + MARKER))
    if not found:
        print(f"ERROR: no *{MARKER} under {rttm}", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    linked, unknown = 0, []

    for f in found:
        clip = f.name[: -len(MARKER)]
        rec = clip_to_rec.get(clip)
        if rec is None:
            unknown.append(clip)
            continue
        out = dest / rec / clip
        out.mkdir(parents=True, exist_ok=True)
        link = out / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f.resolve())
        linked += 1

    print(f"Source     : {rttm}")
    print(f"Reshaped to: {dest}   (<recording>/<clip>/)")
    print(f"  {linked} clip timestamp file(s) linked")
    if unknown:
        print(f"  {len(unknown)} clip(s) had no matching audio and were skipped: "
              f"{', '.join(unknown[:5])}")
    return 0 if linked else 1


if __name__ == "__main__":
    sys.exit(main())
