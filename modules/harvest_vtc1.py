#!/usr/bin/env python3
"""Copy this run's VTC1 RTTM out of the shared model directory into the run dir.

Everything after this point reads from the run directory, so concurrent or later
jobs cannot contaminate each other through the model's global output folder.
Only the recordings belonging to this run are taken.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asrbench.discover import scan_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    out_base = Path(args.model) / "output_voice_type_classifier"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    corpus = scan_audio(args.audio)
    missing = []
    taken = 0

    for rec in corpus.recordings:
        src = out_base / rec.name
        if not (src / "all.rttm").exists():
            missing.append(rec.name)
            continue
        tgt = dest / rec.name
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(src, tgt)
        taken += 1
        print(f"  harvested {rec.name}")

    print()
    if missing:
        print(f"ERROR: no VTC1 output for {len(missing)} recording(s): "
              f"{', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"Harvested {taken} recording(s) into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
