#!/usr/bin/env python3
"""Run VTC1's apply.sh over every recording folder, with a staleness guard.

apply.sh always writes to <model>/output_voice_type_classifier/<recording_name>/,
a single global location keyed only by folder name. That means a previous run's
output for a same-named recording is already sitting there, and a silently
failing apply.sh would leave it in place for us to harvest as though it were
fresh - wrong numbers with no error.

So we record the time before each recording runs and afterwards require
all.rttm to be newer. Nothing is deleted; stale output is reported and fails
the step instead of being trusted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asrbench.discover import scan_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()

    model = Path(args.model)
    apply_sh = model / "apply.sh"
    if not apply_sh.exists():
        print(f"ERROR: apply.sh not found at {apply_sh}", file=sys.stderr)
        return 1

    corpus = scan_audio(args.audio)
    if not corpus.ok:
        print(f"ERROR: no .wav files under {args.audio}", file=sys.stderr)
        return 1

    out_base = model / "output_voice_type_classifier"
    print(f"VTC1 model  : {model}")
    print(f"Device      : {args.device}")
    print(f"Recordings  : {corpus.n_recordings} ({corpus.n_clips} clips)")
    print()

    failures = []
    for i, rec in enumerate(corpus.recordings, 1):
        print(f"[{i}/{corpus.n_recordings}] {rec.name}  ({len(rec.clips)} clips)")
        started = time.time()

        r = subprocess.run(
            ["bash", str(apply_sh), str(rec.path), f"--device={args.device}"],
            cwd=str(model), text=True,
        )
        if r.returncode != 0:
            print(f"  !! apply.sh exited {r.returncode} for {rec.name}")
            failures.append(rec.name)
            continue

        all_rttm = out_base / rec.name / "all.rttm"
        if not all_rttm.exists():
            print(f"  !! no all.rttm produced at {all_rttm}")
            failures.append(rec.name)
            continue
        if all_rttm.stat().st_mtime < started:
            print(f"  !! {all_rttm} is STALE (older than this run) - refusing to use it")
            failures.append(rec.name)
            continue

        size = all_rttm.stat().st_size
        print(f"  ok  all.rttm  {size:,} bytes")

    print()
    if failures:
        print(f"FAILED for {len(failures)} recording(s): {', '.join(failures)}",
              file=sys.stderr)
        return 1
    print(f"VTC1 complete for all {corpus.n_recordings} recording(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
