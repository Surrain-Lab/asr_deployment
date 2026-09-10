#!/usr/bin/env python3
"""Build a hypothesis tree whose recording folders match the reference's names.

evaluation.py locates a hypothesis file by joining the *reference's* recording
folder name:

    <hyp>/<recording from ref>/<clip>/<clip>_<SPEAKER>_clean.txt

but the two trees do not always agree on that name. In this corpus the human
reference calls three recordings `DL-1522001002_pre_1`, `DL-3923001002_pre_1`
and `DL-5624002008_pre_1` while the pipeline outputs call them
`DL-1522001002_pre`, `DL-3923001002_pre` and `DL-5624002008_pre` - the clip
folders inside are named identically on both sides. The project's own
transcript_tags.canonical_key() documents this and sidesteps it, which is why
wer_by_language.py is unaffected; evaluation.py has no such protection and
simply finds nothing, scoring a silent 100% WER for every affected clip.

This builds a symlinked view of the hypothesis under the reference's names, so
evaluation.py finds what it should. Nothing is copied, renamed or deleted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asrbench.discover import scan_text  # noqa: E402


def clip_index(root: Path) -> dict[str, tuple[str, Path]]:
    """clip folder name -> (recording folder name, clip folder path)."""
    out: dict[str, tuple[str, Path]] = {}
    corpus = scan_text(root)
    for rec in corpus.recordings:
        for clip in rec.clips:
            out[clip] = (rec.name, rec.path / clip)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    ref_idx = clip_index(Path(args.ref))
    hyp_idx = clip_index(Path(args.hyp))
    dest = Path(args.dest)

    if not ref_idx:
        print(f"ERROR: no clips found under reference {args.ref}", file=sys.stderr)
        return 1
    if not hyp_idx:
        print(f"ERROR: no clips found under hypothesis {args.hyp}", file=sys.stderr)
        return 1

    renames: dict[str, str] = {}
    linked = 0
    unmatched: list[str] = []

    dest.mkdir(parents=True, exist_ok=True)

    for clip, (hyp_rec, hyp_path) in sorted(hyp_idx.items()):
        entry = ref_idx.get(clip)
        if entry is None:
            unmatched.append(clip)
            # Keep it under its own name so nothing is quietly dropped.
            ref_rec = hyp_rec
        else:
            ref_rec = entry[0]
            if ref_rec != hyp_rec:
                renames[hyp_rec] = ref_rec

        out_rec = dest / ref_rec
        out_rec.mkdir(parents=True, exist_ok=True)
        link = out_rec / clip
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(hyp_path.resolve(), target_is_directory=True)
        linked += 1

    print(f"Reference  : {args.ref}")
    print(f"Hypothesis : {args.hyp}")
    print(f"Aligned to : {dest}")
    print()
    print(f"  reference clips  : {len(ref_idx)}")
    print(f"  hypothesis clips : {len(hyp_idx)}")
    print(f"  linked           : {linked}")

    if renames:
        print()
        print(f"  Recording folders renamed to match the reference ({len(renames)}):")
        for a, b in sorted(renames.items()):
            print(f"    {a}  ->  {b}")
        print("  Without this, evaluation.py would have found no hypothesis for these")
        print("  recordings and scored them 100% WER.")
    else:
        print("\n  Recording folder names already agree; nothing renamed.")

    missing_hyp = sorted(set(ref_idx) - set(hyp_idx))
    if missing_hyp:
        print()
        print(f"  {len(missing_hyp)} reference clip(s) have no hypothesis at all "
              f"(genuinely absent from this pipeline):")
        for c in missing_hyp[:10]:
            print(f"    {c}")
        if len(missing_hyp) > 10:
            print(f"    ... and {len(missing_hyp) - 10} more")

    if unmatched:
        print()
        print(f"  {len(unmatched)} hypothesis clip(s) not present in the reference "
              f"(kept under their original recording name).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
