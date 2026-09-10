#!/usr/bin/env python3
"""Roll the per-clip language WER table up to corpus level.

wer_by_language.py emits one row per clip per language, which does not answer
"what is my Spanish WER?". Averaging those rows would be wrong - a 10-word clip
would count as much as a 500-word one - so this sums the raw counts
(substitutions, deletions, insertions, reference words) and recomputes WER from
the totals, matching how the pipeline aggregates per recording.

Writes a summary CSV and prints a table into the job log.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

LANG_LABEL = {"en": "English", "es": "Spanish", "unknown": "Undetermined",
              "both": "Mixed EN/ES"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="wer_by_language.csv")
    ap.add_argument("--out", required=True, help="summary CSV to write")
    ap.add_argument("--by-recording", default="", help="optional per-recording CSV")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    overall: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_rec: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    counted = skipped = 0

    with open(src, newline="") as f:
        for row in csv.DictReader(f):
            lang = (row.get("language") or "").strip()
            if not lang:
                continue
            try:
                ref = int(row["ref_words"])
                sub = int(row["substitutions"])
                dele = int(row["deletions"])
                ins = int(row["insertions"])
                hits = int(row["hits"])
                hyp = int(row.get("hyp_words") or 0)
            except (KeyError, ValueError):
                skipped += 1
                continue

            # Clips with a reference but no hypothesis at all are usually a
            # mismatched pair of folders rather than a real 100% error, so they
            # are counted separately instead of silently inflating WER.
            for bucket in (overall[lang], per_rec[(row.get("recording", ""), lang)]):
                bucket["ref_words"] += ref
                bucket["hyp_words"] += hyp
                bucket["substitutions"] += sub
                bucket["deletions"] += dele
                bucket["insertions"] += ins
                bucket["hits"] += hits
                bucket["clips"] += 1
                if hyp == 0 and ref > 0:
                    bucket["empty_hyp_clips"] += 1
            counted += 1

    if not overall:
        print("ERROR: no usable rows found", file=sys.stderr)
        return 1

    def wer(b: dict[str, int]) -> float:
        n = b["ref_words"]
        return (b["substitutions"] + b["deletions"] + b["insertions"]) / n if n else 0.0

    fields = ["language", "clips", "ref_words", "hyp_words", "hits",
              "substitutions", "deletions", "insertions", "wer", "accuracy",
              "empty_hyp_clips"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for lang in sorted(overall, key=lambda k: -overall[k]["ref_words"]):
            b = overall[lang]
            e = wer(b)
            w.writerow({
                "language": lang, "clips": b["clips"], "ref_words": b["ref_words"],
                "hyp_words": b["hyp_words"], "hits": b["hits"],
                "substitutions": b["substitutions"], "deletions": b["deletions"],
                "insertions": b["insertions"], "wer": round(e, 4),
                "accuracy": round(1 - e, 4), "empty_hyp_clips": b["empty_hyp_clips"],
            })

    if args.by_recording:
        rec_out = Path(args.by_recording)
        rec_out.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["recording", *fields])
            w.writeheader()
            for (rec, lang) in sorted(per_rec):
                b = per_rec[(rec, lang)]
                e = wer(b)
                w.writerow({
                    "recording": rec, "language": lang, "clips": b["clips"],
                    "ref_words": b["ref_words"], "hyp_words": b["hyp_words"],
                    "hits": b["hits"], "substitutions": b["substitutions"],
                    "deletions": b["deletions"], "insertions": b["insertions"],
                    "wer": round(e, 4), "accuracy": round(1 - e, 4),
                    "empty_hyp_clips": b["empty_hyp_clips"],
                })

    # Printed into the job log so the answer is visible without opening a file.
    ordered = sorted(overall, key=lambda k: -overall[k]["ref_words"])
    # A language the hypothesis can never emit produces 100% WER by construction,
    # not by error. langdetect only ever tags en/es, while the human reference
    # also uses "unknown" and "both", so those rows are reported separately
    # instead of being mistaken for a result.
    comparable = [k for k in ordered if overall[k]["hyp_words"] > 0]
    hollow = [k for k in ordered if overall[k]["hyp_words"] == 0]

    def row(lang: str) -> str:
        b = overall[lang]
        e = wer(b)
        return (f"{LANG_LABEL.get(lang, lang):<14}{b['clips']:>7}{b['ref_words']:>11,}"
                f"{b['hyp_words']:>11,}{e:>8.1%}{1 - e:>10.1%}"
                f"{b['substitutions']:>7,}{b['deletions']:>7,}{b['insertions']:>7,}")

    head = (f"{'Language':<14}{'Clips':>7}{'Ref words':>11}{'Hyp words':>11}"
            f"{'WER':>8}{'Accuracy':>10}{'Sub':>7}{'Del':>7}{'Ins':>7}")

    print()
    print("WER BY LANGUAGE  (corpus level: counts summed, WER recomputed)")
    print("=" * 88)
    print(head)
    print("-" * 88)
    for lang in comparable:
        print(row(lang))
    print("=" * 88)

    if hollow:
        print()
        print("NOT COMPARABLE - reference-only language tags")
        print("-" * 88)
        for lang in hollow:
            print(row(lang))
        total_hollow = sum(overall[k]["ref_words"] for k in hollow)
        total_ref = sum(b["ref_words"] for b in overall.values())
        pct = total_hollow / total_ref if total_ref else 0
        print(f"\nThese rows show 100% WER because the hypothesis contains no words "
              f"tagged\n{' or '.join(repr(k) for k in hollow)} at all - langdetect emits "
              f"only 'en' and 'es', while the human\nreference also uses these tags. "
              f"It is a tagging mismatch, not transcription error.\n"
              f"{total_hollow:,} reference words ({pct:.1%} of the corpus) fall here and are "
              f"excluded above.")

    # Language identification is a separate question from transcription accuracy,
    # and mixing them up is easy: a word put in the wrong language bucket counts
    # as a deletion in one and an insertion in the other.
    ref_en = overall.get("en", {}).get("ref_words", 0)
    ref_es = overall.get("es", {}).get("ref_words", 0)
    hyp_en = overall.get("en", {}).get("hyp_words", 0)
    hyp_es = overall.get("es", {}).get("hyp_words", 0)
    if (ref_en + ref_es) and (hyp_en + hyp_es):
        r_pct = ref_en / (ref_en + ref_es)
        h_pct = hyp_en / (hyp_en + hyp_es)
        print()
        print("LANGUAGE IDENTIFICATION  (English share of EN+ES words)")
        print("-" * 88)
        print(f"  human reference : {r_pct:>6.1%} English  "
              f"({ref_en:,} en / {ref_es:,} es)")
        print(f"  this pipeline   : {h_pct:>6.1%} English  "
              f"({hyp_en:,} en / {hyp_es:,} es)")
        if r_pct > 0:
            ratio = h_pct / r_pct
            direction = "over" if ratio > 1 else "under"
            print(f"  -> the pipeline {direction}-predicts English by {ratio:.1f}x")
        print("\n  Per-language WER mixes two errors: getting words wrong, and putting\n"
              "  the right words in the wrong language bucket. A misidentified word is\n"
              "  a deletion in one language and an insertion in the other, so compare\n"
              "  these shares before reading much into the per-language WER above.")

    empties = sum(b["empty_hyp_clips"] for b in overall.values())
    if empties:
        print(f"\nNote: {empties} clip/language row(s) had a reference but an empty "
              f"hypothesis.\nSome of those are the not-comparable rows above; the rest "
              f"mean the two folders\ndo not cover the same clips, which inflates WER. "
              f"Check clip counts on both sides.")

    print(f"\nSummary written to {out}")
    if args.by_recording:
        print(f"Per recording written to {args.by_recording}")
    print(f"{counted} row(s) counted" + (f", {skipped} skipped" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
