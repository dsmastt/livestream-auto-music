#!/usr/bin/env python3
"""Evaluate predicted mood labels against manual ground truth.

Ground truth format (one CSV per VOD, named <vod_id>.csv in the GT dir):

    timestamp,label
    12.5,hype
    75,chill
    01:02:30,tense

`timestamp` is seconds (float) or HH:MM:SS. `label` must be one of:
hype, comedic, chill, waiting, tense.

Matching rule: for each GT row we take the result window with a NON-EMPTY
committed_label whose start_time is nearest to the GT timestamp, if it falls
within --tolerance seconds. GT rows with no committed window nearby are
counted as "no-prediction" (reported separately, excluded from agreement %).

Usage:
    python evaluate.py --results-dir results/ --gt-dir gt/
    python evaluate.py --results-dir results/ --gt-dir gt/ --vod-ids 123,456
    python evaluate.py --results-dir results/ --gt-dir gt/ --tolerance 10
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

LABELS = ["hype", "comedic", "chill", "waiting", "tense"]
_TS_RE = re.compile(r"^(\d+):([0-5]?\d):([0-5]?\d)$")


def parse_timestamp(value: str) -> float:
    value = value.strip()
    m = _TS_RE.match(value)
    if m:
        h, mi, s = (int(g) for g in m.groups())
        return h * 3600 + mi * 60 + s
    return float(value)


def load_gt(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            ts = parse_timestamp(row["timestamp"])
            label = (row.get("label") or "").strip().lower()
            if label not in LABELS:
                print(f"  WARN {path.name}:{line_no}: unknown label {label!r} -- skipped")
                continue
            rows.append((ts, label))
    return rows


def load_committed(path: Path) -> list[dict]:
    windows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w = json.loads(line)
            if w.get("committed_label"):
                windows.append({
                    "start": float(w["start_time"]),
                    "end": float(w["end_time"]),
                    "label": w["committed_label"],
                })
    return windows


def match(gt: list[tuple[float, str]], windows: list[dict],
          tolerance: float) -> tuple[list[tuple[str, str]], int, int]:
    """Return (list of (gt_label, pred_label) pairs, matched, no_prediction)."""
    pairs: list[tuple[str, str]] = []
    no_pred = 0
    for ts, gt_label in gt:
        best, best_d = None, None
        for w in windows:
            # Distance to the window MIDPOINT: with 15s/5s overlapping windows,
            # start-distance would match GT timestamps to a window that does
            # not contain them.
            d = abs((w["start"] + w["end"]) / 2 - ts)
            if d <= tolerance and (best_d is None or d < best_d):
                best, best_d = w["label"], d
        if best is None:
            no_pred += 1
        else:
            pairs.append((gt_label, best))
    return pairs, len(pairs), no_pred


def confusion_matrix(pairs: list[tuple[str, str]]) -> Counter:
    return Counter(pairs)


def print_matrix(matrix: Counter, header: str) -> None:
    print(f"\n{header}")
    labels = [l for l in LABELS if any(g == l or p == l for g, p in matrix)]
    if not labels:
        print("  (no matched predictions)")
        return
    width = max(len(l) for l in labels) + 1
    print("  " + " " * width + "".join(f"{l:>9}" for l in labels))
    for gt in labels:
        row = "".join(f"{matrix.get((gt, p), 0):>9}" for p in labels)
        print(f"  {gt:<{width}}{row}")
    print("  (rows = ground truth, cols = predicted)")


def evaluate_one(results_path: Path, gt_path: Path, tolerance: float) -> dict:
    gt = load_gt(gt_path)
    windows = load_committed(results_path)
    pairs, matched, no_pred = match(gt, windows, tolerance)
    matrix = confusion_matrix(pairs)
    correct = sum(1 for g, p in pairs if g == p)
    agreement = (100.0 * correct / matched) if matched else 0.0
    return {
        "vod_id": results_path.stem,
        "n_gt": len(gt),
        "n_committed_windows": len(windows),
        "matched": matched,
        "no_prediction": no_pred,
        "correct": correct,
        "agreement_pct": agreement,
        "matrix": matrix,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate vibe predictions vs ground truth.")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--gt-dir", required=True,
                   help="Dir with <vod_id>.csv ground-truth files")
    p.add_argument("--vod-ids", default=None,
                   help="Comma-separated VOD IDs to evaluate (default: all "
                        "results files with a matching GT CSV)")
    p.add_argument("--tolerance", type=float, default=15.0,
                   help="Max seconds between GT timestamp and window start "
                        "for a match (default 15)")
    args = p.parse_args(argv)

    results_dir, gt_dir = Path(args.results_dir), Path(args.gt_dir)
    vod_ids = ([v.strip() for v in args.vod_ids.split(",") if v.strip()]
               if args.vod_ids else None)
    if vod_ids:
        bad = [v for v in vod_ids if not v.isdigit()]
        if bad:
            print(f"Invalid VOD ID(s): {', '.join(bad)} (must be numeric)")
            return 2

    per_vod = []
    if vod_ids:
        paths = [(results_dir / f"{v}.jsonl", gt_dir / f"{v}.csv") for v in vod_ids]
    else:
        paths = sorted(
            (results_dir / f"{r.stem}.jsonl", gt_dir / f"{r.stem}.csv")
            for r in results_dir.glob("*.jsonl")
        )

    for results_path, gt_path in paths:
        if not results_path.exists():
            print(f"SKIP {results_path.name}: results file not found")
            continue
        if not gt_path.exists():
            print(f"SKIP {results_path.name}: ground truth not found "
                  f"(expected {gt_path})")
            continue
        res = evaluate_one(results_path, gt_path, args.tolerance)
        per_vod.append(res)
        print(f"\nVOD {res['vod_id']}: {res['matched']}/{res['n_gt']} matched "
              f"({res['no_prediction']} no-prediction) | agreement "
              f"{res['agreement_pct']:.1f}%")
        print_matrix(res["matrix"], f"Confusion matrix -- {res['vod_id']}")

    if not per_vod:
        print("Nothing to evaluate.")
        return 1

    # Combined summary across the batch
    total_matrix: Counter = Counter()
    total_gt = total_matched = total_correct = total_no_pred = 0
    for res in per_vod:
        total_matrix.update(res["matrix"])
        total_gt += res["n_gt"]
        total_matched += res["matched"]
        total_correct += res["correct"]
        total_no_pred += res["no_prediction"]
    agreement = (100.0 * total_correct / total_matched) if total_matched else 0.0

    print("\n" + "=" * 60)
    print(f"COMBINED: {total_correct}/{total_matched} matched correct "
          f"({agreement:.1f}%) across {total_gt} GT rows, "
          f"{total_no_pred} no-prediction, {len(per_vod)} VOD(s)")
    print_matrix(total_matrix, "Combined confusion matrix")
    print("=" * 60)

    # Per-label precision/recall from the combined matrix
    for label in LABELS:
        tp = total_matrix.get((label, label), 0)
        gt_total = sum(v for (g, _), v in total_matrix.items() if g == label)
        pred_total = sum(v for (_, pr), v in total_matrix.items() if pr == label)
        recall = (100.0 * tp / gt_total) if gt_total else 0.0
        precision = (100.0 * tp / pred_total) if pred_total else 0.0
        print(f"  {label:<8} precision {precision:5.1f}%  recall {recall:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
