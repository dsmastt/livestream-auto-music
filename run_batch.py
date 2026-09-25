#!/usr/bin/env python3
"""Batch runner: process a list of Twitch VOD IDs ONE AT A TIME, sequentially.

Usage:
    python run_batch.py --vod-ids 123456789,987654321 --output-dir ./results/
    python run_batch.py --vod-ids 123456789 --output-dir ./results/ --max-vod-minutes 30
    python run_batch.py --vod-ids ... --config config.yaml --dry-run

Resource guarantees (see README):
  * disk headroom checked before every download (default 5 GiB free)
  * VODs processed strictly sequentially -- no concurrency
  * audio-only downloads; the video stream is never stored
  * Whisper model loaded/unloaded per VOD; model size capped (no "large")
  * all intermediates deleted per VOD; freed bytes logged
  * disk usage + elapsed time printed between VODs
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from vibe.config import load_config
from vibe.disk import free_bytes, human, require_free
from vibe.errors import VibeError
from vibe.pipeline import VodResult, process_vod
from vibe.util import fmt_hms

log = logging.getLogger("vibe.batch")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequentially download + transcribe + classify Twitch VODs.")
    p.add_argument("--vod-ids", required=True,
                   help="Comma-separated Twitch VOD IDs, e.g. 123456789,987654321")
    p.add_argument("--output-dir", default="./results",
                   help="Directory for results/<vod_id>.jsonl (default: ./results)")
    p.add_argument("--config", default=None,
                   help="Optional YAML config (see config.example.yaml)")
    p.add_argument("--max-vod-minutes", type=float, default=None,
                   help="Cap processed audio per VOD (early testing). "
                        "Default: no limit (config.max_vod_minutes).")
    p.add_argument("--work-dir", default="./work",
                   help="Scratch dir for per-VOD intermediates (deleted after "
                        "each VOD). Default: ./work")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan + disk check, download nothing.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    vod_ids = [v.strip() for v in args.vod_ids.split(",") if v.strip()]
    if not vod_ids:
        log.error("No VOD IDs given.")
        return 2
    for v in vod_ids:
        if not v.isdigit():
            log.error("Invalid VOD ID %r (must be numeric).", v)
            return 2

    try:
        config = load_config(args.config, overrides={
            "max_vod_minutes": args.max_vod_minutes,
        })
    except VibeError as exc:
        log.error("Config error: %s", exc)
        return 2

    out_dir = Path(args.output_dir)
    work_root = Path(args.work_dir)

    log.info("Batch of %d VOD(s) | whisper=%s (%s) | classifier=%s/%s | "
             "disk threshold=%.1f GiB | max minutes=%s",
             len(vod_ids), config.whisper_model, config.whisper_device,
             config.classifier_provider, config.classifier_model,
             config.min_free_disk_gb,
             config.max_vod_minutes if config.max_vod_minutes else "none")

    if args.dry_run:
        free = require_free(str(work_root.parent), config.min_free_disk_gb,
                            "dry-run")
        print(f"[dry-run] would process: {', '.join(vod_ids)}")
        print(f"[dry-run] output dir: {out_dir}")
        print(f"[dry-run] free disk now: {human(free)} (>= {config.min_free_disk_gb} GiB required)")
        for v in vod_ids:
            print(f"[dry-run]   {v}: audio-only download -> whisper "
                  f"{config.whisper_model} -> chat align -> classify -> "
                  f"{out_dir / (v + '.jsonl')}")
        return 0

    batch_start = time.time()
    results: list[VodResult] = []
    for i, vod_id in enumerate(vod_ids, start=1):
        free = free_bytes(str(work_root.parent))
        print(f"\n=== VOD {i}/{len(vod_ids)}: {vod_id} | elapsed {fmt_hms(time.time() - batch_start)} "
              f"| free disk {human(free)} ===")
        result = process_vod(vod_id, out_dir, config, work_root=work_root)
        results.append(result)
        print(f"--- VOD {vod_id}: {'OK' if result.ok else 'FAILED'} | "
              f"processed {fmt_hms(result.processed_seconds or 0)} | "
              f"elapsed {fmt_hms(result.elapsed_s)} | freed {human(result.disk_freed_bytes)} | "
              f"free disk now {human(free_bytes(str(work_root.parent)))}")

    # ---- batch summary -----------------------------------------------------
    total_elapsed = time.time() - batch_start
    ok = sum(1 for r in results if r.ok)
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    for r in results:
        status = "OK     " if r.ok else "FAILED "
        err = f" -- {r.error}" if r.error else ""
        print(f"  {status} {r.vod_id}: {fmt_hms(r.processed_seconds or 0)} "
              f"processed, {r.windows} windows, labels={r.label_counts or {}}{err}")
    print(f"  {ok}/{len(results)} VODs succeeded | total elapsed {fmt_hms(total_elapsed)} "
          f"| free disk now {human(free_bytes(str(work_root.parent)))}")
    print("=" * 60)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
