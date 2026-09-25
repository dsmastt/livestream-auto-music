"""Per-VOD pipeline orchestration.

Order of operations for one VOD (fully sequential, nothing runs in parallel):
  1. disk headroom check            (abort BEFORE downloading anything)
  2. Helix metadata                 (non-fatal unless metadata_required)
  3. audio download (audio-only)    (never the video stream)
  4. chat replay download           (TwitchDownloaderCLI or tcd)
  5. stream-chunk -> Whisper -> align chat -> LLM classify -> debounce
  6. write results/<vod_id>.jsonl + .meta.json
  7. delete ALL intermediate files, log bytes freed, unload the model

Only the final JSONL + meta JSON survive; everything else (raw audio, chat
JSON) is deleted immediately after the VOD is processed.
"""

from __future__ import annotations

import bisect
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import audio as audio_mod
from . import chat as chat_mod
from .classify import LLMClassifier, debounce
from .config import Config
from .disk import free_bytes, human, require_free
from .errors import VibeError
from .transcribe import Transcriber
from .twitch_meta import fetch_vod_metadata
from .util import fmt_hms

log = logging.getLogger("vibe.pipeline")


@dataclass
class VodResult:
    vod_id: str
    ok: bool
    error: str | None = None
    title: str | None = None
    game_name: str | None = None
    created_at: str | None = None
    view_count: int | None = None
    duration_seconds: float | None = None
    processed_seconds: float | None = None
    windows: int = 0
    llm_calls: int = 0
    disk_freed_bytes: int = 0
    elapsed_s: float = 0.0
    label_counts: dict = field(default_factory=dict)


def _chat_features(msgs: list[dict], offsets: list[float],
                   start: float, end: float, cfg: Config) -> tuple[float, str]:
    """Bucket chat into [start, end): return (rate_per_min, snippet)."""
    lo = bisect.bisect_left(offsets, start)
    hi = bisect.bisect_left(offsets, end)
    in_win = msgs[lo:hi]
    rate = len(in_win) / (cfg.window_sec / 60.0)

    # Raw sample: the last N messages in the window (closest to the end).
    sample = in_win[-cfg.chat_snippet_max_messages:]
    parts = []
    for m in sample:
        who = m["author"] or "user"
        parts.append(f"[{m['offset_seconds']:.0f}s] {who}: {m['text']}")
    snippet = " | ".join(parts)
    if len(snippet) > cfg.chat_snippet_max_chars:
        snippet = snippet[: cfg.chat_snippet_max_chars] + "..."
    return rate, snippet


def process_vod(vod_id: str, output_dir: Path, config: Config,
                work_root: Path | None = None,
                progress_every_windows: int = 50) -> VodResult:
    t0 = time.time()
    result = VodResult(vod_id=vod_id, ok=False)
    work = (work_root or Path("work")) / vod_id
    out_dir = output_dir

    try:
        # -- 0. validate the ID before it touches any filesystem path ----------
        if not str(vod_id).isdigit():
            raise VibeError(
                f"Invalid VOD ID {vod_id!r}: must be numeric "
                "(Twitch VOD IDs are plain digits)."
            )

        # -- 1. disk headroom, measured on the working filesystem -------------
        work.parent.mkdir(parents=True, exist_ok=True)
        free_before = require_free(str(work.parent), config.min_free_disk_gb,
                                   "start downloading VOD " + vod_id)
        log.info("[%s] free disk %s (threshold %s GiB) -- OK",
                 vod_id, human(free_before), config.min_free_disk_gb)

        # -- 2. metadata (best effort unless required) ------------------------
        try:
            meta = fetch_vod_metadata(
                vod_id, config.twitch_client_id_env,
                config.twitch_access_token_env, config.twitch_api_timeout_s)
            result.title = meta.get("title")
            result.game_name = meta.get("game_name")
            result.created_at = meta.get("created_at")
            result.view_count = meta.get("view_count")
            result.duration_seconds = meta.get("duration_seconds")
            log.info("[%s] metadata: %r | game=%s | duration=%s", vod_id,
                     result.title, result.game_name or "n/a",
                     fmt_hms(result.duration_seconds or 0))
        except VibeError as exc:
            if config.metadata_required:
                raise
            log.warning("[%s] metadata unavailable (continuing): %s", vod_id, exc)

        max_sec = config.max_vod_minutes * 60.0 if config.max_vod_minutes else None
        if max_sec is not None:
            log.info("[%s] capping processing at %s (--max-vod-minutes)",
                     vod_id, fmt_hms(max_sec))

        if result.duration_seconds:
            # Footprint envelope: mono 16k wav ~115 MB/h + full raw audio
            # download (~72 MB/h) + chat JSON slack => ~250 MB/h, conservative.
            # yt-dlp pulls the entire media file even when --download-sections
            # caps the final length, so the estimate uses the full duration.
            est_gb = result.duration_seconds * 250.0 / (1024.0 * 3600.0)
            require_free(
                str(work.parent), config.min_free_disk_gb + est_gb,
                f"process VOD {vod_id} (estimated ~{est_gb:.1f} GiB needed "
                f"for its {fmt_hms(result.duration_seconds)} of audio+chat)")
            log.info("[%s] estimated footprint ~%.1f GiB -- within budget",
                     vod_id, est_gb)

        # -- 3. audio download (audio-only, mono 16 kHz) ----------------------
        wav = audio_mod.download_audio(vod_id, work, config.sample_rate,
                                       config.max_vod_minutes)
        log.info("[%s] audio ready: %s (%s)", vod_id, wav.name,
                 human(wav.stat().st_size))

        # -- 4. chat replay ---------------------------------------------------
        # Re-check headroom: the chat JSON can be hundreds of MB on long VODs
        # and the check above happened before the audio download.
        require_free(str(work.parent), config.min_free_disk_gb,
                     f"download chat for VOD {vod_id}")
        chat_path = work / f"{vod_id}_chat.json"
        tool = chat_mod.download_chat(vod_id, chat_path, config.chat_tool,
                                      end_seconds=max_sec,
                                      timeout=1800)
        messages = chat_mod.load_chat(chat_path)
        log.info("[%s] chat replay: %d messages (tool=%s)", vod_id,
                 len(messages), tool)

        # -- 5. transcribe + align + classify + debounce ----------------------
        offsets = [m["offset_seconds"] for m in messages]
        # Pre-sort by offset if the tool returned them out of order.
        order = sorted(range(len(messages)), key=lambda i: offsets[i])
        messages = [messages[i] for i in order]
        offsets = [offsets[i] for i in order]

        transcriber = Transcriber(config)
        classifier = LLMClassifier(config)

        windows: list[dict] = []
        n_windows = 0
        t_transcribe_start = time.time()

        transcriber.load()
        try:
            for start, end, audio in audio_mod.iter_windows(
                    wav, config.window_sec, config.hop_sec, max_sec,
                    sample_rate=config.sample_rate):
                transcript = transcriber.transcribe(audio)
                rate, snippet = _chat_features(messages, offsets,
                                               start, end, config)
                windows.append({
                    "start": start, "end": end,
                    "transcript": transcript,
                    "chat": snippet,
                    "chat_rate": rate,
                    "raw": "", "committed": "",
                })
                n_windows += 1
                if n_windows % progress_every_windows == 0:
                    log.info("[%s] %d windows done @ %s | elapsed %s | free disk %s",
                             vod_id, n_windows, fmt_hms(start),
                             fmt_hms(time.time() - t0),
                             human(free_bytes(str(work.parent))))
        finally:
            transcriber.unload()

        t_transcribe = time.time() - t_transcribe_start
        if n_windows == 0:
            raise VibeError(f"No audio windows produced for VOD {vod_id} "
                            "(audio too short or empty?)")

        # Classify (batched LLM calls; empty windows short-circuit)
        t_classify_start = time.time()
        raw_labels, llm_calls = classifier.classify(windows, result.game_name)
        committed = debounce(raw_labels, config.debounce_window,
                             config.debounce_majority)
        for i, (w, r, c) in enumerate(zip(windows, raw_labels, committed)):
            w["raw"], w["committed"] = r, c
        t_classify = time.time() - t_classify_start
        log.info("[%s] transcription %s | classification %s (%d LLM calls)",
                 vod_id, fmt_hms(t_transcribe), fmt_hms(t_classify), llm_calls)

        # -- 6. write outputs -------------------------------------------------
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / f"{vod_id}.jsonl"
        with open(result_path, "w", encoding="utf-8") as f:
            for w in windows:
                line = {
                    "start_time": round(w["start"], 2),
                    "end_time": round(w["end"], 2),
                    "transcript_snippet": w["transcript"],
                    "chat_snippet": w["chat"],
                    "chat_rate_per_minute": round(w["chat_rate"], 2),
                    "raw_label": w["raw"],
                    "committed_label": w["committed"],
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        from collections import Counter
        counts = Counter(w["committed"] for w in windows if w["committed"])
        meta_out = {
            "vod_id": vod_id,
            "title": result.title,
            "game_name": result.game_name,
            "created_at": result.created_at,
            "view_count": result.view_count,
            "vod_duration_seconds": result.duration_seconds,
            "processed_seconds": round(windows[-1]["end"], 2),
            "window_sec": config.window_sec,
            "hop_sec": config.hop_sec,
            "n_windows": n_windows,
            "llm_calls": llm_calls,
            "label_counts": dict(sorted(counts.items())),
            "whisper_model": config.whisper_model,
            "classifier_model": config.classifier_model,
            "elapsed_s": round(time.time() - t0, 1),
        }
        meta_path = out_dir / f"{vod_id}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)
            f.write("\n")

        result.ok = True
        result.windows = n_windows
        result.llm_calls = llm_calls
        result.processed_seconds = meta_out["processed_seconds"]
        result.label_counts = dict(counts)
        result.elapsed_s = time.time() - t0
        log.info("[%s] wrote %s (%d windows) + %s", vod_id, result_path.name,
                 n_windows, meta_path.name)
        return result

    except VibeError as exc:
        result.error = str(exc)
        log.error("[%s] FAILED: %s", vod_id, exc)
        return result
    except Exception as exc:  # unexpected: keep batch alive, still report
        result.error = f"{type(exc).__name__}: {exc}"
        log.exception("[%s] unexpected error", vod_id)
        return result
    finally:
        # -- 7. cleanup: delete every intermediate, log what was freed ---------
        if work.exists():
            before = free_bytes(str(work.parent))
            shutil.rmtree(work, ignore_errors=True)
            result.disk_freed_bytes = max(0, free_bytes(str(work.parent)) - before)
            log.info("[%s] temp files deleted from %s -- freed %s",
                     vod_id, work, human(result.disk_freed_bytes))
        if not result.ok:
            result.elapsed_s = time.time() - t0
