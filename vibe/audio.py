"""Audio acquisition and streaming chunking.

Hard rule: we NEVER persist the video stream. yt-dlp is told to extract
audio-only (`-x`), which fetches the smallest audio-only rendition available
and transcodes it to mono 16 kHz WAV directly via the ffmpeg postprocessor.
The intermediate raw audio is deleted by the pipeline after processing.

Memory rule: the full audio is never loaded into RAM. `iter_windows` streams
the WAV in small blocks and yields 15s numpy windows one at a time, keeping
only ~1 MB resident.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from .errors import AudioDownloadError
from .util import find_tool, run_cmd

log = logging.getLogger("vibe.audio")

VOD_URL = "https://www.twitch.tv/videos/{vod_id}"

YTDLP_HINT = (
    "Install yt-dlp with:  pip install -U yt-dlp"
)
FFMPEG_HINT = (
    "Install ffmpeg with:\n"
    "  Ubuntu/Debian: sudo apt install ffmpeg\n"
    "  macOS:         brew install ffmpeg\n"
    "  Windows:       winget install Gyan.FFmpeg"
)


def download_audio(vod_id: str, out_dir: Path, sample_rate: int,
                   max_minutes: float | None = None,
                   timeout_per_dl: float | None = None) -> Path:
    """Download audio-only from a Twitch VOD and convert to mono 16 kHz WAV.

    Returns the path of the final mono WAV. Never writes a video file.
    """
    find_tool("yt-dlp", YTDLP_HINT)          # fail early with instructions
    find_tool("ffmpeg", FFMPEG_HINT)         # needed by -x / --download-sections
    out_dir.mkdir(parents=True, exist_ok=True)

    url = VOD_URL.format(vod_id=vod_id)
    raw = out_dir / f"{vod_id}_raw.wav"

    cmd = [
        "yt-dlp",
        "-x",                       # audio only -- never keep the video stream
        "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ac 1 -ar {sample_rate}",  # mono 16k
        "--no-playlist",
        "--no-warnings",
        "--newline",
        "--retries", "5",
        "-o", str(raw),
    ]
    if max_minutes is not None:
        # Cap the FINAL audio length for early testing. NOTE: yt-dlp downloads
        # the full media file before cutting the section, so this does NOT
        # bound the download footprint -- the pipeline guards that separately
        # with a duration-based free-space estimate.
        cmd += ["--download-sections", f"*0:00-{int(max_minutes * 60)}",
                "--force-keyframes-at-cuts"]
    cmd += [url]

    log.info("Downloading audio-only for VOD %s (yt-dlp -x)...", vod_id)
    proc = run_cmd(cmd, timeout=timeout_per_dl)
    # yt-dlp streams progress to stdout; surface the final lines.
    lines = (proc.stdout or "").strip().splitlines()
    if lines:
        log.debug("yt-dlp last line: %s", lines[-1])

    if not raw.exists() or raw.stat().st_size == 0:
        raise AudioDownloadError(
            f"yt-dlp finished but produced no audio for VOD {vod_id}. The VOD "
            "may be deleted, subscribers-only, or geo-blocked."
        )

    final = out_dir / f"{vod_id}_16k_mono.wav"
    # ensure_mono_16k returns `raw` when the yt-dlp postprocessor already
    # produced mono 16 kHz (the normal case). Only delete raw when we
    # actually converted to a new file.
    out = ensure_mono_16k(raw, final, sample_rate)
    if out != raw:
        raw.unlink(missing_ok=True)   # never keep the raw intermediate
    return out


def ensure_mono_16k(src: Path, dst: Path, sample_rate: int) -> Path:
    """Verify src is mono @ sample_rate; re-encode with ffmpeg if not.

    Returns the canonical mono wav path (== dst when conversion happens).
    """
    try:
        info = sf.info(str(src))
    except Exception:
        info = None
    if info is not None and info.samplerate == sample_rate and info.channels == 1:
        return src

    ffmpeg = find_tool("ffmpeg", FFMPEG_HINT)
    log.info("Converting %s to mono %d Hz...", src.name, sample_rate)
    run_cmd([ffmpeg, "-y", "-i", str(src), "-ac", "1", "-ar", str(sample_rate),
             str(dst)])
    if not dst.exists():
        raise AudioDownloadError(f"ffmpeg did not produce {dst.name}")
    return dst


def iter_windows(path: Path, window_sec: float, hop_sec: float,
                 max_seconds: float | None = None,
                 sample_rate: int = 16000) -> Iterator[tuple[float, float, np.ndarray]]:
    """Yield (start_s, end_s, float32 mono samples) for rolling windows.

    Streaming: only ~1 window of audio is held in memory at any time. Windows
    are emitted at t = 0, hop, 2*hop, ... while at least `window_sec` of audio
    remains; any trailing tail shorter than a full window is dropped.
    """
    if max_seconds is not None and max_seconds < window_sec:
        return

    win = int(round(window_sec * sample_rate))
    hop = int(round(hop_sec * sample_rate))
    max_frames = int(max_seconds * sample_rate) if max_seconds is not None else None

    blocks: list[np.ndarray] = []
    have = 0
    start_s = 0.0
    pos = 0

    with sf.SoundFile(str(path)) as f:
        if f.samplerate != sample_rate:
            raise AudioDownloadError(
                f"{path.name} is {f.samplerate} Hz; expected {sample_rate} Hz"
            )
        while True:
            if max_frames is not None and pos >= max_frames:
                break
            n = hop if max_frames is None else min(hop, max_frames - pos)
            block = f.read(n, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            if block.shape[1] > 1:
                block = block.mean(axis=1)
            else:
                block = block[:, 0]
            pos += block.shape[0]
            blocks.append(block)
            have += block.shape[0]

            while have >= win:
                buf = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
                yield start_s, start_s + window_sec, buf[:win].copy()
                start_s += hop_sec
                drop = hop
                while drop > 0 and blocks:
                    b = blocks[0]
                    if b.shape[0] <= drop:
                        drop -= b.shape[0]
                        have -= b.shape[0]
                        blocks.pop(0)
                    else:
                        blocks[0] = b[drop:]
                        have -= drop
                        drop = 0
