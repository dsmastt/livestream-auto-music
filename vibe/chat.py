"""Chat replay download: TwitchDownloaderCLI first, `tcd` Python package as fallback.

Both tools emit comment objects with a `content_offset_seconds` timestamp so
messages can be aligned to audio windows later. We normalize both formats into
plain dicts: {offset_seconds, created_at, author, text}.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .errors import ChatDownloadError, ToolMissingError
from .util import run_cmd

log = logging.getLogger("vibe.chat")

TWITCHDOWNLOADER_HINT = (
    "TwitchDownloaderCLI was not found. Install it from the GitHub releases "
    "(https://github.com/lay295/TwitchDownloader/releases), put the binary on "
    "your PATH, or set chat_tool: tcd in the config.\n"
    "Alternatively install the Python fallback:  pip install tcd"
)

VOD_URL = "https://www.twitch.tv/videos/{vod_id}"


def download_chat(vod_id: str, out_path: Path, tool: str = "auto",
                  end_seconds: float | None = None,
                  timeout: float | None = None) -> str:
    """Download the full chat replay to `out_path` (JSON). Returns tool used."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tool in ("auto", "twitchdownloader"):
        cli = shutil.which("TwitchDownloaderCLI")
        if cli:
            _chat_via_cli(cli, vod_id, out_path, end_seconds, timeout)
            return "twitchdownloader"
        if tool == "twitchdownloader":
            raise ToolMissingError(TWITCHDOWNLOADER_HINT)
        log.warning("TwitchDownloaderCLI not found; falling back to the 'tcd' "
                    "Python package.")

    _chat_via_tcd(vod_id, out_path, end_seconds, timeout)
    return "tcd"


def _chat_via_cli(cli: str, vod_id: str, out_path: Path,
                  end_seconds: float | None, timeout: float | None) -> None:
    cmd = [cli, "chatdownload", "-u", VOD_URL.format(vod_id=vod_id),
           "-o", str(out_path)]
    if end_seconds is not None:
        # Only keep chat up to the processed audio range (saves disk on long
        # VODs when --max-vod-minutes is in play).
        cmd += ["-b", "0", "-e", str(end_seconds)]
    try:
        run_cmd(cmd, timeout=timeout)
    except RuntimeError as exc:
        raise ChatDownloadError(
            f"TwitchDownloaderCLI chat download failed for VOD {vod_id}:\n{exc}"
        ) from exc
    if not out_path.exists():
        raise ChatDownloadError(
            f"TwitchDownloaderCLI ran but produced no file for VOD {vod_id}."
        )


def _chat_via_tcd(vod_id: str, out_path: Path,
                  end_seconds: float | None, timeout: float | None) -> None:
    try:
        from tcd import Twitch  # type: ignore
    except ImportError as exc:
        raise ToolMissingError(
            "Neither TwitchDownloaderCLI nor the 'tcd' Python package is "
            "available.\n" + TWITCHDOWNLOADER_HINT
        ) from exc

    try:
        t = Twitch()
        # The tcd API differs slightly between versions; try both call shapes.
        try:
            messages = t.get_chat(vod_id)
        except TypeError:
            messages = t.get_chat(video_id=vod_id)
    except Exception as exc:
        raise ChatDownloadError(
            f"'tcd' failed to start downloading chat for VOD {vod_id}: {exc}"
        ) from exc

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            count = 0
            for msg in messages:
                norm = normalize_message(msg)
                if norm is None:
                    continue
                if end_seconds is not None and norm["offset_seconds"] > end_seconds:
                    continue
                f.write(json.dumps(norm, ensure_ascii=False) + "\n")
                count += 1
        if count == 0:
            log.warning("tcd returned no messages for VOD %s", vod_id)
    except Exception as exc:
        raise ChatDownloadError(
            f"Failed while reading 'tcd' messages for VOD {vod_id}: {exc}"
        ) from exc


def normalize_message(raw: dict) -> dict | None:
    """Normalize a TwitchDownloaderCLI or tcd comment object.

    Returns {offset_seconds, created_at, author, text} or None when the
    message has no usable text (e.g. deleted/system entries).
    """
    if not isinstance(raw, dict):
        return None
    offset = raw.get("content_offset_seconds")
    if offset is None:
        offset = raw.get("content_offset")
    if offset is None:
        return None

    commenter = raw.get("commenter") or {}
    author = ""
    if isinstance(commenter, dict):
        author = commenter.get("display_name") or commenter.get("name") or ""

    text = _extract_text(raw.get("message"))
    if not text:
        return None
    if raw.get("is_action"):
        text = f"/me {text}"

    return {
        "offset_seconds": float(offset),
        "created_at": raw.get("created_at"),
        "author": author,
        "text": text,
    }


def _extract_text(message) -> str:
    """Handle both shapes: plain string, or TwitchDownloader's
    {"body": ..., "fragments": [{"text": ...}, ...]} object."""
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, dict):
        body = (message.get("body") or "").strip()
        if body:
            return body
        frags = message.get("fragments") or []
        parts = [f.get("text", "") for f in frags if isinstance(f, dict)]
        return "".join(parts).strip()
    return ""


def load_chat(path: Path) -> list[dict]:
    """Load a chat file produced by either tool into normalized messages.

    Accepts both TwitchDownloaderCLI's JSON array format and tcd's JSON-lines.
    """
    if not path.exists():
        raise ChatDownloadError(f"Chat file not found: {path}")
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            messages = (normalize_message(m) for m in data)
        elif isinstance(data, dict):
            comments = data.get("comments") or data.get("data")
            if comments is None:
                comments = [data]   # a single comment object, not an envelope
            messages = (normalize_message(m) for m in comments)
        else:
            raise ChatDownloadError(f"Unexpected chat file structure in {path.name}")
        return [m for m in messages if m is not None]
    except json.JSONDecodeError:
        pass

    # JSON-lines (tcd fallback output)
    out: list[dict] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            norm = normalize_message(json.loads(line))
        except json.JSONDecodeError:
            continue
        if norm:
            out.append(norm)
    return out
