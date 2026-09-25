"""Twitch Helix metadata (Get Videos endpoint).

NOTE ON GAME/CATEGORY DATA:
    Twitch's Helix "Get Videos" endpoint returns at most one game/category for
    the whole VOD and does NOT reliably report category changes that happen
    mid-stream. We therefore capture whatever `game_name` the API returns
    (usually the starting category, sometimes the current one) as a single
    static context tag, and deliberately do NOT attempt to scrape or otherwise
    reconstruct mid-VOD category changes. If the API returns nothing, the tag
    is simply empty.

Requires:
    TWITCH_CLIENT_ID    (env) - your app's client ID
    TWITCH_ACCESS_TOKEN (env) - a user/app access token for the Helix API
Get both from https://dev.twitch.tv/console/apps
"""

from __future__ import annotations

import logging
import os
import re

import requests

from .errors import MetadataError

log = logging.getLogger("vibe.meta")

HELIX_URL = "https://api.twitch.tv/helix/videos"

_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


def parse_duration(duration: str) -> int:
    """Parse Twitch's '1h23m45s' duration format into seconds."""
    m = _DURATION_RE.fullmatch(duration.strip())
    if not m:
        raise MetadataError(f"Unrecognized Twitch duration format: {duration!r}")
    h, mn, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mn * 60 + s


def fetch_vod_metadata(vod_id: str, client_id_env: str, token_env: str,
                       timeout: float) -> dict:
    """Pull title / duration / view_count / created_at / starting game.

    Returns a dict with keys: id, title, duration_seconds, view_count,
    created_at, game_name. Non-fatal caller-side; raises MetadataError on
    failure with instructions.
    """
    client_id = os.environ.get(client_id_env, "").strip()
    token = os.environ.get(token_env, "").strip()
    if not client_id or not token:
        raise MetadataError(
            "Twitch Helix credentials missing. Set env vars "
            f"{client_id_env} and {token_env} (see README > 'Twitch API "
            "credentials')."
        )

    resp = requests.get(
        HELIX_URL,
        params={"id": vod_id},
        headers={"Client-Id": client_id, "Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    if resp.status_code == 401:
        raise MetadataError(
            f"Twitch API returned 401 for VOD {vod_id}: the access token in "
            f"${token_env} is expired or invalid. Generate a fresh one and retry."
        )
    if resp.status_code != 200:
        raise MetadataError(
            f"Twitch Get Videos failed for {vod_id}: HTTP {resp.status_code} "
            f"{resp.text[:200]}"
        )

    payload = resp.json()
    data = payload.get("data") or []
    if not data:
        raise MetadataError(
            f"Twitch API returned no metadata for VOD {vod_id} (deleted, "
            "unlisted, or wrong ID?)."
        )

    vod = data[0]
    duration_raw = vod.get("duration", "0s")
    return {
        "id": vod.get("id", vod_id),
        "title": vod.get("title"),
        "duration_seconds": parse_duration(duration_raw) if duration_raw else None,
        "view_count": vod.get("view_count"),
        "created_at": vod.get("created_at"),
        # Starting/current category. See module docstring: this is a single
        # static tag; mid-VOD category changes are not available from Helix.
        "game_name": vod.get("game_name"),
    }
