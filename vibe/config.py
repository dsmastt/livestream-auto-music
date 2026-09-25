"""Configuration loading: defaults <- config.yaml (optional) <- CLI overrides.

Secrets (API keys, tokens) are intentionally NOT part of this object: they are
read from environment variables at the point of use so they never end up in
config files or logs. See the `*_env` fields below for which env var each
provider uses.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import yaml

from .errors import VibeError

# Whisper model sizes ordered by VRAM/RAM footprint. "large*" is intentionally
# excluded: this machine has an 8GB VRAM GPU and ~6GB RAM budget per process,
# and a large model can exceed both, so the config refuses to run it.
ALLOWED_WHISPER_MODELS = ("tiny", "base", "small", "medium", "small.en", "medium.en")


@dataclasses.dataclass
class Config:
    # --- disk safety -----------------------------------------------------
    # Abort before any download if free space on the working filesystem is
    # below this many GiB. Raise it if you also want headroom for the model.
    min_free_disk_gb: float = 5.0

    # --- audio / chunking -------------------------------------------------
    sample_rate: int = 16000          # Whisper expects 16 kHz mono
    window_sec: float = 15.0          # rolling window length
    hop_sec: float = 5.0              # step between window starts

    # --- Whisper ----------------------------------------------------------
    whisper_model: str = "small"      # tiny|base|small|medium (never large*)
    whisper_device: str = "auto"      # auto|cuda|cpu
    whisper_compute_type: str = "auto"  # auto -> float16 on GPU, int8 on CPU
    whisper_language: str | None = None  # e.g. "en"; None = auto-detect
    whisper_beam_size: int = 1        # greedy decoding, fastest
    whisper_vad: bool = True          # skip silence inside chunks

    # --- chat -------------------------------------------------------------
    chat_tool: str = "auto"           # auto|twitchdownloader|tcd
    chat_snippet_max_messages: int = 5
    chat_snippet_max_chars: int = 500

    # --- LLM classifier ----------------------------------------------------
    classifier_provider: str = "anthropic"   # anthropic|openai
    classifier_model: str = "claude-sonnet-4-5"
    classifier_batch_size: int = 8    # windows per LLM request
    classifier_max_tokens: int = 1024
    classifier_temperature: float = 0.0
    classifier_timeout_s: float = 60.0
    classifier_max_retries: int = 3
    skip_llm_on_empty: bool = True    # auto-label silent+chatless windows "waiting"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Twitch Helix ------------------------------------------------------
    twitch_client_id_env: str = "TWITCH_CLIENT_ID"
    twitch_access_token_env: str = "TWITCH_ACCESS_TOKEN"
    twitch_api_timeout_s: float = 30.0
    metadata_required: bool = False   # if True, abort when metadata fails

    # --- debounce ----------------------------------------------------------
    debounce_window: int = 3          # commit only after 2 of last 3 agree
    debounce_majority: int = 2

    # --- batch -------------------------------------------------------------
    max_vod_minutes: float | None = None  # cap processed audio length (testing)

    def validate(self) -> None:
        if self.whisper_model not in ALLOWED_WHISPER_MODELS:
            raise VibeError(
                f"whisper_model={self.whisper_model!r} is not allowed. "
                f"Allowed: {', '.join(ALLOWED_WHISPER_MODELS)}. "
                "'large' models are disabled to protect GPU/RAM limits."
            )
        if self.window_sec <= 0 or self.hop_sec <= 0 or self.hop_sec > self.window_sec:
            raise VibeError("Require 0 < hop_sec <= window_sec")
        if self.debounce_majority > self.debounce_window:
            raise VibeError("debounce_majority must be <= debounce_window")
        if self.classifier_provider not in ("anthropic", "openai"):
            raise VibeError(
                f"classifier_provider={self.classifier_provider!r} not supported "
                "(use 'anthropic' or 'openai')"
            )
        if self.chat_tool not in ("auto", "twitchdownloader", "tcd"):
            raise VibeError("chat_tool must be one of: auto, twitchdownloader, tcd")
        if self.max_vod_minutes is not None and self.max_vod_minutes <= 0:
            raise VibeError("max_vod_minutes must be > 0")


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Build a Config from defaults <- YAML file (if present) <- explicit overrides."""
    data: dict[str, Any] = {}
    if path:
        if not os.path.exists(path):
            raise VibeError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise VibeError(f"Config file {path} must contain a YAML mapping")
        data.update(loaded)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise VibeError(f"Unknown config key(s): {', '.join(sorted(unknown))}")

    cfg = Config(**{k: v for k, v in data.items() if k in known})
    cfg.validate()
    return cfg
