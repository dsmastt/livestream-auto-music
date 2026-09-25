"""Typed exceptions with actionable, human-readable messages.

The batch runner catches these (and only prints `str(exc)`, never a stack
trace) so a missing tool or full disk shows install instructions / limits
instead of crashing the whole run.
"""

from __future__ import annotations


class VibeError(Exception):
    """Base class for all pipeline errors."""


class DiskSpaceError(VibeError):
    """Free disk space fell below the configured threshold."""


class ToolMissingError(VibeError):
    """A required external tool (ffmpeg / yt-dlp / TwitchDownloaderCLI) is not installed."""


class MetadataError(VibeError):
    """Twitch Helix metadata request failed."""


class ChatDownloadError(VibeError):
    """Chat replay download failed (tool error or chat JSON unreadable)."""


class AudioDownloadError(VibeError):
    """Audio download or conversion failed."""


class LLMError(VibeError):
    """LLM classifier request failed."""
