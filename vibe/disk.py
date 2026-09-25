"""Disk-space guardrails for a machine with a tight free-space budget."""

from __future__ import annotations

import shutil

from .errors import DiskSpaceError


def free_bytes(path: str = ".") -> int:
    """Free bytes on the filesystem containing `path`."""
    return shutil.disk_usage(path).free


def human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PiB"


def require_free(path: str, min_gb: float, doing: str) -> int:
    """Abort with a clear error if the filesystem holding `path` has less
    than `min_gb` GiB free. Returns the free byte count on success."""
    free = free_bytes(path)
    if free < min_gb * 1024**3:
        raise DiskSpaceError(
            f"Refusing to {doing}: only {human(free)} free on the filesystem "
            f"containing {path!r}, below the safety threshold of {min_gb} GiB. "
            f"Free up space or raise `min_free_disk_gb` in the config (not "
            f"recommended below 2 GiB)."
        )
    return free
