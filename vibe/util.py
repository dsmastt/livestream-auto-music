"""Small shared helpers: time formatting, subprocess/tool discovery, JSON I/O."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .errors import ToolMissingError

log = logging.getLogger("vibe")


def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def fmt_mmss(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def find_tool(name: str, install_hint: str) -> str:
    """Locate an external tool in PATH or raise with install instructions."""
    path = shutil.which(name)
    if not path:
        raise ToolMissingError(
            f"Required tool '{name}' was not found on PATH.\n{install_hint}"
        )
    return path


def run_cmd(cmd: list[str], timeout: float | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a command, surfacing stderr tail on failure instead of a raw traceback."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from exc
    except FileNotFoundError as exc:
        raise ToolMissingError(f"Could not execute '{cmd[0]}'. Is it installed and on PATH?") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail)
        )
    return proc


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
