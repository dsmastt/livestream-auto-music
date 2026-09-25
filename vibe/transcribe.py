"""faster-whisper wrapper with explicit load/unload around each VOD.

The model is deliberately NOT kept resident across VODs: the batch runner
calls load() before transcribing a VOD and unload() immediately after, so VRAM
and RAM are released between VODs. Model size is capped at "small" by default
(large* is refused by config validation).
"""

from __future__ import annotations

import gc
import logging

import numpy as np

from .config import Config

log = logging.getLogger("vibe.transcribe")


def detect_device() -> str:
    """Return 'cuda' if a usable CUDA GPU exists, else 'cpu'."""
    try:
        import ctranslate2  # ships with faster-whisper
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Transcriber:
    def __init__(self, config: Config):
        self.cfg = config
        self.model = None
        self.device = config.whisper_device
        self.used_gpu = False

    def load(self) -> None:
        from faster_whisper import WhisperModel  # imported lazily (heavy)

        if self.cfg.whisper_device == "auto":
            self.device = detect_device()
        if self.device == "cuda":
            self.used_gpu = True
        else:
            log.warning(
                "No CUDA GPU detected (or whisper_device=cpu) -- transcribing on "
                "CPU. This is SLOW; use a shorter VOD, --max-vod-minutes, or a "
                "smaller whisper_model."
            )

        compute_type = self.cfg.whisper_compute_type
        if compute_type == "auto":
            compute_type = "float16" if self.device == "cuda" else "int8"

        log.info("Loading Whisper model %r on %s (compute=%s)...",
                 self.cfg.whisper_model, self.device, compute_type)
        self.model = WhisperModel(
            self.cfg.whisper_model,
            device=self.device,
            compute_type=compute_type,
        )

    def unload(self) -> None:
        if self.model is not None:
            self.model = None
            gc.collect()
            if self.used_gpu:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        log.info("Whisper model unloaded; memory released.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe one mono 16 kHz float32 window. Returns text ('' if none)."""
        if self.model is None:
            raise RuntimeError("Transcriber not loaded")
        segments, _info = self.model.transcribe(
            audio,
            beam_size=self.cfg.whisper_beam_size,
            language=self.cfg.whisper_language,
            vad_filter=self.cfg.whisper_vad,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
