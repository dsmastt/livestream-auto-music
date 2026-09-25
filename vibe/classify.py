"""LLM mood classifier (5-label taxonomy) + debounce logic.

Two providers are supported and selected via config:
  - anthropic: native Anthropic Messages API
  - openai:    any OpenAI-compatible /chat/completions endpoint
               (works for Qwen-VL / DeepSeek style local servers)

Requests are batched (classifier_batch_size windows per call) to keep API
cost and latency sane on multi-hour VODs. Secrets are read from env vars named
by the config so nothing sensitive lands in config files or logs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter, deque

import requests

from .errors import LLMError

log = logging.getLogger("vibe.classify")

LABELS = ["hype", "comedic", "chill", "waiting", "tense"]

SYSTEM_PROMPT = (
    "You are an expert annotator of Twitch livestream moods. You receive short "
    "(~15 second) moments from a VOD, each with: the stream's game, a raw ASR "
    "transcript of the audio (may be noisy or empty), and a few recent chat "
    "messages. Classify each moment into exactly ONE of these labels:\n"
    "- hype: high energy, big moments, viewer spikes, screaming, celebrations\n"
    "- comedic: jokes, banter, laughing, funny chat moments\n"
    "- chill: relaxed, calm, slow pace, normal gameplay\n"
    "- waiting: dead air, streamer AFK/loading, technical difficulties, silence\n"
    "- tense: suspense, clutch moments, conflict, anxiety, awkwardness\n"
    "Respond with JSON only: a single object mapping the window number to its "
    "label, e.g. {\"0\": \"chill\", \"1\": \"hype\"}."
)


def debounce(labels: list[str], window: int = 3, majority: int = 2) -> list[str]:
    """Commit a label only when `majority` of the last `window` windows agree.

    Returns a list parallel to `labels` with '' where no commitment was made
    (the first `window-1` windows can never commit).
    """
    committed: list[str] = []
    recent: deque[str] = deque(maxlen=window)
    for lab in labels:
        recent.append(lab)
        if len(recent) == window:
            top, count = Counter(recent).most_common(1)[0]
            committed.append(top if count >= majority else "")
        else:
            committed.append("")
    return committed


def _empty_window(record: dict) -> bool:
    return not record["transcript"].strip() and not record["chat"].strip()


class LLMClassifier:
    def __init__(self, config):
        self.cfg = config

    # -- public API ---------------------------------------------------------
    def classify(self, windows: list[dict], game_tag: str | None) -> tuple[list[str], int]:
        """Return (one raw label per window, number of LLM requests made)."""
        out = [""] * len(windows)
        calls = 0
        pending: list[tuple[int, dict]] = []   # (original_index, window)

        def flush() -> None:
            nonlocal calls
            if not pending:
                return
            idxs = [i for i, _ in pending]
            labels = self._request_batch([w for _, w in pending], game_tag)
            calls += 1
            for i, lab in zip(idxs, labels):
                out[i] = lab

        for i, w in enumerate(windows):
            if self.cfg.skip_llm_on_empty and _empty_window(w):
                # Silent window with no chat: skip the LLM call entirely.
                out[i] = "waiting"
                continue
            pending.append((i, w))
            if len(pending) >= self.cfg.classifier_batch_size:
                flush()
                pending = []
        flush()
        return out, calls

    # -- provider plumbing ---------------------------------------------------
    def _request_batch(self, windows: list[dict], game_tag: str | None) -> list[str]:
        prompt = self._build_prompt(windows, game_tag)
        last_err: Exception | None = None
        for attempt in range(self.cfg.classifier_max_retries):
            try:
                if self.cfg.classifier_provider == "anthropic":
                    text = self._call_anthropic(prompt)
                else:
                    text = self._call_openai(prompt)
                return self._parse_response(text, len(windows))
            except LLMError as exc:
                last_err = exc
                if attempt + 1 < self.cfg.classifier_max_retries:
                    wait = 2 ** attempt
                    log.warning("LLM request failed (%s); retrying in %ds "
                                "(attempt %d/%d)...", exc, wait, attempt + 1,
                                self.cfg.classifier_max_retries)
                    time.sleep(wait)
        raise LLMError(f"LLM classifier failed after "
                       f"{self.cfg.classifier_max_retries} attempts: {last_err}")

    def _build_prompt(self, windows: list[dict], game_tag: str | None) -> str:
        lines = [f"Stream game: {game_tag or 'unknown'}"]
        lines.append("Classify each of the following moments (window number: label):")
        for i, w in enumerate(windows):
            lines.append(
                f"\nWindow {i} [{w['start']:.0f}-{w['end']:.0f}s]:\n"
                f"transcript: {w['transcript'] or '(silence)'}\n"
                f"chat: {w['chat'] or '(no chat)'}"
            )
        lines.append("\nLabels: " + ", ".join(LABELS))
        return "\n".join(lines)

    @staticmethod
    def _parse_response(text: str, expected: int) -> list[str]:
        """Parse the JSON map from the model; tolerate markdown fences and
        trailing prose via a regex fallback. Labels must be in the taxonomy."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        labels: list[str] = []
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                # Normalize like the regex fallback: models often return
                # capitalized/whitespace-padded labels in otherwise valid JSON.
                labels = [str(obj.get(str(i), "")).strip().lower()
                          for i in range(expected)]
        except json.JSONDecodeError:
            pass

        if len(labels) != expected:
            # Fallback: pull "N": "label" pairs out of free-form text.
            pat = re.compile(r'"?(\d+)"?\s*[:=]\s*"?([a-zA-Z]+)"?')
            found = {}
            for m in pat.finditer(text):
                idx, lab = int(m.group(1)), m.group(2).lower().strip()
                if lab in LABELS:
                    found.setdefault(idx, lab)
            labels = [found.get(i, "") for i in range(expected)]

        clean = [l if l in LABELS else "" for l in labels]
        if any(not l for l in clean):
            log.warning("LLM returned incomplete labels for %d/%d windows",
                        sum(bool(l) for l in clean), expected)
        return clean

    def _get_key(self) -> str:
        if self.cfg.classifier_provider == "anthropic":
            env = self.cfg.anthropic_api_key_env
            key = os.environ.get(env, "").strip()
            if not key:
                raise LLMError(
                    f"Set the ${env} environment variable to use the Anthropic "
                    "classifier, or switch classifier_provider to 'openai'."
                )
            return key
        env = self.cfg.openai_api_key_env
        key = os.environ.get(env, "").strip()
        if not key:
            raise LLMError(
                f"Set the ${env} environment variable to use the OpenAI-"
                "compatible classifier (can be a local server key)."
            )
        return key

    def _call_anthropic(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.cfg.anthropic_base_url.rstrip('/')}/v1/messages",
            headers={
                "x-api-key": self._get_key(),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.cfg.classifier_model,
                "max_tokens": self.cfg.classifier_max_tokens,
                "temperature": self.cfg.classifier_temperature,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.cfg.classifier_timeout_s,
        )
        if resp.status_code != 200:
            raise LLMError(f"Anthropic HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()["content"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected Anthropic response shape: {resp.text[:200]}") from exc

    def _call_openai(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.cfg.openai_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._get_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.cfg.classifier_model,
                "temperature": self.cfg.classifier_temperature,
                "max_tokens": self.cfg.classifier_max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.cfg.classifier_timeout_s,
        )
        if resp.status_code != 200:
            raise LLMError(f"OpenAI-compatible HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected OpenAI response shape: {resp.text[:200]}") from exc
