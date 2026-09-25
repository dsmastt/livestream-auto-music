"""Integration test: run process_vod end-to-end with external tools mocked.

Covers: disk check -> metadata -> (fake) audio+chat download -> chunk ->
transcribe -> chat align -> classify -> debounce -> JSONL+meta output ->
temp cleanup with freed-bytes logging.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from vibe import audio as audio_mod
from vibe import chat as chat_mod
from vibe import pipeline
from vibe.classify import LLMClassifier
from vibe.config import Config
from vibe.pipeline import process_vod
from vibe.transcribe import Transcriber

TESTS = []


def test(desc):
    def deco(fn):
        TESTS.append((desc, fn))
        return fn
    return deco


@test("process_vod produces JSONL + meta, applies debounce, cleans up")
def _():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        work_root, out_dir = d / "work", d / "results"
        vod_id = "111"

        # --- fake downloads: write a real 60s mono wav + chat JSON -----------
        sr, secs = 16000, 60
        sig = (np.sin(2 * np.pi * 440 * np.arange(sr * secs) / sr) * 0.3).astype(np.float32)

        def fake_download_audio(vod, work, sample_rate, max_minutes=None, **kw):
            work.mkdir(parents=True, exist_ok=True)
            p = work / f"{vod}_16k_mono.wav"
            sf.write(str(p), sig, sr)
            return p

        chat_msgs = [
            {"content_offset_seconds": 2, "commenter": {"display_name": "a"},
             "message": {"body": "first msg"}},
            {"content_offset_seconds": 7, "commenter": {"display_name": "b"},
             "message": {"body": "second msg"}},
            {"content_offset_seconds": 40, "commenter": {"display_name": "c"},
             "message": {"body": "later msg"}},
        ]

        def fake_download_chat(vod, out_path, tool="auto", end_seconds=None, timeout=None):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(chat_msgs))
            return "fake"

        audio_mod.download_audio = fake_download_audio
        chat_mod.download_chat = fake_download_chat
        # pipeline imports fetch_vod_metadata by name, so patch it there
        pipeline.fetch_vod_metadata = lambda *a, **k: {
            "id": vod_id, "title": "Test Stream", "duration_seconds": 60,
            "view_count": 100, "created_at": "2024-01-01T00:00:00Z",
            "game_name": "Just Chatting",
        }

        # --- fake transcriber / classifier ------------------------------------
        class FakeTranscriber(Transcriber):
            def __init__(self, config):
                super().__init__(config)
                self.n = 0

            def load(self):
                pass

            def unload(self):
                pass

            def transcribe(self, audio):
                out = "hello world" if self.n % 4 < 2 else ""
                self.n += 1
                return out

        class FakeClassifier(LLMClassifier):
            def _request_batch(self, windows, game_tag):
                # deterministic pattern: hype, hype, chill, chill, ...
                return ["hype" if i % 4 < 2 else "chill"
                        for i in range(len(windows))]

        t_real = pipeline.Transcriber
        pipeline.Transcriber = FakeTranscriber
        pipeline.LLMClassifier = FakeClassifier

        try:
            cfg = Config(metadata_required=True, skip_llm_on_empty=False,
                         min_free_disk_gb=0.5)  # /tmp may have < 5 GiB free
            result = process_vod(vod_id, out_dir, cfg, work_root=work_root,
                                 progress_every_windows=100)
        finally:
            pipeline.Transcriber = t_real
            pipeline.LLMClassifier = LLMClassifier

        assert result.ok, result.error
        assert result.windows == 10         # 60s -> starts 0,5,...,45
        assert result.title == "Test Stream"
        assert result.game_name == "Just Chatting"
        assert result.processed_seconds == 60.0

        lines = [json.loads(l) for l in (out_dir / f"{vod_id}.jsonl").read_text().splitlines()]
        assert len(lines) == 10
        assert set(lines[0]) == {"start_time", "end_time", "transcript_snippet",
                                 "chat_snippet", "chat_rate_per_minute",
                                 "raw_label", "committed_label"}
        # debounce: first 2 windows uncommitted, then commits as 2-of-3 agree
        assert lines[0]["committed_label"] == ""
        assert lines[1]["committed_label"] == ""
        assert all(l["committed_label"] in ("hype", "chill") for l in lines[2:])
        # raw labels follow the fake pattern
        assert [l["raw_label"] for l in lines[:4]] == ["hype", "hype", "chill", "chill"]
        # chat bucketing: window 0-15 has messages at 2s and 7s
        assert lines[0]["chat_rate_per_minute"] == round(2 / 0.25, 2)

        meta = json.loads((out_dir / f"{vod_id}.meta.json").read_text())
        assert meta["n_windows"] == 10
        assert meta["label_counts"]

        # cleanup: work dir gone, freed bytes logged
        assert not (work_root / vod_id).exists(), "temp dir must be deleted"
        assert result.disk_freed_bytes > 0

    print("  integration output verified")


@test("process_vod rejects non-numeric VOD IDs before touching disk")
def _():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        result = process_vod("../evil", d / "results", Config(),
                             work_root=d / "work")
        assert not result.ok
        assert "must be numeric" in (result.error or "")
        assert not (d / "work").exists()  # nothing was created


if __name__ == "__main__":
    failed = 0
    for desc, fn in TESTS:
        try:
            fn()
            print(f"PASS  {desc}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL  {desc}: {exc}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
