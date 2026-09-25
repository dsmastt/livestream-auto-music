"""Self-contained tests for the pure logic (no network, no GPU, no heavy deps).

Run standalone:  python tests/test_core.py
Or with pytest:   pytest tests/ (optional)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from vibe.audio import iter_windows
from vibe.chat import load_chat, normalize_message
from vibe.classify import LLMClassifier, debounce
from vibe.config import Config, load_config
from vibe.errors import VibeError
from vibe.twitch_meta import parse_duration

TESTS = []


def test(desc):
    def deco(fn):
        TESTS.append((desc, fn))
        return fn
    return deco


@test("parse_duration handles Twitch formats")
def _():
    assert parse_duration("1h23m45s") == 3600 + 23 * 60 + 45
    assert parse_duration("45s") == 45
    assert parse_duration("2h") == 7200
    assert parse_duration("1m30s") == 90
    try:
        parse_duration("nonsense")
        raise AssertionError("should have raised")
    except VibeError:
        pass


@test("config refuses large whisper models")
def _():
    cfg = Config()
    assert cfg.whisper_model == "small"
    try:
        Config(whisper_model="large-v3").validate()
        raise AssertionError("should have raised")
    except VibeError as e:
        assert "large" in str(e)
    # YAML load path + CLI override
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.yaml"
        p.write_text("whisper_model: base\nmax_vod_minutes: 30\n")
        cfg = load_config(str(p), overrides={"max_vod_minutes": 10.0})
        assert cfg.whisper_model == "base"
        assert cfg.max_vod_minutes == 10.0
    try:
        load_config(None, overrides={"nope_key": 1})
        raise AssertionError("should have raised")
    except VibeError:
        pass


@test("debounce commits only on 2-of-3 agreement")
def _():
    labels = ["hype", "hype", "chill", "chill", "chill", "tense", "tense", "tense"]
    committed = debounce(labels)
    assert committed[:2] == ["", ""]
    # window 2: hype,hype,chill -> hype (2 of 3)
    assert committed[2] == "hype"
    # window 3: hype,chill,chill -> chill
    assert committed[3] == "chill"
    assert committed[4] == "chill"
    # window 5: chill,chill,tense -> chill still leads
    assert committed[5] == "chill"
    # window 6: chill,tense,tense -> tense
    assert committed[6] == "tense"
    assert committed[7] == "tense"
    # no agreement -> no commit
    assert debounce(["hype", "chill", "tense"])[2] == ""


@test("chat normalization handles TwitchDownloaderCLI and tcd shapes")
def _():
    cli_msg = {
        "_id": "x", "created_at": "2024-01-01T00:00:00Z",
        "content_offset_seconds": 12.5,
        "commenter": {"display_name": "Alice", "_id": "1", "name": "alice"},
        "message": {"body": "hello world", "fragments": [{"text": "hello"}]},
        "is_action": False,
    }
    n = normalize_message(cli_msg)
    assert n["offset_seconds"] == 12.5
    assert n["author"] == "Alice"
    assert n["text"] == "hello world"

    tcd_msg = {"content_offset_seconds": 3, "commenter": {"name": "bob"},
               "message": "pog"}
    n2 = normalize_message(tcd_msg)
    assert n2["text"] == "pog" and n2["author"] == "bob"

    assert normalize_message({"created_at": "x"}) is None
    assert normalize_message("junk") is None
    assert normalize_message({"content_offset_seconds": 1, "message": "   "}) is None


@test("load_chat reads CLI JSON array and JSON-lines")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.json"
        p.write_text(json.dumps([
            {"content_offset_seconds": 1, "commenter": {"display_name": "a"},
             "message": {"body": "one"}},
            {"content_offset_seconds": 2, "commenter": {"display_name": "b"},
             "message": {"body": "two"}},
        ]))
        msgs = load_chat(p)
        assert [m["text"] for m in msgs] == ["one", "two"]

        p2 = Path(d) / "b.json"
        p2.write_text(
            '{"content_offset_seconds": 1, "commenter": {"name": "x"}, "message": "hi"}\n'
        )
        assert load_chat(p2)[0]["text"] == "hi"


@test("iter_windows yields rolling 15s/5s windows from streamed wav")
def _():
    sr, secs = 16000, 30
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.wav"
        sig = (np.sin(2 * np.pi * 440 * np.arange(sr * secs) / sr) * 0.3).astype(np.float32)
        sf.write(str(p), sig, sr)
        wins = list(iter_windows(p, 15.0, 5.0, sample_rate=sr))
        assert [round(w[0]) for w in wins] == [0, 5, 10, 15]
        assert [round(w[1]) for w in wins] == [15, 20, 25, 30]
        for _, _, audio in wins:
            assert audio.shape == (15 * sr,)
            assert audio.dtype == np.float32
        # max_seconds cap
        wins_capped = list(iter_windows(p, 15.0, 5.0, max_seconds=16.0, sample_rate=sr))
        assert [round(w[0]) for w in wins_capped] == [0]
        # sub-window length -> empty
        assert list(iter_windows(p, 15.0, 5.0, max_seconds=3.0)) == []


@test("LLM response parsing tolerates fences/mess and enforces taxonomy")
def _():
    text = '```json\n{"0": "hype", "1": "chill"}\n```'
    assert LLMClassifier._parse_response(text, 2) == ["hype", "chill"]
    messy = 'Sure! Window 0: "HYPE", 1: "comedic", and 2: "tense".'
    assert LLMClassifier._parse_response(messy, 3) == ["hype", "comedic", "tense"]
    bad = '{"0": "angry", "1": "chill"}'
    out = LLMClassifier._parse_response(bad, 2)
    assert out[0] == "" and out[1] == "chill"


class _FakeClassifier(LLMClassifier):
    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    def _request_batch(self, windows, game_tag):
        self.calls += 1
        return ["hype" if i % 2 == 0 else "chill" for i in range(len(windows))]


@test("classifier batches and short-circuits empty windows")
def _():
    cfg = Config(classifier_batch_size=2, skip_llm_on_empty=True)
    fc = _FakeClassifier(cfg)
    windows = [
        {"start": 0, "end": 15, "transcript": "hi", "chat": "x"},     # llm
        {"start": 5, "end": 20, "transcript": "", "chat": ""},        # skip
        {"start": 10, "end": 25, "transcript": "yo", "chat": ""},     # llm
    ]
    labels, calls = fc.classify(windows, "Just Chatting")
    assert labels == ["hype", "waiting", "chill"]
    assert calls == 1  # windows 0 and 2 share one batch of size 2
    fc2 = _FakeClassifier(cfg)
    labels2, calls2 = fc2.classify(
        [{"start": 0, "end": 15, "transcript": "", "chat": ""}], None)
    assert labels2 == ["waiting"] and calls2 == 0


@test("download_audio keeps the downloaded file when already mono 16k")
def _():
    """Regression test: ensure_mono_16k's return value must be honored, or the
    only real audio file gets deleted and a nonexistent path returned."""
    import subprocess
    from vibe import audio as audio_mod

    sr = 16000
    orig_find, orig_run = audio_mod.find_tool, audio_mod.run_cmd
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        vod_id = "777"
        raw = d / f"{vod_id}_raw.wav"
        sig = (np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr) * 0.3).astype(np.float32)
        sf.write(str(raw), sig, sr)  # pre-made mono 16k "download"

        audio_mod.find_tool = lambda name, hint: "/fake/tool"
        audio_mod.run_cmd = lambda cmd, timeout=None, cwd=None: subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr="")

        try:
            out = audio_mod.download_audio(vod_id, d, sr)
            assert out == raw, f"expected {raw}, got {out}"
            assert raw.exists(), "downloaded audio must not be deleted"
        finally:
            audio_mod.find_tool, audio_mod.run_cmd = orig_find, orig_run


@test("download_audio deletes raw only when a real conversion happened")
def _():
    """The other branch of the same bug: raw at 48k stereo must be converted,
    the converted file returned, and raw deleted."""
    import subprocess
    from vibe import audio as audio_mod

    sr = 16000
    orig_find, orig_run = audio_mod.find_tool, audio_mod.run_cmd
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        vod_id = "778"
        raw = d / f"{vod_id}_raw.wav"
        final = d / f"{vod_id}_16k_mono.wav"
        sig = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3).astype(np.float32)
        sf.write(str(raw), np.column_stack([sig, sig]), 48000)  # 48k stereo

        def fake_run(cmd, timeout=None, cwd=None):
            if "ffmpeg" in cmd[0]:  # find_tool is mocked to /fake/tool
                dst = Path(cmd[-1])
                sf.write(str(dst), sig, sr)  # "converted" mono 16k
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        audio_mod.find_tool = lambda name, hint: f"/fake/{name}"
        audio_mod.run_cmd = fake_run
        try:
            out = audio_mod.download_audio(vod_id, d, sr)
            assert out == final
            assert final.exists() and not raw.exists()
        finally:
            audio_mod.find_tool, audio_mod.run_cmd = orig_find, orig_run


@test("LLM parse normalizes case in the primary JSON path")
def _():
    assert LLMClassifier._parse_response('{"0": "Chill", "1": " HYPE "}', 2) == \
        ["chill", "hype"]


@test("evaluate matches GT to a window containing the timestamp")
def _():
    import evaluate  # top-level script, importable from project root

    windows = [
        {"start": 0, "end": 15, "label": "chill"},
        {"start": 5, "end": 20, "label": "tense"},
        {"start": 10, "end": 25, "label": "hype"},
        {"start": 15, "end": 30, "label": "chill"},
    ]
    # ts=14 lies inside [0,15), [5,20), [10,25) but NOT [15,30);
    # start-distance would wrongly pick the last window (chill).
    pairs, matched, no_pred = evaluate.match([(14.0, "tense")], windows, tolerance=15.0)
    assert matched == 1 and pairs[0] == ("tense", "tense"), pairs
    # ts far from any window -> no-prediction
    _, matched2, no_pred2 = evaluate.match([(100.0, "hype")], windows, tolerance=15.0)
    assert matched2 == 0 and no_pred2 == 1
    # non-numeric vod ids are rejected by the CLI
    try:
        rc = evaluate.main(["--gt-dir", "/tmp", "--vod-ids", "../x"])
    except SystemExit as exc:
        rc = exc.code
    assert rc == 2


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
