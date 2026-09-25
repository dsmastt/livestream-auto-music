# Twitch VOD "Vibe Detection" Pipeline

Automatically pulls **audio, chat replay, and metadata** for a list of Twitch
VOD IDs, runs an ASR + LLM mood-classification pipeline over each VOD, and
writes per-VOD timestamped mood logs (JSONL). Built for a personal machine
with an **8 GB VRAM GPU**, a **~6 GB RAM budget per process**, and a
**~10-15 GB disk budget** for the whole project.

**Out of scope:** live capture, OBS integration, music playback.

## Pipeline (per VOD)

1. Check free disk space; abort before downloading anything if below the
   safety threshold.
2. Pull metadata via Twitch Helix (title, duration, view count, created_at,
   starting game/category — *if* the API returns one; see Limitations).
3. Download **audio-only** (yt-dlp `-x`, mono 16 kHz WAV). The video stream is
   never written to disk.
4. Download the full chat replay as JSON (TwitchDownloaderCLI, or the `tcd`
   Python package as fallback).
5. Chunk audio into rolling 15 s windows with a 5 s hop (streamed — never all
   in RAM), transcribe with faster-whisper `small`.
6. Bucket chat messages into each window (message rate + raw message snippet),
   and classify each window with an LLM (5 labels: hype / comedic / chill /
   waiting / tense) using transcript + chat snippet + starting game tag.
7. Debounce: commit a label only after **2 of the last 3 windows agree**.
8. Write `results/<vod_id>.jsonl` + `<vod_id>.meta.json`, then **delete every
   intermediate file** (raw audio, chat JSON) and log how much space was freed.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -U yt-dlp                     # audio download
sudo apt install ffmpeg libsndfile1       # ffmpeg + soundfile backend (Debian/Ubuntu)
# macOS: brew install ffmpeg   |  Windows: winget install Gyan.FFmpeg
```

**TwitchDownloaderCLI** (chat replay; recommended, much faster):

- Download the latest release binary from
  https://github.com/lay295/TwitchDownloader/releases
- Put it somewhere on your `PATH` (e.g. `/usr/local/bin/`) and make it
  executable.

**Fallback if you can't install the CLI:** `pip install tcd` (the Twitch Chat
Downloader Python package), or set `chat_tool: tcd` in the config. If neither
is available the pipeline stops with printed install instructions — it never
fails silently.

**Twitch API credentials** (for metadata; optional but recommended):

1. Create an app at https://dev.twitch.tv/console/apps → get a Client ID.
2. Get an access token (e.g. via the Twitch CLI:
   `twitch token -u -s "user:read:email"` or an app token).
3. Export them:

```bash
export TWITCH_CLIENT_ID=your_client_id
export TWITCH_ACCESS_TOKEN=your_access_token
```

Without these, the pipeline still runs but metadata (title/game) is missing —
metadata is non-fatal by default.

**LLM classifier key** (required for classification):

```bash
export ANTHROPIC_API_KEY=...      # for classifier_provider: anthropic
# or
export OPENAI_API_KEY=...         # for classifier_provider: openai
```

## Test a single VOD first

```bash
# 10 minutes of one VOD, everything else default:
python run_batch.py --vod-ids 123456789 --output-dir ./results/ --max-vod-minutes 10

# or check the plan + disk headroom without downloading anything:
python run_batch.py --vod-ids 123456789 --dry-run
```

## Run the full batch

```bash
python run_batch.py --vod-ids 123456789,987654321,555555555 --output-dir ./results/
```

Outputs per VOD:

- `results/<vod_id>.jsonl` — one line per window:
  `start_time, end_time, transcript_snippet, chat_snippet, chat_rate_per_minute,
  raw_label, committed_label`
- `results/<vod_id>.meta.json` — title, game tag, durations, window count,
  label distribution, elapsed time.

## Evaluate against manual ground truth

Create `gt/<vod_id>.csv` per VOD (seconds or `HH:MM:SS` timestamps):

```csv
timestamp,label
12.5,hype
75,chill
01:02:30,tense
```

```bash
python evaluate.py --results-dir results/ --gt-dir gt/
python evaluate.py --results-dir results/ --gt-dir gt/ --tolerance 10
```

Matching: each GT row is matched to the nearest window with a **committed**
label within `--tolerance` seconds. Reports per-VOD agreement % + confusion
matrix, a combined matrix, and per-label precision/recall.

## Disk / RAM safety limits and how to adjust them

All limits live in `config.yaml` (copy `config.example.yaml`) or as CLI flags.

| Limit | Default | Where | What it does |
|---|---|---|---|
| Free-disk threshold | 5 GiB | `min_free_disk_gb` | Aborts **before** any download if free space is below this |
| Whisper model cap | `small` | `whisper_model` | `large*` is refused by config validation to protect VRAM/RAM |
| Whisper device | `auto` | `whisper_device` | CUDA if available, else CPU **with a printed warning** |
| Per-VOD audio cap | none | `--max-vod-minutes` / `max_vod_minutes` | Caps the *processed/kept* audio length. Note: yt-dlp still downloads the full file before cutting, so the download footprint is guarded separately by a duration-based free-space estimate |
| RAM ceiling | ~1-2 GB | implicit | Audio is streamed in 15 s windows (~1 MB); the model is loaded/unloaded per VOD; only final JSONL/meta files persist |

Hard invariants you don't need to configure:

- **Never stores video** — yt-dlp is invoked with `-x` (audio-only) and the
  audio is converted to mono 16 kHz at download time.
- **Strictly sequential** — VODs are processed one at a time; there is no
  concurrency anywhere in the pipeline.
- **Per-VOD cleanup** — the work directory is wiped after each VOD and the
  freed bytes are logged, along with running disk usage and elapsed time
  between VODs so you can watch progress and kill the run if needed.

To adjust the disk threshold: `min_free_disk_gb: 8` (raise it), or lower it
carefully (not recommended below 2 GiB — a mono 16 kHz WAV needs
~115 MB per hour plus chat JSON).

To lower RAM/VRAM use: `whisper_model: base`, `whisper_vad: true` (default),
smaller `classifier_batch_size` (marginal). To raise accuracy: `whisper_model:
medium`, `whisper_beam_size: 5`.

## Config reference

See `config.example.yaml` for every knob, including switching the classifier
to an OpenAI-compatible endpoint (works for local Qwen-VL / DeepSeek-style
servers: set `classifier_provider: openai`,
`openai_base_url: http://localhost:8000/v1`).

## Limitations

- **Mid-VOD game/category changes:** Twitch's Helix `Get Videos` endpoint does
  NOT reliably return category changes mid-stream. We capture the single
  `game_name` the API returns (usually the starting category) as a static
  context tag and deliberately do not scrape around this.
- Chat replay timestamps (`content_offset_seconds`) occasionally drift by a
  second or two vs. audio; windows are 15 s wide so this rarely matters.
- Windows at the very end of a VOD shorter than `window_sec` are dropped.
- Debounced labels: the first `debounce_window - 1` windows always have an
  empty `committed_label` by design.
- With `skip_llm_on_empty: true`, silent windows with no chat are labeled
  `waiting` without an LLM call (cost saving; disable in config to classify
  everything).
