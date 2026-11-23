# ClipFarm

Paste a video link into Telegram and get back vertical (9:16) clips of its most viral moments, with captions.
You approve each clip with a tap, and approved clips queue up for posting to YouTube Shorts, TikTok and Instagram Reels.

Built to run free on an Oracle Cloud Always Free ARM VM.

```
Telegram link ─► yt-dlp ─► transcript ─► signals + LLM judge ─► snap cuts ─► 9:16 render + captions ─► R2 ─► Telegram review
                           (subs or       (loudness, replay      (sentences,   (facecam / face /
                            Whisper)       heatmap → Gemini,      shot cuts)     blur layouts)
                                           Haiku fallback)
```

## Status

| Step | State |
|---|---|
| 1. Telegram bot + job queue (SQLite) | ✅ |
| 2. Download (yt-dlp) + transcript (YouTube subs / faster-whisper) | ✅ |
| 3. Signals + Gemini judge with Claude Haiku fallback and a monthly cap | ✅ |
| 4. Render: boundary snapping, layout detection, burned captions, preview, R2 | ✅ |
| 5. Telegram approval (Post / Edit title / Skip / Undo) | ✅ |
| 6. Posting scheduler + YouTube / TikTok / Instagram uploaders | ⏳ next |
| 7. Oracle VM setup script + systemd services | ⏳ |
| Twitch signals: viewer clips (Helix API) + chat replay bursts and emotes | ✅ |
| Gemini watching candidate clips (visual re-scoring), laughter/scream detection | ⏳ |

## How clips are picked

| Job | Tool |
|---|---|
| Transcript | YouTube English subtitles (json3) when available, otherwise **faster-whisper** `small.en` |
| Audience signals | **Twitch VODs:** clips viewers made (weighted by views, Helix API) and chat replay bursts with emote mood (KEKW = funny, Pog = hype, monkaS = shock), shifted 5 s earlier for chat delay. **YouTube:** "most replayed" heatmap. **All:** loudness spikes vs. the surrounding minute. Weights: viewer clips 2.0, chat 1.5, replay 1.5, loudness 1.0, or 0.5 when any audience signal exists. |
| Judge | **Gemini** (free tier) reads the full timestamped transcript plus hot zones annotated with chat mood, and the list of viewer clip titles (which describe the on-screen action the transcript can't show), and returns scored clips as JSON. Up to 3 tries, then **Claude Haiku 5.5** with the same prompt and schema. Fallback spend is logged and capped per month. |
| Taste learning | Your 5 latest approved and 5 latest skipped clips are added to the prompt as examples |
| Clean cuts | Start/end snap to sentence boundaries, then to shot cuts within 0.8 s (ffmpeg scene detection); length forced into 20–60 s |
| Ranking | `final = 0.5 × LLM score + 0.5 × signal strength`, non-overlapping, top N |
| Layout | OpenCV **YuNet** face detection on 12 frames: small steady webcam face → `facecam` (cam 40% top / game 60% bottom); big face → `face` (9:16 crop on the speaker); else → `blur` (full frame over a blurred copy). Per-channel overrides in `presets.json`. |
| Captions | Whisper word timestamps on the clip, 3-word uppercase groups, the spoken word highlighted, burned in with ffmpeg/libass |

## Setup (local or on the VM)

Requirements: Python 3.10+, `ffmpeg` (with libass), and a font such as DejaVu Sans (`fonts-dejavu-core` on Ubuntu).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./scripts/fetch_models.sh          # YuNet face model (~230 KB); Whisper downloads itself on first use
cp .env.example .env               # then fill it in
cp presets.example.json presets.json   # optional
```

Fill in `.env` (all keys are described in `.env.example`):

- `TELEGRAM_BOT_TOKEN`: from @BotFather. `TELEGRAM_OWNER_ID`: your numeric id (ask @userinfobot). Only that user can use the bot.
- `GEMINI_API_KEY`: from Google AI Studio (free).
- `ANTHROPIC_API_KEY`: from console.anthropic.com. Optional, used only as the fallback. Also set a spend limit in the console.
- `R2_*`: Cloudflare R2 bucket and keys. Optional; without them, clips stay local only.
- `TWITCH_CLIENT_ID` / `TWITCH_CLIENT_SECRET`: free app at dev.twitch.tv/console/apps. Enables viewer clips for Twitch VODs. Chat replay needs no credentials (`TWITCH_CHAT=1`), but it uses Twitch's unofficial web endpoint: if Twitch changes it, jobs continue without chat and the status message says so.
- `YTDLP_COOKIES`: a Netscape cookies file for YouTube. Usually needed from datacenter IPs ("Sign in to confirm you're not a bot").

## Run

Two processes share `data/clipfarm.db`:

```bash
.venv/bin/python -m clipfarm.bot      # Telegram: links in, approvals
.venv/bin/python -m clipfarm.worker   # processes one link at a time
```

Then send the bot a link. You'll get a status message that updates as the job runs, then each clip as a 720p preview with **✅ Post · ✏️ Edit title · ⏭ Skip**.
Bot commands: `/queue`, `/stats`.

Without Telegram (for testing):

```bash
.venv/bin/python -m clipfarm.cli process "https://www.youtube.com/watch?v=..." --clips 3
.venv/bin/python -m clipfarm.cli cleanup
```

## Storage

- `data/work/job-N/`: the source download and audio, deleted when the job ends.
- `data/clips/job-N/clip-XX.mp4` (+ `.preview.mp4`): kept `LOCAL_RETENTION_DAYS` days for posting.
- R2 `clips/YYYY-MM-DD/job-N/clip-XX.mp4` + `.json` metadata: deleted after `R2_RETENTION_DAYS` days (the worker runs cleanup every 6 h).

## Tests

```bash
.venv/bin/python -m pytest -q
```

Unit tests cover parsing, signals, snapping, layout math, captions, the DB and the judge fallback and cap. The end-to-end test renders real clips with ffmpeg from a synthetic video, with yt-dlp, the LLM and Whisper stubbed.

## Code map

```
clipfarm/
├── bot.py           Telegram bot (python-telegram-bot)
├── worker.py        job loop, Telegram notifications, cleanup
├── cli.py           run the pipeline without Telegram
├── config.py        .env settings
├── db.py            SQLite: jobs, clips, llm_usage, kv
├── messages.py      Telegram texts + buttons
├── notify.py        tiny sync Telegram client for the worker
├── media.py         ffmpeg/ffprobe helpers, scene cuts
├── storage.py       R2 upload + retention, local cleanup
└── pipeline/
    ├── run.py         orchestrates one link end to end
    ├── download.py    yt-dlp: video, heatmap, English subtitles
    ├── transcribe.py  json3 parser, faster-whisper
    ├── signals.py     loudness, heatmap, Twitch signals → weighted hot zones
    ├── twitch.py      viewer clips (Helix) + chat replay (GQL) → signals
    ├── judge.py       prompt, schema, Gemini → Haiku fallback
    ├── select.py      boundary snapping, overlap removal
    ├── layout.py      face detection → facecam / face / blur
    ├── captions.py    word groups → ASS subtitles
    └── render.py      ffmpeg filter graphs, preview
```

## Content rights

Only clip content you own or have permission to repost. Platforms detect reuploads quickly, and strikes can ban all connected accounts.
