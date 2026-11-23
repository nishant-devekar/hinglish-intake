---
name: hinglish-intake
description: Transcribe calls, voice notes and screen recordings in English, Hindi, Hinglish or Marathi into speaker-labelled transcripts, per-scene screenshots and meeting notes with [file @ mm:ss] citations, all locally on an Apple Silicon Mac. Use when the user gives an audio or video file (mp3, m4a, wav, mp4, mov…) and wants a transcript, who said what, what was on screen, or notes, decisions and action items from it, especially for Indian business calls that mix Hindi and English.
---

# Hinglish Intake

You can't listen to audio or watch video. This skill turns a recording into
text and images you can read. Everything runs on the user's Mac, so the
recording never leaves it. `SKILL_DIR` below means the folder this file is in.

## 1. Check the machine first

Run these before anything else. Stop and tell the user if one fails:

- `uname -sm` must print `Darwin arm64` (macOS on Apple Silicon). Nothing here works on Linux, Windows or Intel Macs.
- `command -v ffmpeg uv` must find both. If not, they're installed with `brew install ffmpeg uv`.
- `test -x "$SKILL_DIR/.venv/bin/python"` and `test -d "$SKILL_DIR/models/hf/tara-mlx"` tell you whether setup was already done.

## 2. One-time setup (ask before running)

Setup downloads about 3 GB of models and takes several minutes. **Tell the user
the size and get a yes before you start.** Then, from `SKILL_DIR`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
bin/fetch_models.sh
```

Optional extras, each its own download. Offer them only when needed:

| flag | adds | size | when |
|---|---|---|---|
| `--marathi` | Marathi model + language ID | ~1.5 GB | the recording has Marathi |
| `--qwen3` | Qwen3-ASR | 2.3 GB | a second opinion, or mostly-English audio |
| `--vibevoice` | VibeVoice-ASR | 5.3 GB | hour-long meetings where who-said-what matters more than exact words |
| `--notes-local` | Gemma 12B | ~7 GB | notes without Claude |

For on-screen text in videos (`--ocr`), the Xcode Command Line Tools must be
installed (`xcode-select -p` succeeds).

## 3. Transcribe

```bash
cd "$SKILL_DIR"
bin/ingest.sh /path/to/call.mp3 --speakers 3     # pass --speakers when you know the head-count
bin/ingest.sh /path/to/demo.mp4 --ocr            # video: one screenshot per scene + on-screen text
```

A 22-minute call takes about 2 minutes on an M4 Pro. Run long files in the
background. Output goes to `SKILL_DIR/out/<file name>/`. Note that a second
run on a file with the same name replaces the earlier output.

- `transcript.txt`: `[mm:ss] S1: …` lines. Read this first.
- `transcript.json`: the same with exact times and each line's language (`en`, `hi`, `mr`).
- `frames/frame-0001.jpg` …: screenshots you can look at; `frames/index.txt` maps them to times; `frames/ocr.txt` has the on-screen text.
- `timeline.md`: for videos, each screenshot with what was said while it showed. `evidence.pdf` is the same with images, for attaching in claude.ai.

`--engine` defaults to `auto`, which is right for almost everything. Use
`qwen3` for a second opinion or `vibevoice` for very long meetings. If
`ingest.sh` prints "no .venv - falling back to whisper.cpp", setup hasn't
been done: go back to step 2 rather than relying on that fallback.

## 4. Answer from the transcript

- **Cite every claim** as `[<file> @ mm:ss]`, using the time on the transcript line. If you can't find a line that supports a claim, say so instead of guessing.
- Keep `S1`, `S2` labels as they are. Don't guess who a speaker is from what they say.
- Hinglish lines have English in Latin script and Hindi in Devanagari. Answer in English unless asked otherwise, and quote the original when exact wording matters.
- Speech recognition makes mistakes, especially with names and numbers. For anything that matters (a price, a date, a commitment), quote the line and suggest the user check it by ear in `audio.wav` at that time.

For meeting notes, use these sections: Summary, Decisions, Action items
(Owner | Action | Due | Source), Open questions. Every bullet ends with a citation.

## Speaker names and privacy

Transcripts can say `Asha:` instead of `S1:` once people are enrolled:

```bash
.venv/bin/python bin/voices.py enroll-from out/<file name> S2 Asha   # "S2 in that call was Asha"
.venv/bin/python bin/voices.py list
```

Voice prints are biometric data. **Only enrol someone when the user asks you
to, and remind them to get that person's consent.** Prints stay in
`voices/prints.json` on this Mac. Never copy them, transcripts or recordings
anywhere else unless the user asks.

Misheard names and jargon can be fixed for future runs by adding
`regex<TAB>fix` lines to `config/corrections.tsv` (start from
`config/corrections.example.tsv`).

## More

`README.md` has every option, the team mode (uploads through a Hugging Face
Space) and the model licenses. `eval/REPORT.md` explains why the pipeline
uses these models.
