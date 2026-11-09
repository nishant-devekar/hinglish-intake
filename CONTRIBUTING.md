# Contributing

Thanks for helping. This project exists because speech tools often get
Indian business calls wrong, so the most valuable contributions are the ones
that make it work for more people's calls.

## Good first contributions

- **More languages.** Tamil, Telugu, Bengali, Gujarati, Kannada… Add an
  engine in `bin/transcribe.py` (see `marathi_engine`) and a routing rule in
  the `auto` engine.
- **Linux / CUDA.** The pipeline is MLX-first. A PyTorch path exists
  (`tara-torch`), but OCR (`bin/ocr.swift`) and the worker service
  (`deploy/install_worker.sh`) are macOS-only.
- **Gold transcripts.** A hand-corrected 10-minute stretch of a call gives
  real WER numbers (`eval/gold/README.md`). Share only recordings you have
  the right to share.
- **Bug reports** with the command you ran, the engine, and the relevant
  part of the log (`summary.txt` or the job's log).

## Ground rules

- Never commit recordings, transcripts, voice prints or real people's names.
  `.gitignore` keeps `inbox/`, `out/`, `voices/`, `models/` and `config/`
  out; keep it that way. Use pseudonyms in examples and reports.
- Quality claims need numbers. If a change affects transcripts, say what you
  measured and how (`eval/score.py`, `eval/gold/score_gold.py`).
- Keep the default path free and local: no paid APIs required to get a
  transcript.

## Development

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
bin/fetch_models.sh
bin/ingest.sh some-test-clip.mp3 --speakers 2
```

By contributing, you agree your contributions are licensed under Apache-2.0.
