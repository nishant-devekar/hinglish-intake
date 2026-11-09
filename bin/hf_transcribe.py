#!/usr/bin/env python
"""Transcribe with a Hugging Face Whisper checkpoint from models/hf/.

    .venv/bin/python bin/hf_transcribe.py <audio.wav | chunks_dir> \
        --model indicwhisper-marathi --lang mr -o out/<name>/passes/indic-mr.txt

A directory is read as pre-split chunks (c000.wav, c001.wav, ...) of
--chunk seconds each, which is how the out/<call>/chunks folders are laid
out. A single file is split into --chunk second windows in memory.

Each window is decoded on its own (no carried context), so a repetition
loop in one window cannot spread into the next - the failure mode that
whisper.cpp hits on long Indic calls.
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000


def windows(src, chunk):
    if os.path.isdir(src):
        for f in sorted(glob.glob(os.path.join(src, "c*.wav"))):
            audio, sr = sf.read(f, dtype="float32")
            if sr != SR:
                sys.exit(f"{f}: expected {SR} Hz, got {sr}")
            yield int(os.path.basename(f)[1:-4]) * chunk, audio
    else:
        audio, sr = sf.read(src, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            sys.exit(f"{src}: expected {SR} Hz mono, got {sr} (ffmpeg -ar 16000 -ac 1)")
        step = chunk * SR
        for i in range(0, len(audio), step):
            yield i // SR, audio[i : i + step]


def load_processor(model_dir):
    try:
        return WhisperProcessor.from_pretrained(model_dir)
    except Exception:
        # Some checkpoints ship weights only; fall back to the stock tokenizer
        # for the same architecture size.
        layers = WhisperForConditionalGeneration.config_class.from_pretrained(model_dir).encoder_layers
        base = {4: "tiny", 6: "base", 12: "small", 24: "medium", 32: "large-v2"}[layers]
        print(f"no processor in {model_dir}, using openai/whisper-{base}", file=sys.stderr)
        return WhisperProcessor.from_pretrained(f"openai/whisper-{base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--model", required=True, help="folder name under models/hf/")
    ap.add_argument("--lang", default="mr")
    ap.add_argument("--chunk", type=int, default=12, help="seconds per window")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    model_dir = os.path.join(ROOT, "models", "hf", args.model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = load_processor(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device).eval()

    # Fine-tunes saved with old transformers carry forced_decoder_ids, which
    # newer generate() rejects alongside language=.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as out:
        for start, audio in windows(args.src, args.chunk):
            if len(audio) < SR // 4 or np.abs(audio).max() < 1e-3:
                text = ""
            else:
                feats = processor(audio, sampling_rate=SR, return_tensors="pt").input_features
                feats = feats.to(device, dtype=model.dtype)
                with torch.no_grad():
                    ids = model.generate(
                        feats,
                        language=args.lang,
                        task="transcribe",
                        num_beams=args.beams,
                        no_repeat_ngram_size=4,
                        max_new_tokens=200,
                    )
                text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            line = f"[{start // 60:02d}:{start % 60:02d}] {text}"
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()


if __name__ == "__main__":
    main()
