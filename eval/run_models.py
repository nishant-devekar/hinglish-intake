"""Run the candidate ASR models over eval/clips and time them.

    .venv/bin/python eval/run_models.py tara qwen3 vibevoice [--clips a,b]

Writes eval/results/<clip>.<model>.txt and appends to eval/results/timings.tsv,
the same layout run_whisper_cpp.sh uses, so everything compares side by side.
"""
import sys, time, pathlib
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIPS = ROOT / "eval" / "clips"
RES = ROOT / "eval" / "results"
HF = ROOT / "models" / "hf"

# Same hotwords transcribe.py uses (config/hotwords.txt, else a short default).
sys.path.insert(0, str(ROOT / "bin"))
from transcribe import DEFAULT_HOTWORDS as HOTWORDS


def split_at_pauses(audio, sr=16000, target=25.0, search=5.0):
    """Cut long audio into <=30 s pieces, each cut at the quietest point
    within `search` seconds before the target length, so words aren't split."""
    hop = int(0.02 * sr)
    pieces, start = [], 0
    while start < len(audio):
        end = start + int(target * sr)
        if end >= len(audio) - int(3 * sr):
            pieces.append(audio[start:]); break
        lo = end - int(search * sr)
        win = audio[lo:end + int(search * sr)]
        energy = np.convolve(win ** 2, np.ones(hop) / hop, mode="same")
        cut = lo + int(np.argmin(energy[::hop]) * hop)
        pieces.append(audio[start:cut]); start = cut
    return pieces


class Tara:
    name = "tara-mixed"

    def __init__(self):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.torch = torch
        path = HF / "tara"
        self.proc = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float16).to("mps").eval()
        tk = self.proc.tokenizer
        # Transformers 5 dropped forced_decoder_ids; give the whole Whisper
        # prefix as the decoder prompt instead: sot, hi, mixed-code, transcribe.
        self.prefix = torch.tensor([[tk.convert_tokens_to_ids(t) for t in
                                     ("<|startoftranscript|>", "<|hi|>", "<|mixedcode|>",
                                      "<|transcribe|>", "<|notimestamps|>")]], device="mps")

    def __call__(self, audio):
        out = []
        for piece in split_at_pauses(audio):
            feats = self.proc(piece, sampling_rate=16000, return_tensors="pt") \
                        .input_features.to("mps", self.torch.float16)
            from transformers import GenerationMixin
            with self.torch.no_grad():
                ids = GenerationMixin.generate(self.model, input_features=feats,
                                               decoder_input_ids=self.prefix,
                                               max_new_tokens=440, num_beams=1,
                                               repetition_penalty=1.1)
            ids = ids[0, self.prefix.shape[1]:]
            out.append(self.proc.tokenizer.decode(ids, skip_special_tokens=True).strip())
        return "\n".join(out)


class MLX:
    def __init__(self, name, repo, **kw):
        from mlx_audio.stt.utils import load_model
        self.name, self.kw = name, kw
        self.model = load_model(str(HF / repo))

    def __call__(self, audio):
        r = self.model.generate(audio, **self.kw)
        if getattr(r, "segments", None) and any("speaker" in s or "speaker_id" in s for s in r.segments):
            lines = []
            for s in r.segments:
                spk = s.get("speaker", s.get("speaker_id", "?"))
                lines.append(f"[{s.get('start', 0):6.1f}s S{spk}] {s.get('text', '').strip()}")
            return "\n".join(lines)
        return r.text


def build(key):
    if key == "tara":
        return Tara()
    if key == "qwen3":
        return MLX("qwen3-asr", "qwen3-asr-1.7b-8bit", hotwords=HOTWORDS,
                   repetition_penalty=1.05)
    if key == "qwen3-hi":
        return MLX("qwen3-asr-hi", "qwen3-asr-1.7b-8bit", language="Hindi",
                   hotwords=HOTWORDS, repetition_penalty=1.05)
    if key == "vibevoice":
        return MLX("vibevoice", "vibevoice-asr-4bit", hotwords=HOTWORDS,
                   sampling_rate=16000, repetition_penalty=1.05)
    raise SystemExit(f"unknown model {key}")


def main():
    args = sys.argv[1:]
    only = None
    if "--clips" in args:
        i = args.index("--clips"); only = set(args[i + 1].split(",")); del args[i:i + 2]
    clips = sorted(p for p in CLIPS.glob("*.wav") if only is None or p.stem in only)
    for key in args:
        t0 = time.time(); m = build(key); load = time.time() - t0
        print(f"== {m.name} (load {load:.1f}s)", flush=True)
        for c in clips:
            audio, sr = sf.read(c, dtype="float32")
            assert sr == 16000
            t = time.time(); text = m(audio); dt = time.time() - t
            (RES / f"{c.stem}.{m.name}.txt").write_text(text + "\n")
            with open(RES / "timings.tsv", "a") as f:
                f.write(f"{c.stem}\t{m.name}\t{dt:.1f}\n")
            print(f"   {c.stem}: {dt:.1f}s, {len(text.split())} words", flush=True)
        del m


if __name__ == "__main__":
    main()
