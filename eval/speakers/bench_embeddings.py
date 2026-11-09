"""Which speaker-embedding model tells Asha / Ravi / Meera apart best?

Ground truth is Google Meet captions ("Name: text" cues, aligned to the
recordings at 0 s offset). Voice prints come from the first part of the
first call in calls.json; tests are the rest of that call and every other call.
"""
import sys, pathlib, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bin"), str(ROOT / "eval" / "speakers")]
import json
import sherpa_onnx, transcribe as T, srt

# Your calls, in eval/speakers/calls.json (gitignored: it points at private audio):
# {"split": 2600, "names": {"Full Name In Captions": "Short"},
#  "calls": {"oct10": ["call.mp3", "call.srt"], "oct09": [...], ...}}
# The first call is split at "split" seconds: enrol before, test after.
CFG = json.loads((ROOT / "eval" / "speakers" / "calls.json").read_text())
CALLS = {k: tuple(pathlib.Path(x).expanduser() for x in v) for k, v in CFG["calls"].items()}
SPLIT = float(CFG.get("split", 2600.0))
NAMES = CFG["names"]


def clean_cues(cues, min_s):
    """Cues long enough to embed, with no other person's cue within 1 s, and
    trimmed 0.3 s each side (caption boundaries are loose)."""
    out = []
    for i, (s, e, n) in enumerate(cues):
        if n not in NAMES or e - s < min_s + 0.6:
            continue
        if any(m != n and min(e + 1, e2) - max(s - 1, s2) > 0 for s2, e2, m in cues):
            continue
        out.append((s + 0.3, min(e - 0.3, s + 15), NAMES[n]))
    return out


def extractor(model):
    return sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(ROOT / "models" / "diar" / model), num_threads=8))


def embed(ex, audio, s, e):
    st = ex.create_stream()
    st.accept_waveform(16000, audio[int(s * 16000):int(e * 16000)])
    st.input_finished()
    v = np.array(ex.compute(st))
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    models = sys.argv[1:] or sorted(p.name for p in (ROOT / "models" / "diar").glob("*.onnx")
                                    if p.name != "silero_vad.onnx")
    audio = {k: T.load_audio(a) for k, (a, _) in CALLS.items()}
    cues = {k: srt.cues(s) for k, (_, s) in CALLS.items()}
    first, *rest = CALLS
    enrol = [c for c in clean_cues(cues[first], 3.0) if c[1] < SPLIT]
    tests = {f"{first}@late": [c for c in clean_cues(cues[first], 2.0) if c[0] >= SPLIT],
             **{k: clean_cues(cues[k], 2.0) for k in rest}}
    per = {}
    for s, e, n in enrol:
        per.setdefault(n, []).append((s, e))
    print("enrol:", {n: f"{len(v)} cues/{sum(e - s for s, e in v):.0f}s" for n, v in per.items()})
    print("test: ", {k: len(v) for k, v in tests.items()})
    for model in models:
        ex = extractor(model); t = time.time()
        # A voice print = mean of up to 25 clean segments (~2 min) per person.
        prints = {}
        for n, segs in per.items():
            segs = sorted(segs, key=lambda x: x[0] - x[1])[:25]
            v = np.mean([embed(ex, audio[first], s, e) for s, e in segs], axis=0)
            prints[n] = v / np.linalg.norm(v)
        names = list(prints)
        P = np.stack([prints[n] for n in names])
        row = []
        for k, cs in tests.items():
            ok = {}
            for s, e, n in cs:
                pred = names[int(np.argmax(P @ embed(ex, audio[k.split("@")[0]], s, e)))]
                ok.setdefault(n, []).append(pred == n)
            # balanced accuracy: Asha talks 7x more and would hide errors on others
            row.append(f"{k} " + " ".join(f"{n[:3]} {np.mean(v)*100:3.0f}%" for n, v in sorted(ok.items())))
        print(f"{model[:52]:52} {time.time() - t:5.0f}s | " + " | ".join(row), flush=True)
