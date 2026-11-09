"""Known voices, so transcripts say "Asha:" instead of "S1:".

    voices.py enroll-srt  AUDIO MEET.srt            # every named person in a Google Meet caption file
    voices.py enroll      NAME AUDIO 12:30-12:55 [41:02-41:20 ...]
    voices.py enroll-from OUT_DIR S2 NAME           # a speaker from an earlier transcribe.py output
    voices.py list | remove NAME

Prints live in voices/prints.json: per person, the sum of unit-length
NeMo TitaNet-large embeddings of their clean segments, so enrolling more
audio later just adds to it. These are voice biometrics of real people;
they stay on this machine and are only used to label our own calls.

TitaNet won the comparison in eval/speakers/ (100 % on held-out Oct 9 and
Oct 10 segments, 96-99 % on Oct 5). A cluster is named only if it scores
>= MIN_SCORE against a print and beats the runner-up by MIN_MARGIN; voices
never enrolled scored at most 0.48 against the closest print.
"""
import json, pathlib, re, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "voices" / "prints.json"
MODEL = "nemo_en_titanet_large.onnx"
SR = 16000
MIN_SCORE, MIN_MARGIN = 0.55, 0.10
# A short cluster (one 4 s reply) scores lower against its own print, but an
# unknown voice resembles every print about equally (0.50 vs 0.48 on the
# Marathi call). So a lower score is accepted when the lead is wide.
LOW_SCORE, WIDE_MARGIN = 0.45, 0.20
# Per-segment re-check: move a segment to another enrolled person only on a
# clear lead over the cluster's own name.
SEG_MIN_S, SEG_MARGIN = 2.0, 0.15
# ...and drop a segment's name when its own voice is far from that person's
# print. On the three test calls, segments truly by the named person scored
# >= 0.48 except one (0.24); guests/others merged into the cluster scored
# 0.0-0.33. Such segments become "S?" (unsure): no name beats a wrong one.
SEG_UNSURE = 0.35


def confident(best, second):
    return (best >= MIN_SCORE and best - second >= MIN_MARGIN) or \
           (best >= LOW_SCORE and best - second >= WIDE_MARGIN)
_ex = None


def extractor():
    global _ex
    if _ex is None:
        import sherpa_onnx
        _ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(ROOT / "models" / "diar" / MODEL), num_threads=8))
    return _ex


def embed(audio, s, e):
    st = extractor().create_stream()
    st.accept_waveform(SR, audio[int(s * SR):int(e * SR)])
    st.input_finished()
    v = np.array(extractor().compute(st))
    return v / np.linalg.norm(v)


def load():
    return json.loads(STORE.read_text()) if STORE.exists() else {"model": MODEL, "people": {}}


def prints():
    """{name: unit vector}; empty if nobody is enrolled."""
    db = load()
    if db["model"] != MODEL:
        raise SystemExit(f"voices/prints.json was made with {db['model']}; re-enrol with {MODEL}")
    return {n: np.array(p["sum"]) / np.linalg.norm(p["sum"]) for n, p in db["people"].items()}


def add(name, audio, spans, source):
    """Embed each span (cropped to <= 15 s; longer adds little) and add to name's print."""
    spans = [(s, min(e, s + 15)) for s, e in spans if e - s >= 1.5]
    if not spans:
        print(f"  {name}: no span >= 1.5 s, nothing enrolled"); return
    vs = [embed(audio, s, e) for s, e in spans]
    db = load()
    p = db["people"].setdefault(name, {"sum": [0.0] * len(vs[0]), "segments": 0, "seconds": 0.0, "sources": []})
    p["sum"] = (np.array(p["sum"]) + np.sum(vs, axis=0)).tolist()
    p["segments"] += len(vs)
    p["seconds"] = round(p["seconds"] + sum(e - s for s, e in spans), 1)
    p["sources"] = sorted(set(p["sources"]) | {source})
    STORE.parent.mkdir(exist_ok=True)
    STORE.write_text(json.dumps(db, indent=1))
    print(f"  {name}: +{len(vs)} segments, {p['seconds']:.0f}s total")


def name_speakers(audio, segs, log=print):
    """Map diarization labels in `segs` to enrolled names, in place. Each
    label is embedded from up to 90 s of its own longest segments."""
    known = prints()
    if not known:
        return {}
    names, P = list(known), np.stack(list(known.values()))
    by = {}
    for s in segs:
        if s["speaker"] is not None:
            by.setdefault(s["speaker"], []).append((s["start"], s["end"]))
    found = {}
    for spk, spans in by.items():
        spans = sorted(spans, key=lambda x: x[0] - x[1])
        take, total = [], 0.0
        for s, e in spans:
            if e - s >= 1.5 and total < 90:
                take.append((s, min(e, s + 15))); total += min(e, s + 15) - s
        if not take:
            continue
        v = np.mean([embed(audio, s, e) for s, e in take], axis=0)
        sc = P @ (v / np.linalg.norm(v))
        order = np.argsort(-sc)
        best, second = sc[order[0]], (sc[order[1]] if len(sc) > 1 else -1.0)
        ok = confident(best, second)
        log(f"voice {spk}: {names[order[0]]} {best:.2f}"
            + (f", next {names[order[1]]} {second:.2f}" if len(sc) > 1 else "")
            + ("" if ok else " -> unnamed"))
        if ok:
            found[spk] = names[order[0]]
    for s in segs:
        s["speaker"] = found.get(s["speaker"], s["speaker"])
    # Diarization sometimes puts two voices in one cluster (Marathi call: a
    # guest inside Ravi's), and the cluster name then covers both. Check
    # each segment long enough to embed against the prints on its own.
    moved = unsure = 0
    for s in segs:
        if s["end"] - s["start"] < SEG_MIN_S:
            continue
        sc = P @ embed(audio, s["start"], min(s["end"], s["start"] + 15))
        order = np.argsort(-sc)
        top, cur = names[order[0]], s["speaker"]
        own = sc[names.index(cur)] if cur in names else sc[order[1]] if len(sc) > 1 else -1.0
        if top != cur and sc[order[0]] >= MIN_SCORE and sc[order[0]] - own >= SEG_MARGIN:
            s["speaker"] = top; moved += 1
        elif cur in names and own < SEG_UNSURE:
            s["speaker"] = "S?"; unsure += 1
    if moved or unsure:
        log(f"voice: {moved} segments moved to the person they sound like, {unsure} marked S? (unsure)")
    return found


# --- CLI ---------------------------------------------------------------------

def _audio(path):
    sys.path.insert(0, str(ROOT / "bin"))
    from transcribe import load_audio
    return load_audio(path)


def _secs(t):
    return sum(float(x) * m for x, m in zip(reversed(t.split(":")), (1, 60, 3600)))


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else "list"
    if cmd == "list":
        for n, p in load()["people"].items():
            print(f"{n:12} {p['seconds']:6.0f}s  {p['segments']:3d} segments  from {', '.join(p['sources'])}")
    elif cmd == "remove":
        db = load(); db["people"].pop(a[1], None); STORE.write_text(json.dumps(db, indent=1))
    elif cmd == "enroll":
        name, path, spans = a[1], a[2], [tuple(map(_secs, x.split("-"))) for x in a[3:]]
        add(name, _audio(path), spans, pathlib.Path(path).name)
    elif cmd == "enroll-from":
        out, label, name = pathlib.Path(a[1]), a[2], a[3]
        segs = json.loads((out / "transcript.json").read_text())["segments"]
        spans = [(s["start"], s["end"]) for s in segs if s["speaker"] == label]
        add(name, _audio(out / "audio.wav"), sorted(spans, key=lambda x: x[0] - x[1])[:40], out.name)
    elif cmd == "enroll-srt":
        sys.path.insert(0, str(ROOT / "eval" / "speakers"))
        from srt import cues
        path, cs = a[1], cues(a[2])
        audio = _audio(path)
        for full in sorted({n for _, _, n in cs}):
            # Only cues with nobody else talking within 1 s: Meet's speaker
            # boundaries are loose, and one stray voice skews a print.
            spans = [(s + 0.3, e - 0.3) for s, e, n in cs if n == full and e - s >= 2.6
                     and not any(m != n and min(e + 1, e2) - max(s - 1, s2) > 0 for s2, e2, m in cs)]
            add(re.split(r"\s+", full)[0], audio, sorted(spans, key=lambda x: x[0] - x[1])[:40],
                pathlib.Path(a[2]).name)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
