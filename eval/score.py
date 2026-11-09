"""Reference-free scorecard for eval/results (there is no human transcript yet).

Measures what can be measured without one: speed, repetition loops,
hallucination on a near-silent clip, coverage (how much speech survived),
and whether English words come out in Latin script (readable Hinglish).
"""
import re, csv, collections, pathlib
R = pathlib.Path(__file__).resolve().parent / "results"
CLIP_S = 90.0
SPEECH = ["call1009_hinglish", "call1005_hinglish", "ptt_voicenote_hinglish", "video3_english", "whatsapp"]
HINGLISH = SPEECH[:3]

def words(t):
    t = re.sub(r"\[\s*[\d.]+s S\d+\]", " ", t)
    return re.findall(r"[\wऀ-ॿ']+", t)

def max_loop(w):
    best = 1
    for n in (1, 2, 3):
        run = 1
        for i in range(n, len(w) - n + 1, n):
            run = run + 1 if w[i:i + n] == w[i - n:i] else 1
            best = max(best, run if n == 1 else run)
    return best

time = {}
for row in csv.reader(open(R / "timings.tsv"), delimiter="\t"):
    time[(row[0], row[1])] = float(row[2])          # last run wins
models = sorted({m for _, m in time})
rows = []
for m in models:
    try:
        w = {c: words((R / f"{c}.{m}.txt").read_text(errors="replace")) for c in SPEECH + ["silence_trap"]}
    except FileNotFoundError:
        continue
    rtf = sum(CLIP_S for c in SPEECH) / sum(time.get((c, m), 0) for c in SPEECH)
    loops = max(max_loop(w[c]) for c in SPEECH)
    junk = len(w["silence_trap"])
    hw = [x for c in HINGLISH for x in w[c]]
    latin = sum(bool(re.fullmatch(r"[A-Za-z']+", x)) for x in hw) / max(1, len(hw))
    rows.append((m, rtf, loops, junk, {c: len(w[c]) for c in SPEECH}, latin))

maxw = {c: max(r[4][c] for r in rows) for c in SPEECH}
print(f"{'model':16} {'x realtime':>10} {'worst loop':>10} {'silence junk':>12} {'coverage':>9} {'latin% (hinglish)':>18}")
for m, rtf, loops, junk, wc, latin in sorted(rows, key=lambda r: -sum(r[4][c] / maxw[c] for c in SPEECH)):
    cov = sum(wc[c] / maxw[c] for c in SPEECH) / len(SPEECH)
    print(f"{m:16} {rtf:10.1f} {loops:10d} {junk:12d} {cov*100:8.0f}% {latin*100:17.0f}%")
