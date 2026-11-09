"""Score a transcript against the hand-corrected gold file.

    .venv/bin/python eval/gold/score_gold.py <transcript.json> [gold.txt]

Gold: eval/gold/oct09-gold.txt — lines "[mm:ss] Speaker: text" covering
Oct 9 call 02:00-12:00 (times are from the full call). Reports word and
character error rate on text normalised for spelling variants that don't
change meaning (nukta, chandrabindu vs anusvara, punctuation, case), and
how much of the gold speech carries the right speaker name.
"""
import json, re, sys, pathlib, unicodedata
import jiwer

HERE = pathlib.Path(__file__).parent
START, END = 120.0, 720.0


def norm(t):
    t = unicodedata.normalize("NFC", t.lower())
    t = t.replace("़", "").replace("ँ", "ं")      # nukta; chandrabindu -> anusvara
    t = re.sub(r"[\[\]().,?!।:;\"'“”‘’…-]+", " ", t)
    return " ".join(t.split())


def gold_lines(path):
    out = []
    for l in pathlib.Path(path).read_text().splitlines():
        m = re.match(r"\[(\d+):(\d\d)\]\s*([^:]+):\s*(.*)", l)
        if m:
            out.append((int(m[1]) * 60 + int(m[2]), m[3].strip(), m[4]))
    return out


def main():
    hyp_path = sys.argv[1]
    gold = gold_lines(sys.argv[2] if len(sys.argv) > 2 else HERE / "oct09-gold.txt")
    segs = [s for s in json.loads(pathlib.Path(hyp_path).read_text())["segments"]
            if START <= s["start"] < END]   # same rule the draft was cut with
    ref = norm(" ".join(t for _, _, t in gold if "[?]" not in t))
    hyp = norm(" ".join(s["text"] for s in segs))
    print(f"WER {jiwer.wer(ref, hyp)*100:.1f}%   CER {jiwer.cer(ref, hyp)*100:.1f}%   "
          f"({len(ref.split())} reference words)")
    # speaker: each gold line runs until the next one; compare by time overlap
    ok = tot = 0.0
    for i, (t0, who, _) in enumerate(gold):
        t1 = gold[i + 1][0] if i + 1 < len(gold) else END
        for s in segs:
            o = min(t1, s["end"]) - max(t0, s["start"])
            if o > 0:
                tot += o; ok += o * (s["speaker"] == who)
    print(f"speaker names right on {ok / max(tot, 1) * 100:.1f}% of gold speech time")


if __name__ == "__main__":
    main()
