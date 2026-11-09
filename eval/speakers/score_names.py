"""Second-by-second check of transcript speaker names against Meet captions."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from srt import cues

def score(transcript_json, srt_path):
    segs = json.loads(pathlib.Path(transcript_json).read_text())["segments"]
    ok, tot = {}, {}
    for cs, ce, full in cues(srt_path):
        who = full.split()[0]
        for s in segs:
            o = min(ce, s["end"]) - max(cs, s["start"])
            if o > 0:
                tot[who] = tot.get(who, 0) + o
                ok[who] = ok.get(who, 0) + o * (s["speaker"] == who)
    return {w: (ok[w] / tot[w], tot[w]) for w in tot}

if __name__ == "__main__":
    for w, (acc, t) in sorted(score(sys.argv[1], sys.argv[2]).items()):
        print(f"   {w:8} {acc*100:5.1f}% of {t:4.0f}s captioned speech labelled correctly")
