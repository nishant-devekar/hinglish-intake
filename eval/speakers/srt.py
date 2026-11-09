"""Google Meet caption files (.srt with "Name: text" cues) as speaker ground truth."""
import re

def cues(path):
    t = open(path, encoding="utf-8-sig").read()
    ts = lambda h, ms: sum(int(x) * m for x, m in zip(h.split(":"), (3600, 60, 1))) + int(ms) / 1000
    return [(ts(a, am), ts(b, bm), n.strip())
            for a, am, b, bm, n in re.findall(
                r"(\d\d:\d\d:\d\d),(\d+) --> (\d\d:\d\d:\d\d),(\d+)\s*\n([^:\n]+): ", t)]
