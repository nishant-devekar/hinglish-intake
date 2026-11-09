"""Join a video's screenshots, on-screen text and speech into one timeline.

    timeline.md    audio breakdown (who spoke, how long), then scene by scene:
                   screenshot link, on-screen text, what was said while it was
                   on screen (Claude Code opens the images itself)
    timeline.json  the same join for code: scenes with frame, span, OCR lines,
                   the segments said during them
    evidence.pdf   the same with the screenshots embedded, for claude.ai, which
                   reads PDF pages as images. Rendered by Chrome so Devanagari
                   shapes correctly; capped at PDF_MAX_SHOTS screenshots (~2 per
                   page) so claude.ai keeps every page visual.

A line of speech is listed under the scene showing when it started; ASR
segments run up to ~30 s, so scenes it carries on into say so ("still
speaking") instead of looking silent. Text that is on screen in most scenes
(browser tabs, bookmarks, an app's menu bar) is listed once at the top and
left out of each scene. timeline.md keeps every OCR line (it is what
claude.ai reads, via bundle.md); the PDF trims each scene's text to
OCR_CHARS so a scene still fits beside its screenshot.

Input: <out>/frames/scenes.json + ocr.txt (bin/frames.py), <out>/transcript.json.
"""
import html, json, math, pathlib, re, subprocess, sys, tempfile
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from frames import read_ocr

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDF_MAX_SHOTS = 160
OCR_CHARS = 500
CHROME_SHARE = 0.6      # OCR lines in at least this share of scenes are window chrome
CARRY_S = 2             # a line still running this long into a scene is "still speaking" there


def mmss(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def chrome_key(l):
    """OCR reads the same tab title a little differently frame to frame
    ("ATK | C nish", "ATK | Q) nish"); compare lines by their longer words."""
    return " ".join(re.findall(r"[^\W_]{3,}", l.lower())) or l


def scenes_with_speech(out):
    """(scenes, segments, chrome lines); each scene gets ocr, said (lines that
    started in it) and carried (lines from earlier scenes still being spoken)."""
    out = pathlib.Path(out)
    scenes = json.loads((out / "frames" / "scenes.json").read_text())
    ocr = read_ocr(out / "frames")
    segs = json.loads((out / "transcript.json").read_text())["segments"] if (out / "transcript.json").exists() else []
    seen = {}
    for s in scenes:
        s["ocr"] = [l for l in ocr.get(s["frame"], []) if re.search(r"\w", l)]
        for k in {chrome_key(l) for l in s["ocr"]}:
            seen[k] = seen.get(k, 0) + 1
    chrome = {k for k, n in seen.items() if len(scenes) >= 5 and n >= CHROME_SHARE * len(scenes)}
    shown = {}                  # one spelling of each chrome line, for the header
    for s in scenes:
        for l in s["ocr"]:
            if chrome_key(l) in chrome:
                shown.setdefault(chrome_key(l), l)
        s["ocr"] = [l for l in s["ocr"] if chrome_key(l) not in chrome]
        s["said"], s["carried"] = [], []
    for seg in segs:            # a line belongs to the scene on screen when it started
        i = next((k for k, s in enumerate(scenes) if s["start"] <= seg["start"] < s["end"]), len(scenes) - 1)
        scenes[i]["said"].append(seg)
        for s in scenes[i + 1:]:
            if seg["end"] - s["start"] < CARRY_S:
                break
            s["carried"].append(seg)
    return scenes, segs, list(shown.values())


def ocr_text(lines):
    t = " · ".join(lines)
    return t if len(t) <= OCR_CHARS else t[:OCR_CHARS].rsplit(" ", 1)[0] + " …"


def line(seg):
    who = f"{seg['speaker']}: " if seg.get("speaker") else ""
    return f"[{mmss(seg['start'])}] {who}{seg['text']}"


def carried(seg):
    who = f"{seg['speaker']} " if seg.get("speaker") else ""
    return f"({who}still speaking, from [{mmss(seg['start'])}])"


def speaker_stats(segs):
    """Per speaker, most talk first: talk time, share, words, lines, first and last time heard."""
    by = {}
    for s in segs:
        d = by.setdefault(s.get("speaker") or "unlabelled",
                          {"talk_s": 0.0, "words": 0, "lines": 0, "first_s": s["start"], "last_s": s["end"]})
        d["talk_s"] += s["end"] - s["start"]; d["words"] += len(s["text"].split()); d["lines"] += 1
        d["last_s"] = max(d["last_s"], s["end"])
    total = sum(d["talk_s"] for d in by.values()) or 1
    return [dict(speaker=k, share=round(d["talk_s"] / total, 3), **{k2: round(v, 1) for k2, v in d.items()})
            for k, d in sorted(by.items(), key=lambda x: -x[1]["talk_s"])]


def audio_breakdown(segs):
    """speaker_stats as a Markdown table."""
    if not segs:
        return ["_(no speech)_"]
    rows = [f"| {d['speaker']} | {mmss(d['talk_s'])} | {100 * d['share']:.0f}% | {d['words']} | {d['lines']} "
            f"| {mmss(d['first_s'])}–{mmss(d['last_s'])} |" for d in speaker_stats(segs)]
    return ["| speaker | talk time | share | words | lines | heard |", "|---|---|---|---|---|---|"] + rows


def write_timeline(out, name, scenes, segs, chrome):
    parts = [f"# Timeline: {name}\n",
             f"{len(scenes)} scenes. Each scene: the screenshot (its sharpest second), the text on "
             f"screen, and what was said while it was showing. Cite as `[{name} @ mm:ss]`; "
             f"mention the frame when a point depends on what's on screen.\n",
             "## Audio\n", "\n".join(audio_breakdown(segs)) + "\n"]
    if chrome:
        parts.append("**On screen in most scenes** (mostly window/app chrome; left out of the scenes below): "
                     + " · ".join(chrome) + "\n")
    prev = None
    for n, s in enumerate(scenes, 1):
        parts.append(f"## Scene {n} · {mmss(s['start'])}–{mmss(s['end'])} · `frames/{s['frame']}` (at {mmss(s['t'])})\n")
        parts.append(f"![{name} @ {mmss(s['t'])}](frames/{s['frame']})\n")
        if s["ocr"]:
            # Only identical text is collapsed: handwriting OCR differs every frame, and
            # fuzzy matching couldn't tell "same board, misread" from new content.
            parts.append("**On screen:** " + ("(same as the previous scene)" if s["ocr"] == prev
                                               else " · ".join(s["ocr"])) + "\n")
        prev = s["ocr"]
        said = [f"- {carried(x)}" for x in s["carried"]] + [f"- {line(x)}" for x in s["said"]]
        parts.append("**Said:**\n" + "\n".join(said) + "\n" if said else "_(no speech)_\n")
    (pathlib.Path(out) / "timeline.md").write_text("\n".join(parts))


def write_json(out, name, scenes, chrome):
    (pathlib.Path(out) / "timeline.json").write_text(json.dumps({
        "file": name, "chrome_ocr": chrome,
        "scenes": [{"scene": n, "frame": f"frames/{s['frame']}", "t": s["t"], "start": s["start"], "end": s["end"],
                    "ocr": s["ocr"], "said": s["said"],
                    "still_speaking": [{"start": x["start"], "speaker": x.get("speaker")} for x in s["carried"]]}
                   for n, s in enumerate(scenes, 1)]}, ensure_ascii=False, indent=1))


CSS = """
@page { size: A4; margin: 12mm }
body { font: 10pt -apple-system, "Helvetica Neue", "Kohinoor Devanagari", sans-serif; color: #111 }
h1 { font-size: 16pt; margin: 0 0 4pt } .sub { color: #555; margin-bottom: 10pt }
table { border-collapse: collapse; margin: 0 0 8pt; font-size: 9pt } td, th { border: 1px solid #ccc; padding: 2pt 6pt; text-align: left }
.scene { page-break-inside: avoid; border-top: 1px solid #ccc; padding: 6pt 0 8pt }
.hd { font-weight: 600; margin-bottom: 4pt } .hd span { color: #666; font-weight: 400 }
img { max-width: 100%; max-height: 88mm; display: block; margin: 0 auto 4pt; border: 1px solid #ddd }
.ocr { color: #444; font-size: 8.5pt; margin: 2pt 0 } .said p { margin: 1pt 0 } .none, .cont { color: #888 }
"""


def write_pdf(out, name, title, scenes, segs, chrome):
    out = pathlib.Path(out)
    if not pathlib.Path(CHROME).exists():
        return None
    # Group scenes so the PDF stays within PDF_MAX_SHOTS screenshots; all speech is kept.
    k = max(1, math.ceil(len(scenes) / PDF_MAX_SHOTS))
    groups = [scenes[i:i + k] for i in range(0, len(scenes), k)]
    rows = [r for r in audio_breakdown(segs) if not r.startswith("|---")]
    table = "<table>" + "".join("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in r.strip("|").split("|"))
                                + "</tr>" for r in rows) + "</table>" if segs else ""
    with tempfile.TemporaryDirectory(prefix="evidence-") as tmp:
        tmp = pathlib.Path(tmp)
        blocks, prev = [], None
        for n, g in enumerate(groups, 1):
            shot = g[0]
            small = tmp / shot["frame"]
            im = Image.open(out / "frames" / shot["frame"]).convert("RGB")
            im.thumbnail((1000, 1000)); im.save(small, quality=72)
            ocr = shot["ocr"]
            said = [x for s in g for x in s["said"]]
            cont = "".join(f'<p class="cont">{html.escape(carried(x))}</p>' for x in g[0]["carried"])
            blocks.append(
                f'<div class="scene"><div class="hd">Scene {n} · {mmss(g[0]["start"])}–{mmss(g[-1]["end"])} '
                f'<span>· frames/{shot["frame"]} at {mmss(shot["t"])}</span></div>'
                f'<img src="{shot["frame"]}">'
                + (f'<div class="ocr"><b>On screen:</b> {html.escape("(same as the previous scene)" if ocr == prev else ocr_text(ocr))}</div>' if ocr else "")
                + ('<div class="said">' + cont + "".join(f"<p>{html.escape(line(x))}</p>" for x in said) + "</div>"
                   if said or cont else '<div class="none">(no speech)</div>') + "</div>")
            prev = ocr
        page = (f'<!doctype html><meta charset="utf-8"><style>{CSS}</style>'
                f'<h1>{html.escape(title)}</h1><div class="sub">{html.escape(name)} · {len(scenes)} scenes'
                + (f", shown {k} per screenshot" if k > 1 else "") +
                " · screenshot, on-screen text, and what was said while it was showing</div>" + table
                + (f'<div class="ocr"><b>On screen in most scenes</b> (left out below): {html.escape(ocr_text(chrome))}</div>'
                   if chrome else "") + "".join(blocks))
        (tmp / "evidence.html").write_text(page)
        pdf = out / "evidence.pdf"
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={pdf}", (tmp / "evidence.html").as_uri()],
                           capture_output=True, text=True, timeout=600)
    return pdf if pdf.exists() and pdf.stat().st_size else None


def build(out, name, title=None, log=print):
    if not (pathlib.Path(out) / "frames" / "scenes.json").exists():
        return
    scenes, segs, chrome = scenes_with_speech(out)
    if not scenes:
        return
    write_timeline(out, name, scenes, segs, chrome)
    write_json(out, name, scenes, chrome)
    pdf = write_pdf(out, name, title or name, scenes, segs, chrome)
    log(f"timeline: {len(scenes)} scenes" + (f", evidence.pdf {pdf.stat().st_size / 1e6:.1f} MB" if pdf else
                                            ", no PDF (Google Chrome not found)"))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(sys.argv[1]).name)
