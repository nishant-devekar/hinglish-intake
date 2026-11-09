"""Process one job: every file in <job>/input -> <job>/result.

    .venv/bin/python bin/process.py <job_dir>

A job is a folder with job.json (title, options, status) and input/ holding
any mix of audio, video and transcripts. Result:

    result/README.md          the job (who, when, options), every file's media details,
                              speakers and languages, how notes were written, the
                              processing log, and how to cite (for Claude Code)
    result/job.json           the job record with its full log (written by bin/worker.py)
    result/notes.md           summary, decisions, action items, Trello cards
    result/bundle.md          README + notes + all transcripts/timelines in one file (for claude.ai)
    result/<file>/transcript.txt|.srt|.json
    result/<file>/meta.json   media details (ffprobe), speaker stats, language mix, processing log
    result/<file>/audio.m4a   audio/video: the recording's sound (mono AAC), to check a quote by ear
    result/<file>/frames/     videos: one screenshot per scene, index.txt, scenes.json, ocr.txt
    result/<file>/timeline.md videos: speaker breakdown, then scene by scene: screenshot + on-screen text + what was said
    result/<file>/timeline.json videos: the same join, for code
    result/<file>/evidence.pdf videos: the timeline with screenshots embedded (for claude.ai)

Meet caption files (.srt/.vtt with "Name: text" cues) uploaded next to a
recording name the speakers the voice prints don't know; on their own they
become the transcript.
"""
import json, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))
import transcribe, frames, notes, evidence

AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".amr", ".mpeg", ".mpga", ".wma"}
VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp"}
TIMED = {".srt", ".vtt"}
TEXT = {".txt", ".md", ".docx", ".doc", ".rtf"}


def kind(path):
    ext = path.suffix.lower()
    if ext in AUDIO | VIDEO:
        return "video" if frames.has_video(path) else "audio"
    if ext in TIMED:
        return "captions"
    if ext in TEXT:
        return "text"
    return None


def mmss(t):
    return transcribe.mmss(t)


def parse_timed(path):
    """SRT/VTT cues -> segments; "Name: text" cues carry the speaker's first name."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    ts = lambda h, m, s, ms: int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    segs = []
    for m in re.finditer(r"(?:(\d+):)?(\d\d):(\d\d)[,.](\d+)\s*-->\s*(?:(\d+):)?(\d\d):(\d\d)[,.](\d+)[^\n]*\n"
                         r"((?:.+\n?)+)", raw):
        text = " ".join(l.strip() for l in m.group(9).splitlines() if l.strip())
        text = re.sub(r"<[^>]+>", "", text)
        who = None
        sp = re.match(r"^([A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,3}):\s+(.*)$", text)
        if sp:
            who, text = sp.group(1).split()[0], sp.group(2)
        segs.append({"start": ts(*m.group(1, 2, 3, 4)), "end": ts(*m.group(5, 6, 7, 8)),
                     "speaker": who, "text": text.strip()})
    return segs


def read_text(path):
    if path.suffix.lower() in {".docx", ".doc", ".rtf"}:     # macOS built-in converter
        return subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)],
                              capture_output=True, text=True).stdout
    return path.read_text(encoding="utf-8-sig", errors="replace")


def names_from_captions(segs, cues, min_share=0.6, min_s=10.0):
    """Name speaker groups still unnamed after voice matching (S1, S2, S?)
    from overlapping caption cues; named lines are left alone. Captions are
    not trusted line by line: Meet credited a 4 s reply to the person who was
    talking around it, where diarization and the voice print both heard a
    second speaker. Meet captions line up with recordings at 0 s (3 calls)."""
    share = {}
    for s in segs:
        k = s["speaker"]
        if not (isinstance(k, str) and re.fullmatch(r"S(\d+|\?)", k)):
            continue
        for c in cues:
            o = min(s["end"], c["end"]) - max(s["start"], c["start"])
            if o > 0 and c["speaker"]:
                share.setdefault(k, {}).setdefault(c["speaker"], 0.0)
                share[k][c["speaker"]] += o
    found = {}
    for k, d in share.items():
        name, t = max(d.items(), key=lambda x: x[1])
        if t >= min_s and t / sum(d.values()) >= min_share:
            found[k] = name
    for s in segs:
        s["speaker"] = found.get(s["speaker"], s["speaker"])
    return found


def media_info(path):
    """What ffprobe knows about a recording: container, size, length, recording
    date, and per stream codec, resolution, frame rate or sample rate."""
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {}
    f = d.get("format", {})
    info = {"bytes": int(f.get("size", 0) or path.stat().st_size), "container": f.get("format_name"),
            "duration_s": round(float(f.get("duration", 0) or 0), 2),
            "bitrate_kbps": round(int(f.get("bit_rate", 0) or 0) / 1000) or None,
            "recorded_at": (f.get("tags") or {}).get("creation_time")}
    for st in d.get("streams", []):
        if st.get("codec_type") == "video" and not (st.get("disposition") or {}).get("attached_pic"):
            num, _, den = (st.get("avg_frame_rate") or "0/1").partition("/")
            info["video"] = {"codec": st.get("codec_name"), "width": st.get("width"), "height": st.get("height"),
                             "fps": round(int(num) / int(den), 2) if den and int(den) else None}
        elif st.get("codec_type") == "audio" and "audio" not in info:
            info["audio"] = {"codec": st.get("codec_name"), "sample_rate": int(st.get("sample_rate", 0) or 0),
                             "channels": st.get("channels")}
    return {k: v for k, v in info.items() if v is not None}


def language_mix(segs):
    """Share of words by script. Tara writes English words in Latin script and
    Hindi in Devanagari, so for Hinglish this is the mix inside lines (whose
    language tag just says "hi"); Marathi lines are all Devanagari, English
    loanwords included. The per-line language split is transcribe.languages()."""
    n = {"English (Latin script)": 0, "Hindi/Marathi (Devanagari)": 0}
    for s in segs:
        for w in s["text"].split():
            if re.search(r"[\u0900-\u097F]", w):
                n["Hindi/Marathi (Devanagari)"] += 1
            elif re.search(r"[A-Za-z]", w):
                n["English (Latin script)"] += 1
    total = sum(n.values())
    return {k: round(v / total, 3) for k, v in sorted(n.items(), key=lambda x: -x[1]) if v} if total else {}


def export_audio(src, out):
    """A small listenable copy of the sound (~15 MB/hour), so any quote can be checked by ear."""
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1",
                    "-c:a", "aac", "-b:a", "32k", str(out / "audio.m4a")])


def safe(name):
    return re.sub(r"[^\w.-]+", "_", name).strip("_")[:80] or "file"


def process(job_dir, log=print, keep_notes=False):
    """keep_notes: reuse result/notes.md if Claude already wrote it (re-runs)."""
    job_dir = pathlib.Path(job_dir)
    lines = []                  # this run's log, for README and each file's meta.json
    log = lambda m, _log=log: (lines.append(m), _log(m))
    job = json.loads((job_dir / "job.json").read_text())
    opt = job.get("options", {})
    res = job_dir / "result"; res.mkdir(exist_ok=True)
    inputs = sorted(p for p in (job_dir / "input").iterdir() if p.is_file() and not p.name.startswith("."))
    kinds = {p: kind(p) for p in inputs}
    media = [p for p in inputs if kinds[p] in ("audio", "video")]
    captions = {p: parse_timed(p) for p in inputs if kinds[p] == "captions"}
    skipped = [p.name for p in inputs if kinds[p] is None]
    if skipped:
        log(f"skipping unsupported files: {', '.join(skipped)}")

    # Named captions next to a recording name its unknown speakers; pair by closest length.
    paired = {}
    for c, cues in captions.items():
        if media and cues and sum(bool(x["speaker"]) for x in cues) >= len(cues) / 2:
            end = max(x["end"] for x in cues)
            m = min(media, key=lambda p: abs(frames.duration(p) - end))
            if abs(frames.duration(m) - end) < max(120, 0.2 * end):
                paired.setdefault(m, []).extend(cues); captions[c] = None
                log(f"{c.name}: captions matched to {m.name}, used to name its speakers")

    files = []
    for p in media:
        out = res / safe(p.stem)
        k = kinds[p]
        log(f"{p.name}: {k}, {mmss(frames.duration(p))}")
        if k == "video":
            frames.extract(p, out / "frames", ocr=opt.get("ocr", True), log=lambda m, n=p.name: log(f"{n}: {m}"))
        segs, meta = transcribe.run(p, out, opt.get("engine", "auto"), int(opt.get("speakers", -1)),
                                    log=lambda m, n=p.name: log(f"{n}: {m}"))
        if p in paired:
            found = names_from_captions(segs, paired[p])
            if found:
                log(f"{p.name}: named from captions: {', '.join(f'{k}->{v}' for k, v in found.items())}")
                meta["speakers"] = transcribe.write(out, segs, meta)
        if k == "video":
            evidence.build(out, p.name, f"{job.get('title') or job['id']} — {p.name}",
                           log=lambda m, n=p.name: log(f"{n}: {m}"))
        export_audio(p, out)
        meta.update(source_info=media_info(p), speaker_stats=evidence.speaker_stats(segs),
                    language_mix=language_mix(segs))
        if k == "video" and (out / "frames" / "scenes.json").exists():
            meta["scenes"] = len(json.loads((out / "frames" / "scenes.json").read_text()))
        files.append((p, out, k, meta))

    for p, cues in captions.items():
        if cues is None:
            continue
        out = res / safe(p.stem)
        segs = [dict(s, lang=None) for s in cues]
        meta = {"source": p.name, "engine": "captions", "duration_s": max((s["end"] for s in segs), default=0)}
        meta["speakers"] = transcribe.write(out, segs, meta)
        meta["words"] = sum(len(s["text"].split()) for s in segs)
        meta["speaker_stats"] = evidence.speaker_stats(segs)
        files.append((p, out, "captions", meta))

    for p in inputs:
        if kinds[p] != "text":
            continue
        out = res / safe(p.stem); out.mkdir(exist_ok=True)
        text = read_text(p)
        (out / "transcript.txt").write_text(text)
        files.append((p, out, "text", {"words": len(text.split()), "duration_s": 0}))

    for p, out, k, meta in files:
        (out / "meta.json").write_text(json.dumps({
            "file": p.name, "kind": k, "duration": mmss(meta.get("duration_s", 0)) if meta.get("duration_s") else "",
            "duration_s": meta.get("duration_s"), "words": meta.get("words"), "speakers": meta.get("speakers"),
            "engine": meta.get("engine"), "processing_s": meta.get("elapsed_s"),
            "scenes": meta.get("scenes"), "source": meta.get("source_info") or {"bytes": p.stat().st_size},
            "speaker_stats": meta.get("speaker_stats"), "languages": meta.get("languages"),
            "language_mix": meta.get("language_mix"),
            "log": [l.split(": ", 1)[1] for l in lines if l.startswith(f"{p.name}: ") and "asr:" not in l]},
            ensure_ascii=False, indent=1))

    if not files:
        raise RuntimeError("no audio, video or transcript files in this job")
    kept = res / "notes.md"
    if keep_notes and kept.exists() and not notes.needs_retry(kept.read_text()):
        log("notes: kept from the earlier run")
        notes_by = "claude"     # only pending notes and local drafts need a retry
    else:
        notes_by = notes.generate(job_dir, log) if opt.get("notes", True) else None
    write_index(job, job_dir, files, notes_by, lines)
    return [f[0].name for f in files]


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def media_line(info):
    """One line of media details: size, container, video and audio streams, recording date."""
    bits = [human(info["bytes"])] if info.get("bytes") else []
    if info.get("container"):
        bits.append(info["container"].split(",")[0])
    if v := info.get("video"):
        bits.append(f"video {v.get('codec')} {v.get('width')}×{v.get('height')}" + (f" @ {v['fps']:g} fps" if v.get("fps") else ""))
    if a := info.get("audio"):
        bits.append(f"audio {a.get('codec')} {a.get('sample_rate', 0) / 1000:g} kHz, {a.get('channels')} ch")
    if info.get("recorded_at"):
        bits.append(f"recorded {info['recorded_at']}")
    return " · ".join(bits)


NOTES_BY = {"claude": "Claude (Claude Code on the processing Mac)",
            "local": "the local fallback model (Gemma 4 12B on the processing Mac) because Claude was "
                     "unavailable; a draft, replaced by Claude's notes automatically once Claude works",
            None: "not written (see the notes section)"}


def write_index(job, job_dir, files, notes_by=None, lines=()):
    res = job_dir / "result"
    names = sorted({s["speaker"] for _, out, _, _ in files if (out / "transcript.json").exists()
                    for s in json.loads((out / "transcript.json").read_text())["segments"]
                    if isinstance(s.get("speaker"), str) and not re.fullmatch(r"S(\d+|\?)", s["speaker"])})
    opt = job.get("options", {})
    head = (f"# {job.get('title') or job['id']}\n\n"
            f"| | |\n|---|---|\n"
            f"| job | `{job['id']}` |\n"
            f"| submitted | {job.get('submitted_at', '')} by {job.get('submitted_by', '?')} |\n"
            f"| processed | {job.get('started_at', '')} on the processing Mac |\n"
            f"| uploaded files | {', '.join(job.get('files', [])) or '—'} |\n"
            f"| options | speech engine `{opt.get('engine', 'auto')}`, speakers "
            f"{'auto' if int(opt.get('speakers', -1)) <= 0 else opt['speakers']}, on-screen text "
            f"{'on' if opt.get('ocr', True) else 'off'}, notes {'on' if opt.get('notes', True) else 'off'} |\n"
            f"| known speakers | {', '.join(names) or '—'} |\n"
            f"| notes written by | {NOTES_BY.get(notes_by, notes_by) if opt.get('notes', True) else 'turned off for this job'} |\n\n")
    rows = "\n".join(f"| `{out.name}/` | {p.name} | {k} | {m.get('duration_s') and mmss(m['duration_s']) or ''} "
                     f"| {m.get('words', '')} | {m.get('speakers') or ''} | {m.get('scenes') or ''} |"
                     for p, out, k, m in files)
    per_file = []
    for p, out, k, m in files:
        part = [f"### {p.name}\n"]
        if m.get("source_info"):
            part.append(f"{media_line(m['source_info'])}\n")
        if m.get("engine"):
            part.append(f"Transcribed by `{m['engine']}`" + (f" in {m['elapsed_s']:.0f} s" if m.get("elapsed_s") else "")
                        + (f"; {m['scenes']} scenes captured (one screenshot each)" if m.get("scenes") else "") + ".\n")
        if m.get("languages"):
            part.append("Speech by language: " + ", ".join(f"{l} {100 * v:.0f}%" for l, v in m["languages"].items())
                        + " (each stretch transcribed by the model for its language)\n")
        if m.get("language_mix"):
            part.append("Words by script: " + ", ".join(f"{l} {100 * v:.0f}%" for l, v in m["language_mix"].items()) + "\n")
        if m.get("speaker_stats"):
            part.append("| speaker | talk time | share | words | lines | heard |\n|---|---|---|---|---|---|\n" + "\n".join(
                f"| {d['speaker']} | {mmss(d['talk_s'])} | {100 * d['share']:.0f}% | {d['words']} | {d['lines']} "
                f"| {mmss(d['first_s'])}–{mmss(d['last_s'])} |" for d in m["speaker_stats"]) + "\n")
        per_file.append("\n".join(part))
    log = "\n".join(l for l in lines if "asr:" not in l)
    (res / "README.md").write_text(
        head + "## Files\n\n| folder | original file | kind | length | words | speakers | scenes |\n"
        "|---|---|---|---|---|---|---|\n" + rows + "\n\n" + "\n".join(per_file) + "\n"
        "## What's here\n\n"
        "- `notes.md` — summary, decisions, action items (with owners), open questions, Trello cards\n"
        "- `<folder>/transcript.txt` — `[mm:ss] Speaker: text`; `.srt`/`.json` are the same lines with exact "
        "times (`.json` also has each line's language)\n"
        "- `<folder>/timeline.md` — videos: who spoke how long, then scene by scene, the screenshot, all of its "
        "on-screen text and what was said while it showed; open the `frames/frame-NNNN.jpg` it links to see "
        "the screen. `timeline.json` is the same as data\n"
        "- `<folder>/evidence.pdf` — videos: the same timeline with screenshots embedded (attach in claude.ai)\n"
        "- `<folder>/frames/` — videos: one screenshot per scene; `index.txt` maps each to its time span, "
        "`ocr.txt` is the raw on-screen text per screenshot\n"
        "- `<folder>/audio.m4a` — the recording's sound, to check a quote by ear\n"
        "- `<folder>/meta.json` — media details, speaker stats, language mix and that file's processing log\n"
        "- `job.json` — the job record: who uploaded what, when, options, and the full log\n"
        "- `bundle.md` — this page + notes + all transcripts/timelines in one file, for a claude.ai chat or "
        "Project (add the evidence.pdf files for the pictures)\n\n"
        "Cite as `[<original file> @ mm:ss]`. English and Hindi/Hinglish are transcribed by Tara (English "
        "words in Latin script, Hindi in Devanagari); Marathi by IndicWhisper Marathi (all Devanagari, "
        "English loanwords too). `transcript.json` gives each line's language. `S1`/`S2` are speakers whose voices aren't enrolled; `S?` lines "
        "didn't match their group's voice and were left unnamed.\n\n"
        + (f"## Processing log\n\n```\n{log}\n```\n" if log else ""))
    parts = [(res / "README.md").read_text()]
    if (res / "notes.md").exists():
        parts.append((res / "notes.md").read_text())
    for p, out, k, m in files:
        tl = out / "timeline.md"
        if tl.exists():         # videos: speech already grouped under each screenshot, with on-screen text
            parts.append(f"## Video: {p.name} — screenshots are in evidence.pdf\n\n" + tl.read_text())
        else:
            parts.append(f"## Transcript: {p.name}\n\n{(out / 'transcript.txt').read_text()}")
    (res / "bundle.md").write_text("\n\n---\n\n".join(parts) + "\n")


def main():
    job_dir = pathlib.Path(sys.argv[1])
    t = time.time()
    done = process(job_dir, lambda m: print(f"    {m}", file=sys.stderr, flush=True))
    print(f"{len(done)} files in {time.time() - t:.0f}s -> {job_dir / 'result'}")


if __name__ == "__main__":
    main()
