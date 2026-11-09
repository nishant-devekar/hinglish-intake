"""Meeting notes from processed transcripts, written by Claude Code on this Mac.

    .venv/bin/python bin/notes.py <job_dir>       # (re)generate result/notes.md

Uses the `claude` CLI headless (your Claude subscription, no API key).
Every point cites where it came from, e.g. [call.mp3 @ 12:40], so readers
can jump to the transcript line.

When Claude fails (usage limit, logged out, timeout), a local model writes
them instead: Gemma 4 12B, 4-bit MLX (~7 GB, Apache-2.0), in its own
process so the memory is returned as soon as it's done. Its notes start
with a DRAFT banner, and the worker swaps in Claude's notes once Claude
works again (bin/worker.py). Gemma 4 26B-A4B writes better notes but at
15 GB it pushes a 24 GB Mac into swap while people are using it.
"""
import json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCAL_MODEL = ROOT / "models" / "hf" / "gemma-4-12b-it-4bit"
LOCAL_NAME = "Gemma 4 12B, on this Mac"
LOCAL_MAX_TOKENS = 100_000     # prompt size the local model is trusted with (context is 128k)
PENDING = "# Notes pending"
DRAFT = "> **Draft by the local fallback model"

PROMPT = """You are writing meeting notes for the team from the
transcript(s) below. The speech is English, Hindi-English (Hinglish) and/or
Marathi; Hindi and Marathi are in Devanagari, and Marathi lines write English
loanwords in Devanagari too ("स्टॅंडर्ड ऑपरेटिंग प्रोसिजर"). Write the notes in clear
English, but keep product names, numbers and exact commitments as said.

Every bullet ends with a citation to where it was said, in the form
[<file> @ mm:ss] (use the timestamps and file names given). Videos come as a
timeline of scenes (screenshot file, on-screen text, what was said while it
showed); when a point depends on what was on screen, add the frame, e.g.
[Video 3.mp4 @ 07:17, frames/frame-0045.jpg]. On-screen text from
handwriting is noisy OCR: use it as a hint, not as a quote. Name people as
the transcript does; "S1"/"S2" are speakers whose names are unknown - keep
those labels, don't guess names. If something is unclear in the transcript,
say so rather than inventing detail.

Output Markdown with exactly these sections:

# <short title for the meeting>
**Participants:** ...  **Length:** ...

## Summary
3-6 sentences: what was discussed and what changed.

## Decisions
- <decision> — <who decided> [cite]

## Action items
| Owner | Action | Due | Source |
|---|---|---|---|
(owner as named; due as said, or "—"; source = citation)

## Open questions
- <question still unresolved> [cite]

## Topics
- mm:ss — <topic> (one line each, in order, for navigation)

## Trello cards
```json
[{"title": "...", "owner": "...", "due": "...", "description": "... [cite]"}]
```
(one card per action item; [] if none)
"""


OCR_CHARS = 500     # per scene, in the notes prompt


def clip(line):
    """timeline.md keeps every OCR line for the record; for notes, a scene's
    on-screen text is cut to OCR_CHARS. All of it made the prompt ~40% longer,
    mostly noise, and the local model's notes fell apart."""
    if line.startswith("**On screen") and len(line) > OCR_CHARS:
        return line[:OCR_CHARS].rsplit(" ", 1)[0] + " …"
    return line


def build_input(job_dir):
    """All transcripts of a job as one text; videos as their scene timeline
    (screenshot file + on-screen text + speech), so notes can point at screens."""
    res = pathlib.Path(job_dir) / "result"
    parts = []
    for t in sorted(res.glob("*/transcript.txt")):
        meta = json.loads((t.parent / "meta.json").read_text()) if (t.parent / "meta.json").exists() else {}
        tl = t.parent / "timeline.md"
        body = "\n".join(clip(l) for l in tl.read_text().splitlines() if not l.startswith("![")) if tl.exists() \
            else t.read_text()
        parts.append(f"=== FILE: {meta.get('file', t.parent.name)} "
                     f"({meta.get('kind', '')}, {meta.get('duration', '')}) ===\n{body}")
    return "\n\n".join(parts)


def needs_retry(notes_text):
    """Failed notes, or the local model's draft: worth asking Claude again."""
    return notes_text.startswith((PENDING, DRAFT))


def claude(text, cwd):
    """(notes, None) or (None, error message)."""
    cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence",
           "--tools", ""]
    if os.environ.get("INTAKE_NOTES_MODEL"):
        cmd += ["--model", os.environ["INTAKE_NOTES_MODEL"]]
    # Started from inside a Claude Code session (or its desktop app), the CLI
    # inherits that session's auth plumbing and fails with "OAuth session
    # expired". Strip it so the CLI uses its own login (your Pro plan).
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE_CODE_") or k in ("CLAUDECODE", "ANTHROPIC_BASE_URL"))}
    stderr = ""
    try:
        r = subprocess.run(cmd, input=PROMPT + "\n\n" + text, capture_output=True, text=True,
                           timeout=900, cwd=cwd, env=env)
        stderr = r.stderr
        out = json.loads(r.stdout or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        out = {"is_error": True, "result": str(e)}
    if out.get("is_error") or not out.get("result"):
        return None, (out.get("result") or stderr or "unknown error")[:300]
    return out["result"].strip(), None


def local(text):
    """The same prompt through the local model, in a child process."""
    if not (LOCAL_MODEL / "config.json").exists():
        return None, f"local model missing ({LOCAL_MODEL.relative_to(ROOT)})"
    try:
        r = subprocess.run([sys.executable, __file__, "--local"], input=PROMPT + "\n\n" + text,
                           capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return None, "local model timed out"
    if r.returncode or not r.stdout.strip():
        return None, (r.stderr.strip().splitlines() or ["local model failed"])[-1][:300]
    return r.stdout.strip(), None


LOCAL_LOG = pathlib.Path.home() / "Library" / "Logs" / "hinglish-intake-local-notes.log"


def notes_from(raw):
    """The notes in the local model's output, or None. It reasons first (in a
    thought channel, or in plain text) and repeats draft fragments there, so
    take the last "# title" that has a Summary under it, after any thought,
    and reject it if it repeats itself."""
    out = raw.rsplit("<channel|>", 1)[-1]
    starts = [m.start() for m in re.finditer(r"^# \S.*$", out, re.M) if "## Summary" in out[m.start():]]
    if not starts:
        return None
    out = out[starts[-1]:].strip()
    lines = [l for l in out.splitlines() if l.strip()]
    if re.search(r"(.{1,8}?)\1{15,}", out) or len(lines) - len(set(lines)) > 5:
        return None
    return out


THINK_TOKENS = 3000      # the local model's reasoning budget before it must answer


def run_local():
    """Child process: prompt on stdin, notes on stdout."""
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(str(LOCAL_MODEL))
    # Gemma 4 12B reasons before answering on a long transcript whatever the
    # template says. Stopped from reasoning at all it looped ("_r_r_r...");
    # left alone it sometimes reasoned in circles ("Wait, the prompt says...")
    # until the token limit and never answered. So it reasons, but at most
    # THINK_TOKENS and only while it isn't repeating itself; then the thought
    # is closed for it and it writes the notes. Gemma's own sampling settings
    # (temp 1.0, top-p 0.95, top-k 64) loop less than a low temperature.
    prompt = tok.apply_chat_template([{"role": "user", "content": sys.stdin.read()}],
                                     add_generation_prompt=True, enable_thinking=True)
    if len(prompt) > LOCAL_MAX_TOKENS:
        sys.exit(f"transcript too long for the local model ({len(prompt)} tokens)")
    sampler = make_sampler(temp=1.0, top_p=0.95, top_k=64)
    close = tok.encode("\n<channel|>", add_special_tokens=False)
    for attempt in (1, 2):     # a rejected draft gets one more go, model still loaded
        ids, raw, cut = [], "", None
        for r in stream_generate(model, tok, prompt, max_tokens=THINK_TOKENS + 4096, sampler=sampler):
            ids.append(r.token); raw += r.text
            if "<channel|>" in raw:
                continue        # answering: let it finish
            if len(ids) >= THINK_TOKENS:
                cut = "budget"
            elif len(ids) % 50 == 0 and len(raw) > 1500 and raw[-300:] in raw[:-300]:
                cut = "repeating"
            if cut:
                break
        if cut:                 # close the thought for it; the answer follows
            raw += "\n<channel|>"
            for r in stream_generate(model, tok, prompt + ids + close, max_tokens=4096, sampler=sampler):
                raw += r.text
        out = notes_from(raw)
        if out:
            break
        with open(LOCAL_LOG, "a") as f:
            f.write(f"\n===== {time.strftime('%F %T')} attempt {attempt}: rejected"
                    f"{f' (thought cut: {cut})' if cut else ''} =====\n{raw}\n")
    else:
        sys.exit(f"local model's notes were unusable twice (raw output in {LOCAL_LOG})")
    print(out.strip())


def fix_citations(notes, job_dir):
    """The local model sometimes garbles a file name in a citation
    ("WhatsApp Video 2021-10-12 ...") or leaves it out ("[02:17]"): snap
    near misses to the real names, and fill in the only file there is."""
    import difflib
    names = [json.loads(m.read_text()).get("file") for m in (pathlib.Path(job_dir) / "result").glob("*/meta.json")]
    names = [n for n in names if n]
    def fix(m):
        close = difflib.get_close_matches(m.group(1), names, n=1, cutoff=0.6)
        return f"[{close[0] if close else m.group(1)} @ "
    notes = re.sub(r"\[([^\[\]@]+?) @ ", fix, notes) if names else notes
    if len(names) == 1:         # a bare [mm:ss] can only mean the one recording
        notes = re.sub(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", lambda m: f"[{names[0]} @ {m.group(1)}]", notes)
    return notes


def generate(job_dir, log=print, fallback=True):
    """Write result/notes.md. Returns "claude", "local" (the fallback's
    draft) or None. Never raises: a job whose notes failed is still a
    finished job; the worker retries."""
    job_dir = pathlib.Path(job_dir)
    text = build_input(job_dir)
    if not text.strip():
        log("notes: no transcript text, skipped"); return None
    notes = job_dir / "result" / "notes.md"
    out, err = claude(text, job_dir)
    if out:
        notes.write_text(out + "\n")
        log(f"notes: {len(out.split())} words")
        return "claude"
    log(f"notes: Claude failed ({err})")
    if fallback:
        t0 = time.time()
        out, lerr = local(text)
        if out:
            out = fix_citations(out, job_dir)
            notes.write_text(f"{DRAFT} ({LOCAL_NAME}) because Claude was unavailable:** {err}. "
                             f"Check names, numbers and owners against the transcript; the processing "
                             f"Mac replaces these with Claude's notes when it can.\n\n{out}\n")
            log(f"notes: {len(out.split())} words by the local fallback model in {time.time() - t0:.0f}s")
            return "local"
        log(f"notes: local fallback failed ({lerr})")
        err += f"; local fallback: {lerr}"
    if notes.exists() and notes.read_text().startswith(DRAFT):
        return None             # keep the earlier draft until Claude replaces it
    notes.write_text(f"{PENDING}\n\nAutomatic notes failed: {err}\n\n"
                     f"The processing Mac retries every 30 minutes until it works (this page updates "
                     f"then); meanwhile ask Claude in your own session to write them from `bundle.md`.\n")
    return None


if __name__ == "__main__":
    if sys.argv[1:] == ["--local"]:
        run_local()
    else:
        sys.exit(0 if generate(sys.argv[1]) else 1)
