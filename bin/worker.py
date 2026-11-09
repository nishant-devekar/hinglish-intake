"""Processing worker: runs on this Mac, takes queued jobs one at a time.

    .venv/bin/python bin/worker.py                       # local store ./store
    .venv/bin/python bin/worker.py hf://<org>/intake      # the team's HF dataset
    .venv/bin/python bin/worker.py ... --once            # process what's queued, then exit
    .venv/bin/python bin/worker.py ... --reprocess <id>  # redo a finished job with the current
                                                         # pipeline, from its archived uploads;
                                                         # add --keep-notes to leave Claude's notes as they are

Jobs submitted while the Mac is off or asleep simply wait in the queue.
Progress lines go to the job's log (shown in the web app) at each stage,
not per line, to keep Hugging Face commits few.

Notes that failed (usually the Claude plan's usage limit), or that the local
fallback model drafted instead (bin/notes.py), are retried with Claude every
NOTES_RETRY_S while idle; only notes.md and bundle.md are re-uploaded.
"""
import json, pathlib, re, shutil, sys, tempfile, time, traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))
from store import open_store, now
import notes

IDLE_S = 30
ARCHIVE = ROOT / "archive"          # uploaded recordings, kept here once processed
SQUASH_EVERY_S = 7 * 24 * 3600     # reclaim HF storage at most weekly, when idle
NOTES_RETRY_S = 30 * 60            # usage limits reset after a few hours
notes_retry = {}                   # job id -> time of the next notes attempt
local_tried = set()                # jobs the local model already had its go at (~4 min, ~11 GB)


def stamp(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def write_job(job, d):
    """result/job.json: the job record and full log travel with the results
    (the web app's zip and `hf download` of result/ include it)."""
    if (d / "result").exists():
        (d / "result" / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=1))


def run_job(store, job):
    stamp(f"job {job['id']} ({len(job['files'])} files) from {job['submitted_by']}")
    job.update(status="processing", started_at=now())
    job["log"].append(f"{now()} started on the processing Mac")
    store.save(job)
    with tempfile.TemporaryDirectory(prefix="intake-") as tmp:
        d = store.fetch(job["id"], tmp)
        lines = []
        log = lambda m: (lines.append(m), stamp(f"  {m}"))
        try:
            from process import process
            process(d, log)
            job.update(status="done")
        except Exception as e:
            traceback.print_exc()
            job.update(status="failed", error=f"{type(e).__name__}: {e}")
        # Only stage lines into the shared log; per-batch ASR progress stays local.
        job["log"] += [f"{now()} {l}" for l in lines if "asr:" not in l]
        job.update(finished_at=now())
        done = job["status"] == "done"
        notes_md = d / "result" / "notes.md"
        if done and notes_md.exists() and notes.needs_retry(notes_md.read_text()):
            notes_retry[job["id"]] = time.time() + NOTES_RETRY_S
            local_tried.add(job["id"])      # process() already gave the local model its go
        write_job(job, d)
        if done and store.remote:
            # Keep the free 100 GB for results: originals move to this Mac.
            shutil.copytree(d / "input", ARCHIVE / job["id"], dirs_exist_ok=True)
            job["inputs"] = "archived on the processing Mac"
        store.save(job, d, with_result=done, drop_inputs=done and store.remote)
    stamp(f"job {job['id']}: {job['status']}")


def record_notes(res, old, by, line):
    """Put retried notes into README.md and bundle.md: the "notes written by"
    row, the processing log (so it doesn't end on the failure) and, in
    bundle.md, the notes themselves. bundle.md starts with README.md verbatim."""
    from process import NOTES_BY
    readme, bundle = res / "README.md", res / "bundle.md"
    was = readme.read_text() if readme.exists() else ""
    now_ = re.sub(r"^\| notes written by \|.*\|$", f"| notes written by | {NOTES_BY[by]} |",
                  was, count=1, flags=re.M)
    if now_.endswith("\n```\n"):                # README ends with the processing log block
        now_ = now_[:-4] + f"{line}\n```\n"
    if was:
        readme.write_text(now_)
    if bundle.exists():
        t = bundle.read_text()
        if was and t.startswith(was):
            t = now_ + t[len(was):]
        bundle.write_text(t.replace(old, (res / "notes.md").read_text(), 1))


def retry_notes(store, job):
    with tempfile.TemporaryDirectory(prefix="intake-notes-") as tmp:
        d = store.fetch(job["id"], tmp, text_result=True)
        old = (d / "result" / "notes.md").read_text()
        if not notes.needs_retry(old):          # written some other way meanwhile
            notes_retry.pop(job["id"]); return
        # The local model runs once per job: a draft waits for Claude, it isn't redrafted,
        # and a job it couldn't do isn't tried again every 30 minutes.
        fallback = old.startswith(notes.PENDING) and job["id"] not in local_tried
        local_tried.add(job["id"])
        by = notes.generate(d, lambda m: stamp(f"  {job['id']} {m}"), fallback=fallback)
        notes_retry[job["id"]] = time.time() + NOTES_RETRY_S
        if not by:
            return
        who = "Claude" if by == "claude" else "the local fallback model"
        record_notes(d / "result", old, by, f"notes: written by {who} on retry, {time.strftime('%H:%M')}")
        job["log"].append(f"{now()} notes written by {who}")
        write_job(job, d)
        store.save(job, d, with_result=True)
        if by == "claude":
            notes_retry.pop(job["id"])
    stamp(f"job {job['id']}: notes written by {by}")


def reprocess(store, job_id, keep_notes=False):
    """Run a finished job again with the current pipeline (e.g. after an
    upgrade), from the uploads archived on this Mac, and replace its results.
    keep_notes: keep Claude's notes as they are, only redo the evidence."""
    job = next(j for j in store.list_jobs() if j["id"] == job_id)
    src = ARCHIVE / job_id if store.remote else None
    with tempfile.TemporaryDirectory(prefix="intake-") as tmp:
        d = store.fetch(job_id, tmp, text_result=keep_notes)    # inputs come from the archive
        if src:
            if not src.exists():
                raise SystemExit(f"no archived uploads for {job_id} in {src}")
            shutil.copytree(src, d / "input", dirs_exist_ok=True)
        job.update(started_at=now())
        (d / "job.json").write_text(json.dumps(job, indent=1))
        lines = []
        from process import process
        process(d, lambda m: (lines.append(m), stamp(f"  {m}")), keep_notes)
        job["log"] += [f"{now()} re-processed with the current pipeline"] + \
                      [f"{now()} {l}" for l in lines if "asr:" not in l]
        job.update(status="done", finished_at=now())
        write_job(job, d)
        store.save(job, d, with_result=True)
    stamp(f"job {job_id}: re-processed")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    store = open_store(args[0] if args else None)
    once = "--once" in sys.argv
    if "--reprocess" in sys.argv:
        return reprocess(store, sys.argv[sys.argv.index("--reprocess") + 1], "--keep-notes" in sys.argv)
    # A job left "processing" means this worker died mid-job: queue it again.
    for j in store.list_jobs():
        if j["status"] == "processing":
            j["status"] = "queued"; j["log"].append(f"{now()} re-queued after worker restart")
            store.save(j)
    stamp(f"worker up, store={args[0] if args else 'store'}")
    for j in store.list_jobs():     # notes that failed before this start
        if j["status"] == "done" and notes.needs_retry(store.read(j["id"], "result/notes.md") or ""):
            notes_retry[j["id"]] = time.time()
    stamp_file = ROOT / "archive" / ".last-squash"
    while True:
        jobs = store.list_jobs()
        queued = sorted((j for j in jobs if j["status"] == "queued"), key=lambda j: j["submitted_at"])
        if queued:
            run_job(store, queued[0])
            continue
        due = [j for j in jobs if notes_retry.get(j["id"], float("inf")) <= time.time()]
        for j in due:
            try:
                retry_notes(store, j)
            except Exception as e:
                stamp(f"notes retry for {j['id']} failed: {e}")
                notes_retry[j["id"]] = time.time() + NOTES_RETRY_S
        if store.remote and time.time() - (stamp_file.stat().st_mtime if stamp_file.exists() else 0) > SQUASH_EVERY_S:
            try:
                store.reclaim(); stamp("squashed repo history to reclaim storage")
            except Exception as e:
                stamp(f"squash skipped: {e}")
            stamp_file.parent.mkdir(exist_ok=True); stamp_file.touch()
        if once:
            return
        time.sleep(IDLE_S)


if __name__ == "__main__":
    main()
