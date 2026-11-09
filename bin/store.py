"""Where jobs live: a local folder, or a private Hugging Face dataset repo.

Layout (same in both):

    jobs/<id>/job.json        id, title, submitted_by/at, options, status, log
    jobs/<id>/input/...       what the teammate uploaded
    jobs/<id>/result/...      what bin/process.py wrote

status: queued -> processing -> done | failed. The web app only creates
jobs; the worker on the Mac is the only writer after that, so there's no
write contention to manage.

    open_store("store")                      # local folder (tests, offline use)
    open_store("hf://your-org/intake")     # HF dataset; token from HF_TOKEN or `hf auth login`
"""
import datetime, json, os, pathlib, re, shutil, uuid


def new_id(title):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "job").lower()).strip("-")[:40] or "job"
    return f"{stamp}-{slug}-{uuid.uuid4().hex[:4]}"


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def new_job(title, submitted_by, files, options):
    return {"id": new_id(title), "title": title or "", "submitted_by": submitted_by or "unknown",
            "submitted_at": now(), "options": options or {}, "status": "queued",
            "files": [pathlib.Path(n).name for n, _ in files], "log": []}


class LocalStore:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)

    def create_job(self, title, submitted_by, files, options=None):
        """files: [(name, local_path)]"""
        job = new_job(title, submitted_by, files, options)
        d = self.root / "jobs" / job["id"]
        (d / "input").mkdir(parents=True)
        for name, path in files:
            shutil.copy(path, d / "input" / pathlib.Path(name).name)
        (d / "job.json").write_text(json.dumps(job, indent=1))
        return job

    def list_jobs(self):
        jobs = [json.loads(p.read_text()) for p in (self.root / "jobs").glob("*/job.json")]
        return sorted(jobs, key=lambda j: j["submitted_at"], reverse=True)

    def fetch(self, job_id, workdir=None, text_result=False):
        return self.root / "jobs" / job_id

    remote = False

    def save(self, job, local_dir=None, with_result=False, drop_inputs=False):
        (self.root / "jobs" / job["id"] / "job.json").write_text(json.dumps(job, indent=1))

    def reclaim(self):
        pass

    def read(self, job_id, rel):
        p = self.root / "jobs" / job_id / rel
        return p.read_text() if p.exists() else None

    def local_path(self, job_id, rel):
        p = self.root / "jobs" / job_id / rel
        return str(p) if p.exists() else None


class HFStore:
    def __init__(self, repo_id, token=None):
        from huggingface_hub import HfApi
        self.repo = repo_id
        self.api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        self.kw = {"repo_id": repo_id, "repo_type": "dataset"}

    def create_job(self, title, submitted_by, files, options=None):
        from huggingface_hub import CommitOperationAdd
        job = new_job(title, submitted_by, files, options)
        base = f"jobs/{job['id']}"
        ops = [CommitOperationAdd(f"{base}/input/{pathlib.Path(n).name}", p) for n, p in files]
        ops.append(CommitOperationAdd(f"{base}/job.json", json.dumps(job, indent=1).encode()))
        self.api.create_commit(operations=ops, commit_message=f"new job {job['id']}", **self.kw)
        return job

    def _job_files(self):
        return [f for f in self.api.list_repo_files(**self.kw) if re.fullmatch(r"jobs/[^/]+/job\.json", f)]

    def list_jobs(self):
        """One listing call plus one paths-info call; only job.json files whose
        content changed since last time are downloaded (the app refreshes
        every 30 s per open tab)."""
        from huggingface_hub import hf_hub_download
        if not hasattr(self, "_cache"):
            self._cache = {}                    # path -> (blob id, job)
        files = self._job_files()
        info = {i.path: i.blob_id for i in self.api.get_paths_info(paths=files, **self.kw)} if files else {}
        jobs = []
        for f in files:
            blob = info.get(f)
            if f not in self._cache or self._cache[f][0] != blob:
                try:
                    job = json.loads(pathlib.Path(hf_hub_download(filename=f, token=self.api.token,
                                                                  **self.kw)).read_text())
                except Exception:
                    continue
                self._cache[f] = (blob, job)
            jobs.append(self._cache[f][1])
        return sorted(jobs, key=lambda j: j["submitted_at"], reverse=True)

    def fetch(self, job_id, workdir, text_result=False):
        """Download job.json + input/ into workdir/<id>; returns that folder.
        text_result: instead of input/, the result's text (notes, bundle,
        transcripts, timelines), enough to rewrite the notes."""
        from huggingface_hub import snapshot_download
        base = f"jobs/{job_id}"
        more = [f"{base}/result/*.md", f"{base}/result/*/transcript.txt", f"{base}/result/*/meta.json"] \
            if text_result else [f"{base}/input/*"]
        snapshot_download(local_dir=workdir, allow_patterns=[f"{base}/job.json"] + more,
                          token=self.api.token, **self.kw)
        return pathlib.Path(workdir) / "jobs" / job_id

    remote = True

    def save(self, job, local_dir=None, with_result=False, drop_inputs=False):
        """drop_inputs: remove the uploaded recordings from the repo (the
        worker archives them on the Mac first) to stay inside the free 100 GB."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        base = f"jobs/{job['id']}"
        ops = [CommitOperationAdd(f"{base}/job.json", json.dumps(job, indent=1).encode())]
        if drop_inputs:
            ops.append(CommitOperationDelete(f"{base}/input/", is_folder=True))
        if with_result and local_dir and (pathlib.Path(local_dir) / "result").exists():
            for p in sorted((pathlib.Path(local_dir) / "result").rglob("*")):
                if p.is_file():
                    ops.append(CommitOperationAdd(f"{base}/{p.relative_to(local_dir)}", str(p)))
        self.api.create_commit(operations=ops, commit_message=f"{job['id']}: {job['status']}", **self.kw)

    def reclaim(self):
        """Deleted files keep counting against storage until history is
        squashed. Current files are untouched; only old versions go."""
        self.api.super_squash_history(commit_message="squash: reclaim storage of processed uploads", **self.kw)

    def read(self, job_id, rel):
        p = self.local_path(job_id, rel)
        return pathlib.Path(p).read_text() if p else None

    def local_path(self, job_id, rel):
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(filename=f"jobs/{job_id}/{rel}", token=self.api.token, **self.kw)
        except Exception:
            return None


def open_store(spec=None):
    spec = spec or os.environ.get("INTAKE_STORE", "store")
    if spec.startswith("hf://"):
        return HFStore(spec[5:])
    return LocalStore(spec)
