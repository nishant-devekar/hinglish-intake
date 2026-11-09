"""Video -> scenes: one sharp screenshot per visually distinct stretch, with
its time span and on-screen text.

Every second of video is looked at (1 fps; 30 fps would be the same picture
30 times). A new scene starts when the picture differs clearly from the
start of the current one, measured by a 16x16 difference hash, so slow
changes (scrolling, typing, a pen on a whiteboard) add up until they count.
The threshold adapts: a screen recording keeps the sensitive default, so a
screen shown for a second still gets its screenshot; hand-held camera
footage changes every second (shake, a hand moving), so there it is raised
until scenes average >= MIN_AVG_S (Video 3, a phone filming a whiteboard:
1030 "scenes" at the default, ~1 per second). The two are told apart by how
many seconds are still: 56-74% in our screen recordings, none in Video 3.
Applying the budget to screen recordings too merged 45 s of a UI demo (four
different screens) into one scene.
Scenes longer than MAX_SCENE_S are split so long static stretches still get
a frame now and then. Each scene keeps its *sharpest* second, which matters
for phone-filmed screens where most frames are motion-blurred.

    frames/frame-0001.jpg ...    one per scene
    frames/index.txt             frame -> time (and scene span)
    frames/scenes.json           [{frame, t, start, end}]
    frames/ocr.txt               on-screen text per frame (macOS Vision)
"""
import json, pathlib, shutil, subprocess, tempfile
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
WIDTH = 1280          # stills are evidence: keep text legible
MAX_SCENE_S = 45      # a frame at least this often, even on a static screen
CHANGE_BITS = 28      # of 256 hash bits: the sensitive default for screen recordings
MIN_AVG_S = 8         # camera footage: raise the threshold until scenes average this long
STILL_BITS = 6        # a second that changed by at most this much counts as still
STILL_SHARE = 0.25    # fewer still seconds than this = hand-held camera footage


def has_video(path):
    """True for a real video stream; an mp3's embedded cover art doesn't count."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                          "-show_entries", "stream=codec_name:stream_disposition=attached_pic",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return any(line and not line.endswith(",1") for line in out.splitlines())


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def _gray(path, size):
    return np.asarray(Image.open(path).convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


def dhash(path):
    g = _gray(path, (17, 16))
    return (g[:, 1:] > g[:, :-1]).flatten()


def sharpness(path):
    """Variance of the Laplacian: higher = crisper edges = more legible text."""
    g = _gray(path, (640, 360))
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _cut(hashes, bits):
    """Indices where a new scene starts, for a given change threshold."""
    starts, ref = [0], hashes[0]
    for i in range(1, len(hashes)):
        if np.count_nonzero(hashes[i] != ref) > bits or i - starts[-1] >= MAX_SCENE_S:
            starts.append(i); ref = hashes[i]
    return starts


def scenes_from(stills):
    """stills: [(t, path)] at 1 s steps -> [(start, end, best_path, best_t)]."""
    if not stills:
        return [], CHANGE_BITS
    hashes = [dhash(p) for _, p in stills]
    still = [np.count_nonzero(a != b) <= STILL_BITS for a, b in zip(hashes, hashes[1:])]
    bits = CHANGE_BITS
    if still and np.mean(still) < STILL_SHARE:
        budget = max(1, len(stills) // MIN_AVG_S)
        while len(_cut(hashes, bits)) > budget and bits < 256:
            bits += 4
    starts = _cut(hashes, bits) + [len(stills)]
    res = []
    for a, b in zip(starts, starts[1:]):
        t, p = max(stills[a:b], key=lambda x: sharpness(x[1]))
        res.append((stills[a][0], stills[b - 1][0] + 1, p, t))
    return res, bits


def extract(src, out, ocr=True, log=print):
    """Write frames/ for a video; returns the scene list."""
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frames-") as tmp:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-vf",
                        f"fps=1,scale='min({WIDTH},iw)':-2", "-q:v", "3", f"{tmp}/s-%06d.jpg"], check=True)
        stills = [(i, p) for i, p in enumerate(sorted(pathlib.Path(tmp).glob("s-*.jpg")))]
        scenes, bits = scenes_from(stills)
        meta = []
        for n, (start, end, p, t) in enumerate(scenes, 1):
            name = f"frame-{n:04d}.jpg"
            shutil.copy(p, out / name)
            meta.append({"frame": name, "t": float(t), "start": float(start), "end": float(end)})
    log(f"{len(stills)} s of video -> {len(meta)} scenes (change threshold {bits}/256)")
    (out / "scenes.json").write_text(json.dumps(meta, indent=1))
    (out / "index.txt").write_text("".join(f"{m['frame']}  t={m['t']:.1f}s  scene={m['start']:.0f}-{m['end']:.0f}s\n"
                                           for m in meta))
    if ocr and meta:
        r = subprocess.run(["swift", str(ROOT / "bin" / "ocr.swift"), str(out)], capture_output=True, text=True)
        if r.returncode == 0:
            (out / "ocr.txt").write_text(r.stdout)
    return meta


def read_ocr(frames_dir):
    """{frame: [lines]}; frames whose text didn't change map to the previous list."""
    p = pathlib.Path(frames_dir) / "ocr.txt"
    res, cur, prev = {}, None, []
    if not p.exists():
        return res
    for line in p.read_text().splitlines():
        if line.startswith("== "):
            cur = line[3:]; res[cur] = []
        elif cur and line == "(same as previous)":
            res[cur] = list(prev)
        elif cur and line and line != "(no text)":
            res[cur].append(line)
        if cur and not line.startswith("== "):
            prev = res[cur]
    return res


if __name__ == "__main__":     # frames.py <video> <frames_dir> [--ocr]   (used by ingest.sh)
    import sys
    extract(sys.argv[1], sys.argv[2], ocr="--ocr" in sys.argv[3:], log=lambda m: print(f"    {m}"))
