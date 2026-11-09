"""Speaker-labelled transcript for English, Hindi/Hinglish and Marathi audio.

    .venv/bin/python bin/transcribe.py <audio-or-video> --out DIR
        [--engine auto|tara|tara-torch|qwen3|marathi|vibevoice] [--speakers N] [--no-diarize] [--no-names]
        [--hotwords "Asha,GST,..."]

Writes DIR/transcript.txt ("[mm:ss] Asha: ..." for enrolled voices, see bin/voices.py; else S1), transcript.srt, transcript.json.

Why the pipeline looks like this (see eval/REPORT.md for the numbers):
  * loudness normalisation: some calls are recorded at -50 dB and Whisper
    hallucinates on them;
  * Silero VAD before ASR: every Whisper-family model invents text on
    silence ("झाल झाल", "एक ज़ीरो एक ज़ीरो ...") and VAD removes the silence;
  * diarization (CPU, child process) in parallel with ASR (GPU) on VAD units;
    each line then takes its majority speaker, and the few units that span
    two people are re-transcribed turn by turn;
  * Tara (Whisper-large-v3 fine-tuned for Hinglish) in mixed-code mode:
    English words in Latin script, Hindi in Devanagari, verbatim;
  * the "auto" engine (default) picks the model per unit: Marathi stretches
    go to IndicWhisper Marathi, the rest to Tara (see Auto, LangID).
"""
import argparse, json, pathlib, re, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SR = 16000
MAX_UNIT = 28.0     # Whisper sees 30 s windows; leave headroom

CONFIG = ROOT / "config"


def _lines(name):
    """Non-empty, non-comment lines of config/<name>, or [] if it doesn't exist."""
    f = CONFIG / name
    if not f.exists():
        return []
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


# Names and jargon your calls use, one per line in config/hotwords.txt
# (see config/hotwords.example.txt). Biasing the models toward them is cheap
# and fixes the most visible errors.
DEFAULT_HOTWORDS = _lines("hotwords.txt") or ["WhatsApp", "GST", "UPI", "onboarding"]


# --- audio ---------------------------------------------------------------

def load_audio(path):
    """Decode anything ffmpeg reads to 16 kHz mono float32, loudness-normalised.
    dynaudnorm lifts quiet speakers without pumping silence up to speech level."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn",
           "-af", "highpass=f=80,dynaudnorm=f=250:g=15:p=0.9:m=20",
           "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout,
                         dtype=np.float32).copy()


def speech_regions(audio):
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(MODELS / "diar" / "silero_vad.onnx")
    cfg.silero_vad.threshold = 0.45
    cfg.silero_vad.min_silence_duration = 0.4
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = MAX_UNIT
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    win = cfg.silero_vad.window_size
    out = []
    for i in range(0, len(audio), win):
        vad.accept_waveform(audio[i:i + win])
        while not vad.empty():
            s = vad.front.start / SR
            out.append((s, s + len(vad.front.samples) / SR)); vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front.start / SR
        out.append((s, s + len(vad.front.samples) / SR)); vad.pop()
    pad = 0.15
    return [(max(0, s - pad), e + pad) for s, e in out]


def fold_minor_speakers(turns, min_share=0.05, min_s=8.0):
    """A "speaker" with a few seconds of talk is almost always the clusterer
    splitting one voice (a laugh, a one-word reply). Give those turns to the
    nearest real speaker instead of inventing a person."""
    talk = {}
    for s, e, k in turns:
        talk[k] = talk.get(k, 0) + e - s
    total = sum(talk.values()) or 1
    major = {k for k, t in talk.items() if t >= min_s and t / total >= min_share}
    if not major:
        return turns
    out = []
    for i, (s, e, k) in enumerate(turns):
        if k not in major:
            near = [t for t in turns if t[2] in major]
            k = min(near, key=lambda t: min(abs(t[0] - e), abs(s - t[1])))[2]
        out.append((s, e, k))
    return out


def units(regions, turns):
    """Intersect VAD speech with speaker turns, then merge neighbours from the
    same speaker into pieces of at most MAX_UNIT seconds. Speech that falls in
    a gap between diarization turns is kept and given to the nearest speaker;
    dropping it loses words at every turn boundary."""
    pieces = []
    for s, e in regions:
        cuts = sorted((max(s, ts), min(e, te), k) for ts, te, k in turns
                      if min(e, te) - max(s, ts) > 0)
        t = s
        for a, b, k in cuts:
            if a - t >= 0.3:
                pieces.append([t, a, None])
            if b - max(a, t) >= 0.3:
                pieces.append([max(a, t), b, k])
            t = max(t, b)
        if e - t >= 0.3:
            pieces.append([t, e, None])
    for i, p in enumerate(pieces):
        if p[2] is None:
            near = [q for q in pieces if q[2] is not None]
            if near:
                p[2] = min(near, key=lambda q: min(abs(q[0] - p[1]), abs(p[0] - q[1])))[2]
    merged = []
    for s, e, spk in pieces:
        if merged and merged[-1][2] == spk and s - merged[-1][1] < 1.0 and e - merged[-1][0] <= MAX_UNIT:
            merged[-1][1] = e
        else:
            merged.append([s, e, spk])
    return merged


# Words the models reliably mishear. Applied after ASR; cheaper and more
# predictable than prompting. Extend as new misspellings show up.
CORRECTIONS = {
    r"\b[Ww]hats ?[Aa]pp?\b": "WhatsApp",
    r"व्हाट्सएप|वाट्सएप|वॉट्सऐप|व्हाट्सऐप": "WhatsApp",
    r"\bwhatsapp\b": "WhatsApp",
    r"\b[Tt]rello\b|\b[Tt]rillo\b": "Trello",
    # Tara writes the filler "uh" as "अः", even mid-English sentence.
    r"(?<!\S)अः(?!\S)": "uh",
}
# Your own misheard names, as "regex<TAB>replacement" lines in
# config/corrections.tsv (see config/corrections.example.tsv), e.g. a
# colleague's name in Devanagari -> Latin script.
for _l in _lines("corrections.tsv"):
    _pat, _, _rep = _l.partition("\t")
    if _rep:
        CORRECTIONS[_pat] = _rep


def correct(text):
    for pat, rep in CORRECTIONS.items():
        text = re.sub(pat, rep, text)
    return text


def collapse_loops(text, keep=2):
    """Cap any 1-4 word phrase repeated back-to-back at `keep` copies; the
    tell-tale failure of autoregressive ASR on noise."""
    w = text.split()
    for n in (4, 3, 2, 1):
        out, i = [], 0
        while i < len(w):
            j = i + n
            reps = 1
            while w[j:j + n] == w[i:i + n] and j + n <= len(w):
                reps += 1; j += n
            out += w[i:i + n] * min(reps, keep) if reps > 1 else w[i:i + 1]
            i = j if reps > 1 else i + 1
        w = out
    return " ".join(w)


# --- engines ---------------------------------------------------------------

def tara_engine(hotwords):
    """Tara on MLX (bin/tara_mlx.py) when converted weights exist, else the
    PyTorch build. Same weights and decoding rules; MLX batches far better."""
    if (MODELS / "hf" / "tara-mlx" / "weights.safetensors").exists():
        sys.path.insert(0, str(ROOT / "bin"))
        from tara_mlx import TaraMLX
        return TaraMLX()
    print("    tara-mlx not converted (bin/tara_mlx.py convert); using PyTorch", file=sys.stderr)
    return Tara(hotwords)


class Tara:
    def __init__(self, hotwords):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "mps" else torch.float32
        path = MODELS / "hf" / "tara"
        self.proc = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            path, torch_dtype=self.dtype).to(self.dev).eval()
        tk = self.proc.tokenizer
        self.tok = {t: tk.convert_tokens_to_ids(f"<|{t}|>") for t in
                    ("startoftranscript", "hi", "en", "mixedcode", "transcribe", "notimestamps")}
        # No hotword prompt: in testing it didn't fix names ("footbridge") and
        # made Tara Title-Case ordinary English words. CORRECTIONS does names.

    def _prefix(self, lang):
        t = self.tok
        ids = [t["startoftranscript"], t[lang]] + ([t["mixedcode"]] if lang == "hi" else []) \
            + [t["transcribe"], t["notimestamps"]]
        return self.torch.tensor([ids], device=self.dev)

    def batch(self, clips):
        """Transcribe several <=30 s clips at once; Whisper pads every clip to
        30 s anyway, so batching short speaker turns is close to free."""
        from transformers import GenerationMixin
        feats = self.proc(clips, sampling_rate=SR, return_tensors="pt") \
                    .input_features.to(self.dev, self.dtype)
        out = [None] * len(clips)
        with self.torch.no_grad():
            # Language from the first decoder step, limited to hi/en: pure-English
            # stretches read better in English mode, everything else in mixed-code.
            # (Tara can't hear Marathi: its fine-tune gives that token ~0.)
            sot = self.torch.full((len(clips), 1), self.tok["startoftranscript"], device=self.dev)
            logits = self.model(input_features=feats, decoder_input_ids=sot).logits[:, -1]
            langs = ["en" if l[self.tok["en"]] > l[self.tok["hi"]] + 2.0 else "hi" for l in logits]
            for lang in set(langs):
                idx = [i for i, l in enumerate(langs) if l == lang]
                prefix = self._prefix(lang).expand(len(idx), -1)
                # No repetition penalty: people really do repeat phrases, and the
                # penalty made Tara stop early instead. Loops are bounded by a
                # token budget from clip length and removed by collapse_loops.
                budget = min(440, int(max(len(clips[i]) for i in idx) / SR * 12) + 24)
                ids = GenerationMixin.generate(self.model, input_features=feats[idx],
                                               decoder_input_ids=prefix, max_new_tokens=budget,
                                               num_beams=1)
                for i, row in zip(idx, ids[:, prefix.shape[1]:]):
                    out[i] = (self.proc.tokenizer.decode(row, skip_special_tokens=True).strip(), lang)
        return out


def marathi_engine(hotwords):
    """IndicWhisper Marathi on MLX (bin/tara_mlx.py MarathiMLX: 12x realtime,
    greedy on <= 12 s pieces) when converted, else the PyTorch beam-5 build
    (0.9x realtime, and it drops phrases)."""
    if (MODELS / "hf" / "indicwhisper-marathi-mlx" / "weights.safetensors").exists():
        sys.path.insert(0, str(ROOT / "bin"))
        from tara_mlx import MarathiMLX
        return MarathiMLX()
    print("    indicwhisper-marathi-mlx not converted (bin/tara_mlx.py convert marathi); using PyTorch",
          file=sys.stderr)
    return Marathi(hotwords)


class Marathi:
    """AI4Bharat IndicWhisper: Whisper-medium fine-tuned on Marathi (Vistaar,
    ~18% WER). The general models loop or drift into Hindi on Marathi calls.
    English loanwords come out in Devanagari ("स्टॅंडर्ड ऑपरेटिंग प्रोसेजर")."""
    def __init__(self, hotwords):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "mps" else torch.float32
        path = MODELS / "hf" / "indicwhisper-marathi"
        self.proc = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            path, torch_dtype=self.dtype).to(self.dev).eval()
        # Saved by transformers 4.27: its forced_decoder_ids clash with language=.
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.max_length = None

    def batch(self, clips):
        feats = self.proc(clips, sampling_rate=SR, return_tensors="pt") \
                    .input_features.to(self.dev, self.dtype)
        budget = min(440, int(max(len(c) for c in clips) / SR * 12) + 24)
        with self.torch.no_grad():
            # Beam search + no 4-gram repeats: greedy decoding loops on this
            # checkpoint far more than on Tara.
            ids = self.model.generate(feats, language="mr", task="transcribe", num_beams=5,
                                      no_repeat_ngram_size=4, max_new_tokens=budget)
        return [(t.strip(), "mr") for t in self.proc.batch_decode(ids, skip_special_tokens=True)]


class LangID:
    """Hindi or Marathi, per speech unit: VoxLingua107 ECAPA (SpeechBrain,
    Apache-2.0, 81 MB, CPU). On our calls it put the Marathi call at
    99 % Marathi and three Hinglish calls at 100 % Hindi; Tara can't hear
    Marathi at all. It is poor at Indian-accented English (calls it Hindi or
    Marathi), so English is left to Tara's own language guess."""
    def __init__(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        path = str(MODELS / "hf" / "voxlingua107-ecapa")
        self.clf = EncoderClassifier.from_hparams(source=path, savedir=path, run_opts={"device": "cpu"},
                                                  overrides={"pretrained_path": path})
        lab = self.clf.hparams.label_encoder.lab2ind
        self.hi, self.mr = (lab[next(l for l in lab if l.startswith(k + ":"))] for k in ("hi", "mr"))

    def marathi(self, audio, us, sure=3.0, short=3.0, window=90.0):
        """True per unit if it sounds like Marathi rather than Hindi. A unit
        the model is unsure of (log-odds under `sure`), or under `short`
        seconds ("हा", "okay"), takes the duration-weighted vote of the units
        within `window` seconds: a conversation stays in one language for a while.
        On the Marathi call that took Marathi from 94 % to 99 % of its units."""
        import torch
        score = []
        for s, e, _ in us:
            lp = self.clf.classify_batch(torch.from_numpy(audio[int(s * SR):int(e * SR)]).unsqueeze(0))[0][0]
            score.append(float(np.clip(lp[self.mr] - lp[self.hi], -5, 5)))     # log-odds, capped
        out = []
        for (s, e, _), v in zip(us, score):
            if abs(v) < sure or e - s < short:
                mid = (s + e) / 2
                near = [(w, u[1] - u[0]) for u, w in zip(us, score) if abs((u[0] + u[1]) / 2 - mid) <= window]
                v = sum(w * d for w, d in near) / sum(d for _, d in near)
            out.append(v > 0)
        return out


def latin_share(text):
    words = [w for w in text.split() if re.search(r"[A-Za-zऀ-ॿ]", w)]
    return sum(bool(re.search(r"[A-Za-z]", w)) for w in words) / len(words) if words else 0.0


class Auto:
    """The default engine: each speech unit goes to the best model for its
    language, so one file can move between English, Hindi/Hinglish and Marathi.
      English          Tara, English mode (Tara's own first-token guess)
      Hindi, Hinglish  Tara, mixed-code mode
      Marathi          IndicWhisper Marathi (loaded the first time it's needed)
    Every unit is transcribed by Tara first. It is redone in Marathi when
    LangID hears Marathi AND Tara's text is mostly Devanagari: LangID calls
    Indian-accented English "Marathi" too, but Tara writes that in Latin
    script (78-100 % Latin on an English screen recording, vs mostly < 60 %
    on the Marathi call). Marathi that's mostly English words stays with
    Tara, which writes the English in Latin script."""
    LATIN_MAX = 0.6

    def __init__(self, hotwords):
        self.hotwords = hotwords
        self.tara = get_engine("tara", hotwords)
        self.marathi_ok = True
        try:
            self.lid = LangID()
        except Exception as e:      # missing weights shouldn't stop Hindi/English jobs
            print(f"    auto: language ID unavailable ({e}); Marathi will go through Tara", file=sys.stderr)
            self.lid = None

    def plan(self, audio, us):
        return self.lid.marathi(audio, us) if self.lid else None

    def batch(self, clips, marathi=None):
        out = self.tara.batch(clips)
        idx = [i for i, (text, lang) in enumerate(out)
               if marathi and marathi[i] and lang == "hi" and latin_share(text) < self.LATIN_MAX]
        if idx and self.marathi_ok:
            try:
                res = get_engine("marathi", self.hotwords).batch([clips[i] for i in idx])
            except Exception as e:
                print(f"    auto: Marathi model unavailable ({e}); keeping Tara's text", file=sys.stderr)
                self.marathi_ok = False
                return out
            for i, r in zip(idx, res):
                out[i] = r
        return out


class Qwen3:
    def __init__(self, hotwords):
        from mlx_audio.stt.utils import load_model
        self.model = load_model(str(MODELS / "hf" / "qwen3-asr-1.7b-8bit"))
        self.hotwords = hotwords

    def batch(self, clips):
        return [self._one(c) for c in clips]

    def _one(self, audio):
        # Language left to auto-detect on purpose: forcing Hindi made the model
        # read the hotword list back out on near-silent audio.
        r = self.model.generate(audio, hotwords=self.hotwords or None, repetition_penalty=1.05)
        text = r.text.strip()
        if re.search(r"[぀-ヿ一-鿿가-힯]", text):
            # Auto-detect sometimes picks Chinese on a short "हाँ"/"ना"; our
            # audio is only ever Hindi/English, so retry pinned to Hindi.
            r = self.model.generate(audio, language="Hindi", repetition_penalty=1.05)
            text = r.text.strip()
        if self.hotwords and sum(h.lower() in text.lower() for h in self.hotwords) >= len(self.hotwords) // 2:
            text = ""   # prompt regurgitation, not speech
        return text, getattr(r, "language", None)


def vibevoice(audio, hotwords):
    """One pass over the whole file: ASR + speakers + timestamps together.
    Best at who-said-what; less verbatim (it sometimes romanises or
    translates Hindi), so it is an option rather than the default."""
    from mlx_audio.stt.utils import load_model
    m = load_model(str(MODELS / "hf" / "vibevoice-asr-4bit"))
    r = m.generate(audio, sampling_rate=SR, hotwords=hotwords or None, repetition_penalty=1.05)
    return [{"start": float(s.get("start", 0)), "end": float(s.get("end", 0)),
             "speaker": s.get("speaker_id", s.get("speaker")), "text": correct(s.get("text", "").strip())}
            for s in (r.segments or [])]


def transcribe_units(engine, audio, us, log, B=8):
    segs = []
    plan = engine.plan(audio, us) if hasattr(engine, "plan") else None     # "auto": which units sound Marathi
    for b in range(0, len(us), B):
        chunk = us[b:b + B]
        clips = [audio[int(s * SR):int(e * SR)] for s, e, _ in chunk]
        res = engine.batch(clips, plan[b:b + B]) if plan else engine.batch(clips)
        for (s, e, spk), (text, lang) in zip(chunk, res):
            text = correct(collapse_loops(text))
            if text:
                segs.append({"start": round(s, 2), "end": round(e, 2), "speaker": spk,
                             "lang": lang, "text": text})
        log(f"asr: {min(b + B, len(us))}/{len(us)}")
    return segs


def attach_speakers(segs, turns, mixed_s=1.0):
    """Give each segment the speaker who talks most inside it. Segments where
    a second speaker has >= mixed_s seconds are dropped and returned as spans
    to re-transcribe turn by turn, so no line mixes two people's words."""
    kept, redo = [], []
    for seg in segs:
        share = {}
        for ts, te, k in turns:
            o = min(seg["end"], te) - max(seg["start"], ts)
            if o > 0:
                share[k] = share.get(k, 0) + o
        if share and sum(share.values()) - max(share.values()) >= mixed_s:
            redo.append((seg["start"], seg["end"]))
            continue
        if share:
            seg["speaker"] = max(share, key=share.get)
        elif turns:     # in a gap between turns: nearest speaker
            seg["speaker"] = min(turns, key=lambda t: min(abs(t[0] - seg["end"]),
                                                          abs(seg["start"] - t[1])))[2]
        kept.append(seg)
    return kept, redo


# --- output -----------------------------------------------------------------

def mmss(t):
    return f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}" if t >= 3600 \
        else f"{int(t // 60):02d}:{int(t % 60):02d}"


def srt_time(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(t % 1 * 1000):03d}"


def write(out, segs, meta):
    names, n = {}, 0
    for s in segs:      # enrolled voices keep their name; others S1, S2... by first appearance
        k = s["speaker"]
        if k is not None and k not in names:
            if isinstance(k, str):
                names[k] = k
            else:
                n += 1; names[k] = f"S{n}"
    lines, srt = [], []
    for i, s in enumerate(segs, 1):
        who = names.get(s["speaker"])
        lines.append(f"[{mmss(s['start'])}] {who + ': ' if who else ''}{s['text']}")
        srt.append(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n"
                   f"{'[' + who + '] ' if who else ''}{s['text']}\n")
        s["speaker"] = who
    (out / "transcript.txt").write_text("\n".join(lines) + "\n")
    (out / "transcript.srt").write_text("\n".join(srt))
    (out / "transcript.json").write_text(json.dumps({"meta": meta, "segments": segs},
                                                    ensure_ascii=False, indent=1))
    return len(names)


ENGINES = ["auto", "tara", "tara-torch", "qwen3", "marathi", "vibevoice"]
LANGS = {"en": "English", "hi": "Hindi/Hinglish", "mr": "Marathi"}
_engines = {}


def get_engine(name, hotwords):
    """Engines are cached per process: the worker keeps Tara loaded between jobs."""
    if name not in _engines:
        _engines[name] = {"auto": Auto, "tara": tara_engine, "tara-torch": Tara, "qwen3": Qwen3,
                          "marathi": marathi_engine}[name](hotwords)
    return _engines[name]


def languages(segs):
    """Share of transcribed speech time per language, from each line's engine."""
    t = {}
    for s in segs:
        k = LANGS.get(s.get("lang"))
        if k:
            t[k] = t.get(k, 0) + s["end"] - s["start"]
    total = sum(t.values())
    return {k: round(v / total, 3) for k, v in sorted(t.items(), key=lambda x: -x[1])} if total else {}


def run(src, out, engine="auto", speakers=-1, diarize=True, names=True,
        hotwords=DEFAULT_HOTWORDS, log=None):
    """Transcribe one audio/video file into out/transcript.{txt,srt,json}.
    Returns (segments, meta)."""
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    log = log or (lambda m: print(f"    {m}", file=sys.stderr, flush=True))
    t0 = time.time()
    audio = load_audio(src)
    dur = len(audio) / SR

    if engine == "vibevoice":
        log(f"vibevoice: one pass over {dur:.0f}s")
        segs = vibevoice(audio, hotwords)
    else:
        regions = speech_regions(audio)
        log(f"vad: {sum(e - s for s, e in regions):.0f}s speech of {dur:.0f}s")
        # Diarization (CPU) runs in a child process while ASR (GPU) works on
        # VAD units; speakers are attached afterwards. On the Oct 9 call only
        # 4 of 91 units held a second speaker, and those are redone per turn.
        pending, tmp = None, None
        if diarize:
            import soundfile as sf, tempfile, multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            sys.path.insert(0, str(ROOT / "bin"))
            from diarize import diarize as diarize_file
            tmp = tempfile.NamedTemporaryFile(suffix=".wav")
            sf.write(tmp.name, audio, SR)
            pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
            pending = pool.submit(diarize_file, tmp.name, speakers)
        eng = get_engine(engine, hotwords)
        us = units(regions, [])
        segs = transcribe_units(eng, audio, us, log)
        if pending:
            try:
                turns = pending.result()
                if speakers <= 0:       # a head-count given by the user is trusted as is
                    turns = fold_minor_speakers(turns)
            except Exception as e:      # a transcript without speakers beats none
                log(f"diarize failed ({e}); continuing without speaker labels")
                turns = []
            pool.shutdown(); tmp.close()
            log(f"diarize: {len({t[2] for t in turns})} speakers, {len(turns)} turns")
            segs, redo = attach_speakers(segs, turns)
            if redo:
                log(f"re-transcribing {len(redo)} units that span two speakers")
                segs += transcribe_units(eng, audio, units(redo, turns), log)
            segs.sort(key=lambda s: s["start"])
            if names:
                from voices import name_speakers
                name_speakers(audio, segs, log)
    meta = {"source": str(src), "engine": engine, "duration_s": round(dur, 1),
            "elapsed_s": round(time.time() - t0, 1), "languages": languages(segs)}
    if meta["languages"]:
        log("languages: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in meta["languages"].items()))
    n = write(out, segs, meta)
    meta["speakers"] = n
    meta["words"] = sum(len(s["text"].split()) for s in segs)
    log(f"{engine}: {meta['words']} words, {n} speakers, {time.time() - t0:.0f}s")
    return segs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="auto", choices=ENGINES)
    ap.add_argument("--speakers", type=int, default=-1, help="known speaker count (best accuracy)")
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--no-names", action="store_true", help="keep S1/S2 even for enrolled voices")
    ap.add_argument("--hotwords", default=",".join(DEFAULT_HOTWORDS))
    a = ap.parse_args()
    hot = [h.strip() for h in a.hotwords.split(",") if h.strip()]
    _, meta = run(a.src, a.out, a.engine, a.speakers, not a.no_diarize, not a.no_names, hot)
    print(f"{meta['words']} words, {meta['speakers']} speakers")


if __name__ == "__main__":
    main()
