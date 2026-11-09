"""Whisper models on MLX, natively on Apple Silicon instead of PyTorch-on-MPS:
Tara (Whisper-large-v3 Hinglish) and IndicWhisper Marathi (Whisper-medium).

    .venv/bin/python bin/tara_mlx.py convert [--q8]      # once: models/hf/tara -> tara-mlx(-q8)
    .venv/bin/python bin/tara_mlx.py convert marathi     # once: indicwhisper-marathi -> ...-mlx
    .venv/bin/python bin/tara_mlx.py clip.wav [marathi]  # quick test on one <=30 s clip

The layers come from mlx-audio's Whisper; the decoding loop is ours because
Tara needs a prefix the stock decoder can't express
(<|startoftranscript|><|hi|><|mixedcode|><|transcribe|><|notimestamps|>),
and because speaker turns are batched.
"""
import json, pathlib, shutil, sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_audio.stt.models.whisper.whisper import Model, ModelDimensions
from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram, N_SAMPLES, N_FRAMES

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "hf" / "tara"
SR = 16000

# Whisper special-token ids (large-v3 vocabulary; Tara appends <|mixedcode|>).
EOT, SOT = 50257, 50258
SPECIAL_FROM = 50258          # everything from SOT up is control, never text
TOKENIZER_FILES = ["added_tokens.json", "merges.txt", "normalizer.json", "preprocessor_config.json",
                   "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
                   "vocab.json", "generation_config.json", "config.json"]


def convert(q8=False, src=SRC):
    dst = src.parent / (src.name + ("-mlx-q8" if q8 else "-mlx"))
    dst.mkdir(exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    model = Model(ModelDimensions.from_dict(cfg), mx.float16)
    weights = model.sanitize(mx.load(str(src / "model.safetensors")))
    model.update(tree_unflatten(list(weights.items())))
    if q8:
        # Decoding is memory-bound; 8-bit linears halve the bytes read per token.
        nn.quantize(model, group_size=64, bits=8,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear))
    mx.eval(model.parameters())
    flat = dict(_flatten(model.parameters()))
    mx.save_safetensors(str(dst / "weights.safetensors"), flat)
    for f in TOKENIZER_FILES:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    (dst / "mlx.json").write_text(json.dumps({"q8": q8}))
    print(f"{dst}: {sum(v.nbytes for v in flat.values()) / 1e9:.2f} GB")


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flatten(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], tree


class WhisperMLX:
    """A converted Whisper checkpoint and a batched greedy decoder."""
    def __init__(self, path, how):
        from transformers import WhisperTokenizer
        if not (path / "weights.safetensors").exists():
            raise SystemExit(f"{path.name} missing: run  bin/tara_mlx.py {how}")
        cfg = json.loads((path / "config.json").read_text())
        self.model = Model(ModelDimensions.from_dict(cfg), mx.float16)
        if json.loads((path / "mlx.json").read_text())["q8"]:
            nn.quantize(self.model, group_size=64, bits=8,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear))
        self.model.load_weights(str(path / "weights.safetensors"))
        mx.eval(self.model.parameters())
        # MLX keeps freed buffers for reuse; uncapped, that cache reached 5 GB
        # on top of the 6 GB working set and pushed a 24 GB Mac into swap.
        mx.set_cache_limit(512 * 1024 ** 2)
        self.tk = WhisperTokenizer.from_pretrained(path)
        gen = json.loads((path / "generation_config.json").read_text())
        n_vocab = cfg["vocab_size"]
        # Same constraints HF generate applied: no control tokens mid-text,
        # Whisper's suppress list, and no blank/EOT as the very first token.
        sup = np.zeros(n_vocab, dtype=np.float32)
        sup[SPECIAL_FROM:] = -np.inf
        sup[gen["suppress_tokens"]] = -np.inf
        self.suppress = mx.array(sup, dtype=mx.float16)
        first = sup.copy(); first[gen["begin_suppress_tokens"]] = -np.inf
        self.suppress_first = mx.array(first, dtype=mx.float16)

    def _mel(self, clips):
        mels = []
        for a in clips:
            a = mx.array(np.asarray(a[:N_SAMPLES], dtype=np.float32))
            a = mx.pad(a, (0, N_SAMPLES - a.shape[0]))
            m = log_mel_spectrogram(a, n_mels=self.model.dims.n_mels, padding=0)
            mels.append(m[:N_FRAMES] if m.shape[0] >= N_FRAMES else mx.pad(m, ((0, N_FRAMES - m.shape[0]), (0, 0))))
        return mx.stack(mels).astype(mx.float16)

    # mlx-audio's decoder grows its KV cache by concatenation, which reallocates
    # every layer's cache on every token; at batch 8 that made each step ~10x
    # slower by token 200. This step function writes into a cache allocated
    # once and uses MLX's fused attention kernel, with the same weights.

    def _heads(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)

    def _merge(self, x):
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, -1)

    def _step(self, tokens, offset, self_kv, cross_kv):
        dec = self.model.decoder
        L = tokens.shape[1]
        x = dec.token_embedding(tokens) + dec.positional_embedding[offset:offset + L]
        for blk, (kc, vc), (ck, cv) in zip(dec.blocks, self_kv, cross_kv):
            h = blk.attn_ln(x)
            kc[:, :, offset:offset + L, :] = self._heads(blk.attn.key(h))
            vc[:, :, offset:offset + L, :] = self._heads(blk.attn.value(h))
            o = mx.fast.scaled_dot_product_attention(
                self._heads(blk.attn.query(h)), kc[:, :, :offset + L], vc[:, :, :offset + L],
                scale=self.scale, mask="causal" if L > 1 else None)
            x = x + blk.attn.out(self._merge(o))
            h = blk.cross_attn_ln(x)
            o = mx.fast.scaled_dot_product_attention(self._heads(blk.cross_attn.query(h)), ck, cv,
                                                     scale=self.scale)
            x = x + blk.cross_attn.out(self._merge(o))
            x = x + blk.mlp2(nn.gelu(blk.mlp1(blk.mlp_ln(x))))
        return dec.token_embedding.as_linear(dec.ln(x[:, -1:]))[:, 0]

    def tok(self, t):
        return self.tk.convert_tokens_to_ids(f"<|{t}|>")

    def _greedy(self, xa, prefix, budget, no_repeat=0):
        """no_repeat: never emit a token that repeats an n-gram already in the
        row (HF's no_repeat_ngram_size), the guard against looping. With it,
        self.logprobs holds each row's chosen-token log-probabilities."""
        dec, dims = self.model.decoder, self.model.dims
        self.n_head = dims.n_text_head
        self.scale = (dims.n_text_state // self.n_head) ** -0.5
        B = xa.shape[0]
        T = min(dims.n_text_ctx, len(prefix) + budget)
        dh = dims.n_text_state // self.n_head
        self_kv = [(mx.zeros((B, self.n_head, T, dh), mx.float16),
                    mx.zeros((B, self.n_head, T, dh), mx.float16)) for _ in dec.blocks]
        cross_kv = [(self._heads(b.cross_attn.key(xa)), self._heads(b.cross_attn.value(xa)))
                    for b in dec.blocks]
        logits = self._step(mx.array([prefix] * B), 0, self_kv, cross_kv)
        offset = len(prefix)
        out = [[] for _ in range(B)]
        self.logprobs = [[] for _ in range(B)]
        live = list(range(B))           # rows still decoding
        for step in range(min(budget, T - offset)):
            logits = logits + (self.suppress_first if step == 0 else self.suppress)
            if no_repeat:
                l = np.array(logits.astype(mx.float32))
                for j, r in enumerate(live):
                    seq = out[r]
                    if len(seq) >= no_repeat - 1:
                        key = seq[len(seq) - no_repeat + 1:]
                        for i in range(len(seq) - no_repeat + 1):
                            if seq[i:i + no_repeat - 1] == key:
                                l[j, seq[i + no_repeat - 1]] = -np.inf
                nxt = l.argmax(-1)
                top = l.max(-1, keepdims=True)
                lse = (top + np.log(np.exp(l - top).sum(-1, keepdims=True)))[:, 0]
                for j, r in enumerate(live):
                    self.logprobs[r].append(float(l[j, nxt[j]] - lse[j]))
            else:
                nxt = np.array(mx.argmax(logits, axis=-1))
            keep = []
            for j, tok in enumerate(nxt.tolist()):
                if tok == EOT:
                    continue
                out[live[j]].append(tok); keep.append(j)
            if not keep:
                break
            if len(keep) < len(live):   # drop finished rows so they stop costing compute
                k = mx.array(keep)
                self_kv = [(kc[k], vc[k]) for kc, vc in self_kv]
                cross_kv = [(ck[k], cv[k]) for ck, cv in cross_kv]
                live = [live[j] for j in keep]; nxt = nxt[keep]
            logits = self._step(mx.array(nxt)[:, None], offset, self_kv, cross_kv)
            offset += 1
        return [self.tk.decode(r, skip_special_tokens=True).strip() for r in out]


class TaraMLX(WhisperMLX):
    def __init__(self, q8=False):
        super().__init__(SRC.parent / ("tara-mlx-q8" if q8 else "tara-mlx"), "convert" + (" --q8" if q8 else ""))
        self.ids = {t: self.tok(t) for t in ("hi", "en", "mixedcode", "transcribe", "notimestamps")}

    def _prefix(self, lang):
        i = self.ids
        return [SOT, i[lang]] + ([i["mixedcode"]] if lang == "hi" else []) + [i["transcribe"], i["notimestamps"]]

    def batch(self, clips):
        """[(text, lang)] for each <=30 s clip; same rules as the PyTorch engine."""
        xa = self.model.encoder(self._mel(clips))
        sot_logits, _, _ = self.model.decoder(mx.full((len(clips), 1), SOT), xa)
        l = np.array(sot_logits[:, -1].astype(mx.float32))
        langs = ["en" if r[self.ids["en"]] > r[self.ids["hi"]] + 2.0 else "hi" for r in l]
        out = [None] * len(clips)
        for lang in set(langs):
            idx = [i for i, g in enumerate(langs) if g == lang]
            budget = min(440, int(max(len(clips[i]) for i in idx) / SR * 12) + 24)
            for i, t in zip(idx, self._greedy(xa[mx.array(idx)], self._prefix(lang), budget)):
                out[i] = (t, lang)
        return out


class MarathiMLX(WhisperMLX):
    """IndicWhisper Marathi (Whisper-medium), greedy with no repeated 4-grams,
    on pieces of at most PIECE_S seconds.

    Measured on 30 lines (413 s) of the Marathi call against the old
    PyTorch beam-5 decoding (0.9x realtime): beam search there silently drops
    whole phrases (a line's opening sentence), and greedy over a full 28 s
    unit runs into invented text in its second half. Greedy on <= 12 s pieces
    keeps the openings, reads coherently to the end, has the highest model
    confidence of the three, and runs 12x faster than realtime."""
    PIECE_S = 12.0

    def __init__(self):
        super().__init__(SRC.parent / "indicwhisper-marathi-mlx", "convert marathi")
        self.prefix = [SOT, self.tok("mr"), self.tok("transcribe"), self.tok("notimestamps")]

    @classmethod
    def pieces(cls, clip):
        """Cut at the quietest 100 ms in the middle half until each piece fits."""
        n, f = len(clip), SR // 10
        if n <= cls.PIECE_S * SR:
            return [clip]
        energy = [float(np.mean(clip[i:i + f] ** 2)) for i in range(n // 4, 3 * n // 4 - f, f // 2)]
        cut = n // 4 + int(np.argmin(energy)) * (f // 2) + f // 2
        return cls.pieces(clip[:cut]) + cls.pieces(clip[cut:])

    def batch(self, clips, B=8):
        flat, owner = [], []
        for i, c in enumerate(clips):
            for p in self.pieces(np.asarray(c)):
                flat.append(p); owner.append(i)
        texts = []
        for b in range(0, len(flat), B):
            part = flat[b:b + B]
            budget = min(440, int(max(len(c) for c in part) / SR * 12) + 24)
            texts += self._greedy(self.model.encoder(self._mel(part)), self.prefix, budget, no_repeat=4)
        return [(" ".join(t for t, o in zip(texts, owner) if o == i and t), "mr") for i in range(len(clips))]


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "convert":
        convert(q8="--q8" in a, src=SRC.parent / "indicwhisper-marathi" if "marathi" in a else SRC)
    else:
        import soundfile as sf, time
        audio, sr = sf.read(a[0], dtype="float32")
        m = MarathiMLX() if "marathi" in a else TaraMLX(q8="--q8" in a)
        t = time.time()
        print(m.batch([audio[:N_SAMPLES]]), f"{time.time() - t:.1f}s")
