#!/usr/bin/env bash
# Download the models into models/. Safe to re-run: finished downloads are skipped.
#
#   bin/fetch_models.sh                 # core: VAD + diarization + speaker names (~50 MB),
#                                       #       Tara for English/Hindi/Hinglish (~3 GB) + MLX conversion
#   bin/fetch_models.sh --marathi       # + Hindi/Marathi language ID (81 MB) and IndicWhisper Marathi (1.4 GB)
#   bin/fetch_models.sh --qwen3         # + Qwen3-ASR 1.7B, second-opinion engine (2.3 GB)
#   bin/fetch_models.sh --vibevoice     # + VibeVoice-ASR 4-bit, long meetings (5.3 GB)
#   bin/fetch_models.sh --notes-local   # + Gemma 4 12B 4-bit, notes when Claude is unavailable (~7 GB)
#   bin/fetch_models.sh --all
#
# Needs the venv first (README "Install"): it uses .venv/bin/hf and .venv/bin/python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="$ROOT/models"; HF="$ROOT/.venv/bin/hf"; PY="$ROOT/.venv/bin/python"
[ -x "$HF" ] || { echo "error: no .venv yet - see README \"Install\"" >&2; exit 1; }

MARATHI=""; QWEN=""; VIBE=""; NOTES=""
for a in "$@"; do
  case "$a" in
    --marathi) MARATHI=1 ;; --qwen3) QWEN=1 ;; --vibevoice) VIBE=1 ;; --notes-local) NOTES=1 ;;
    --all) MARATHI=1; QWEN=1; VIBE=1; NOTES=1 ;;
    *) sed -n '2,12p' "$0"; exit 1 ;;
  esac
done

get() {   # url dest
  [ -s "$2" ] && return 0
  echo "fetching $(basename "$2")"
  curl -fL --retry 3 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}
hfget() { # repo dir
  [ -f "$2/.done" ] && return 0
  echo "fetching $1 -> ${2#$ROOT/}"
  "$HF" download "$1" --local-dir "$2" && touch "$2/.done"
}

# --- core -----------------------------------------------------------------
SO=https://github.com/k2-fsa/sherpa-onnx/releases/download
mkdir -p "$M/diar" "$M/hf"
get "$SO/asr-models/silero_vad.onnx" "$M/diar/silero_vad.onnx"
get "$SO/speaker-recongition-models/nemo_en_titanet_large.onnx" "$M/diar/nemo_en_titanet_large.onnx"
if [ ! -f "$M/diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]; then
  get "$SO/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" "$M/diar/seg.tar.bz2"
  tar -xjf "$M/diar/seg.tar.bz2" -C "$M/diar" && rm "$M/diar/seg.tar.bz2"
fi

hfget Trelis/tara "$M/hf/tara"
[ -d "$M/hf/tara-mlx" ] || "$PY" "$ROOT/bin/tara_mlx.py" convert

# --- optional ---------------------------------------------------------------
if [ -n "$MARATHI" ]; then
  hfget speechbrain/lang-id-voxlingua107-ecapa "$M/hf/voxlingua107-ecapa"
  cp -n "$M/hf/voxlingua107-ecapa/label_encoder.txt" "$M/hf/voxlingua107-ecapa/label_encoder.ckpt" || true
  D="$M/hf/indicwhisper-marathi"
  if [ ! -f "$D/model.safetensors" ]; then
    # The host drops long transfers; curl -C - resumes, so just re-run on failure.
    get https://indicwhisper.objectstore.e2enetworks.net/marathi_models.zip "$M/hf/marathi_models.zip"
    T="$(mktemp -d)"; unzip -q "$M/hf/marathi_models.zip" 'marathi_models/whisper-medium-mr_alldata_multigpu/*' -d "$T"
    mkdir -p "$D" && cp -R "$T/marathi_models/whisper-medium-mr_alldata_multigpu/." "$D/" && rm -rf "$T"
    "$PY" -c "import sys; from transformers import WhisperForConditionalGeneration as W; d=sys.argv[1]; W.from_pretrained(d).save_pretrained(d)" "$D"
    rm "$M/hf/marathi_models.zip"
  fi
  [ -d "$M/hf/indicwhisper-marathi-mlx" ] || "$PY" "$ROOT/bin/tara_mlx.py" convert marathi
fi
[ -n "$QWEN" ]  && hfget mlx-community/Qwen3-ASR-1.7B-8bit "$M/hf/qwen3-asr-1.7b-8bit"
[ -n "$VIBE" ]  && hfget mlx-community/VibeVoice-ASR-4bit "$M/hf/vibevoice-asr-4bit"
[ -n "$NOTES" ] && hfget mlx-community/gemma-4-12B-it-4bit "$M/hf/gemma-4-12b-it-4bit"
echo "models ready in models/"
