#!/usr/bin/env bash
# Video intake: turn a video into frames + a timestamped transcript.
#
#   ingest.sh <video> [--scene | --fps 0.2 | --every 5]
#             [--engine tara|tara-torch|qwen3|marathi|vibevoice|whisper] [--speakers N] [--ocr]
#             [--model large-v3-turbo]
#
# --engine picks the speech model (default tara: Hinglish-tuned Whisper, with
# VAD + speaker labels; see eval/REPORT.md). whisper is the old whisper.cpp
# path and needs only brew, no .venv; --model applies to it alone.
#
# Output lands in ../out/<video-basename>/ as:
#   frames/frame-0001.jpg ...   stills Claude can look at (scene mode: one per scene)
#   frames/index.txt            frame number -> timestamp in the video
#   frames/ocr.txt              on-screen text per frame (--ocr; macOS Vision)
#   timeline.md / .json         scene mode: each screenshot with what was said while it showed
#   evidence.pdf                the same with screenshots embedded (for claude.ai)
#   transcript.txt             "[mm:ss] S1: ..." speaker-labelled speech
#   transcript.srt / .json      same, for players / for code
#   audio.wav                   16kHz mono, what whisper actually read
#   summary.txt                 metadata + what was produced

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/models"

die() { printf 'error: %s\n' "$1" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: ingest.sh <video> [--scene N|--fps N|--every N] [--engine NAME] [--speakers N] [--ocr] [--model NAME]"
SRC="$1"; shift
[ -f "$SRC" ] || die "no such file: $SRC"

MODE="scene"; FPS=""; EVERY=""; MODEL="large-v3-turbo"; ENGINE="tara"; SPEAKERS=""; OCR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scene) MODE="scene"; shift; [[ "${1:-}" =~ ^[0-9.]+$ ]] && shift ;;   # old --scene N: value unused
    --fps)   MODE="fps";   FPS="$2";   shift 2 ;;
    --every) MODE="every"; EVERY="$2"; shift 2 ;;
    --model) MODEL="$2";               shift 2 ;;
    --engine) ENGINE="$2";             shift 2 ;;
    --speakers) SPEAKERS="$2";         shift 2 ;;
    --ocr)   OCR=1;                    shift ;;
    *) die "unknown option: $1" ;;
  esac
done

command -v ffmpeg >/dev/null || die "ffmpeg not installed (brew install ffmpeg)"

BASE="$(basename "${SRC%.*}" | tr ' ' '_')"
OUT="$ROOT/out/$BASE"
rm -rf "$OUT"; mkdir -p "$OUT/frames"

printf '==> %s\n' "$BASE"

# --- probe -------------------------------------------------------------
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC" 2>/dev/null || echo 0)
DUR_INT=${DUR%.*}; DUR_INT=${DUR_INT:-0}
HAS_AUDIO=$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$SRC" 2>/dev/null | head -1)
VINFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height -of csv=p=0 "$SRC" 2>/dev/null || echo "?")
printf '    %ss, video: %s, audio: %s\n' "$DUR_INT" "$VINFO" "$([ -n "$HAS_AUDIO" ] && echo yes || echo no)"

# --- frames ------------------------------------------------------------
# Scene mode (bin/frames.py) looks at every second and keeps one sharp
# screenshot per distinct screen; it needs the .venv. Fixed rates are there
# for footage that changes continuously.
PY="$ROOT/.venv/bin/python"
if [ "$MODE" = "scene" ] && [ -x "$PY" ]; then
  "$PY" "$ROOT/bin/frames.py" "$SRC" "$OUT/frames" ${OCR:+--ocr}
  N=$(find "$OUT/frames" -name 'frame-*.jpg' | wc -l | tr -d ' ')
else
case "$MODE" in
  scene) VF="select='eq(n\,0)+gt(scene,0.3)',showinfo" ;;
  fps)   VF="fps=$FPS,showinfo" ;;
  every) VF="fps=1/$EVERY,showinfo" ;;
esac

ffmpeg -nostdin -v info -i "$SRC" -vf "$VF" -fps_mode vfr -q:v 3 \
       "$OUT/frames/frame-%04d.jpg" 2>"$OUT/.showinfo" || true

# Recover each frame's real timestamp from showinfo so frames can be cited
# against the transcript.
awk '/pts_time:/ { for (i=1;i<=NF;i++) if ($i ~ /^pts_time:/) { split($i,a,":"); print a[2] } }' \
    "$OUT/.showinfo" > "$OUT/.times" 2>/dev/null || true

N=$(find "$OUT/frames" -name 'frame-*.jpg' | wc -l | tr -d ' ')

# Scene detection can come up empty on static footage; fall back so we never
# hand back zero frames.
if [ "$N" -eq 0 ] && [ "$MODE" = "scene" ]; then
  INT=$(( DUR_INT / 12 )); [ "$INT" -lt 1 ] && INT=1
  printf '    no scene changes found, falling back to 1 frame / %ss\n' "$INT"
  ffmpeg -nostdin -v info -i "$SRC" -vf "fps=1/$INT,showinfo" -fps_mode vfr -q:v 3 \
         "$OUT/frames/frame-%04d.jpg" 2>"$OUT/.showinfo" || true
  awk '/pts_time:/ { for (i=1;i<=NF;i++) if ($i ~ /^pts_time:/) { split($i,a,":"); print a[2] } }' \
      "$OUT/.showinfo" > "$OUT/.times" 2>/dev/null || true
  N=$(find "$OUT/frames" -name 'frame-*.jpg' | wc -l | tr -d ' ')
fi

: > "$OUT/frames/index.txt"
i=1
while IFS= read -r t; do
  [ -n "$t" ] || continue
  printf 'frame-%04d.jpg  t=%.1fs\n' "$i" "$t" >> "$OUT/frames/index.txt"
  i=$((i+1))
done < "$OUT/.times"
rm -f "$OUT/.showinfo" "$OUT/.times"

printf '    %s frames\n' "$N"

if [ -n "$OCR" ] && [ "$N" -gt 0 ]; then
  swift "$ROOT/bin/ocr.swift" "$OUT/frames" > "$OUT/frames/ocr.txt" 2>/dev/null || true
fi
fi
[ -n "$OCR" ] && [ -s "$OUT/frames/ocr.txt" ] && printf '    ocr: frames/ocr.txt\n'

# --- audio + transcript ------------------------------------------------
TRANSCRIPT="(no audio track)"
if [ -n "$HAS_AUDIO" ]; then
  # 16kHz mono PCM, loudness-normalised: some calls arrive at -50 dB and
  # every model hallucinates on audio that quiet.
  ffmpeg -nostdin -v error -i "$SRC" -af "highpass=f=80,dynaudnorm=f=250:g=15:p=0.9:m=20" \
         -ar 16000 -ac 1 -c:a pcm_s16le -y "$OUT/audio.wav"

  if [ "$ENGINE" != "whisper" ] && [ -x "$PY" ]; then
    printf '    transcribing (%s)...\n' "$ENGINE"
    if W=$("$PY" "$ROOT/bin/transcribe.py" "$OUT/audio.wav" --out "$OUT" --engine "$ENGINE" \
             ${SPEAKERS:+--speakers "$SPEAKERS"} 2>"$OUT/.asr.log"); then
      TRANSCRIPT="$W ($ENGINE)"; rm -f "$OUT/.asr.log"
    else
      TRANSCRIPT="(transcribe.py failed - see .asr.log; audio.wav kept)"
    fi
  else
    [ "$ENGINE" = "whisper" ] || printf '    no .venv - falling back to whisper.cpp\n'
    WHISPER=""
    for c in whisper-cli whisper-cpp whisper; do
      command -v "$c" >/dev/null && { WHISPER="$c"; break; }
    done
    VAD=()
    [ -f "$MODEL_DIR/ggml-silero-v5.1.2.bin" ] && VAD=(--vad -vm "$MODEL_DIR/ggml-silero-v5.1.2.bin")

    if [ -z "$WHISPER" ]; then
      TRANSCRIPT="(whisper not installed - audio.wav kept)"
    elif [ ! -f "$MODEL_DIR/ggml-$MODEL.bin" ]; then
      TRANSCRIPT="(model ggml-$MODEL.bin missing - audio.wav kept)"
    elif [ "$(stat -f%z "$MODEL_DIR/ggml-$MODEL.bin")" -lt 20000000 ]; then
      TRANSCRIPT="(model ggml-$MODEL.bin incomplete - re-download it)"
    else
      printf '    transcribing (whisper.cpp %s)...\n' "$MODEL"
      # VAD + no carried context: the two settings that stop whisper looping
      # on silence and on its own previous output.
      "$WHISPER" -m "$MODEL_DIR/ggml-$MODEL.bin" -f "$OUT/audio.wav" -l auto -mc 0 \
                 ${VAD[@]+"${VAD[@]}"} -otxt -osrt -of "$OUT/transcript" >"$OUT/.whisper.log" 2>&1 || true
      if [ -s "$OUT/transcript.txt" ]; then
        W=$(wc -w < "$OUT/transcript.txt" | tr -d ' ')
        TRANSCRIPT="$W words"
        rm -f "$OUT/.whisper.log"
      else
        TRANSCRIPT="(no transcript - see .whisper.log)"
      fi
    fi
  fi
  printf '    speech: %s\n' "$TRANSCRIPT"
fi

# --- timeline: screenshots joined with what was said -------------------
# Needs speech timings (transcript.json); whisper.cpp only writes .txt/.srt.
if [ -f "$OUT/frames/scenes.json" ] && { [ -f "$OUT/transcript.json" ] || [ -z "$HAS_AUDIO" ]; }; then
  "$PY" "$ROOT/bin/evidence.py" "$OUT" "$(basename "$SRC")" | sed 's/^/    /' || true
fi

# --- summary -----------------------------------------------------------
{
  printf 'source:     %s\n' "$SRC"
  printf 'duration:   %ss\n' "$DUR_INT"
  printf 'video:      %s\n' "$VINFO"
  printf 'sampling:   %s\n' "$MODE"
  printf 'frames:     %s\n' "$N"
  printf 'speech:     %s\n' "$TRANSCRIPT"
  [ -z "$OCR" ] || printf 'ocr:        frames/ocr.txt\n'
  [ ! -f "$OUT/timeline.md" ] || printf 'timeline:   timeline.md, timeline.json%s\n' "$([ -f "$OUT/evidence.pdf" ] && echo ', evidence.pdf')"
} > "$OUT/summary.txt"

printf '==> %s\n' "$OUT"
