# Speech model evaluation — 2026-10-11

> Measured on our own private team calls, which are not in this repo. Speaker
> names in this report are pseudonyms (Asha, Ravi, Meera, Guest 1/2).
> To run the same comparison on your calls, see `eval/` and `eval/gold/README.md`.

Goal: replace the hand-run whisper.cpp passes (`out/*/run.sh`, `passes/`,
`chunks/`) with one command that gives an accurate, speaker-labelled
transcript of our team calls (Hinglish, 2–4 people, Google Meet / phone
audio) and of English screen recordings.

## Test set

Six 90 s clips cut from real recordings in `out/` (`eval/clips/`):

| clip | what it is | why it's there |
|---|---|---|
| `call1009_hinglish` | Oct 9 call, 2:30–4:00 | dense code-switching, Asha |
| `call1005_hinglish` | Oct 5 call, 11:00–12:30 | architecture talk, 2 speakers |
| `ptt_voicenote_hinglish` | Oct 5 voice note | single speaker; has a hand-cleaned version in `out/ptt-…/transcript-clean.md` |
| `video3_english` | Video 3, 2:00–3:30 | English screen recording |
| `whatsapp` | WhatsApp demo video | English narration over music |
| `silence_trap` | Oct 5 call, 5:00–6:30 | ~95 % silence/screen-share; tests hallucination |

**There is no human reference transcript yet**, so there's no WER. The
scorecard below measures what can be checked without one, and every output
was also read side by side (`eval/results/<clip>.<model>.txt`).

## Scorecard (`eval/score.py`)

| model | speed (× realtime) | junk words on silence | coverage | verdict |
|---|---|---|---|---|
| **Tara** (Whisper-large-v3 Hinglish FT, mixed-code mode) | 10.5 | 92 → **0 with VAD** | 97 % | **Best verbatim Hinglish.** English in Latin, Hindi in Devanagari, names/terms right (GST, WhatsApp IVR, insight engine). Hallucinates on silence, so VAD is mandatory. |
| **Qwen3-ASR 1.7B** (auto language) | 11.5 | **0** | 96 % | Best all-rounder: punctuation, zero hallucination, fast. Sometimes writes English words in Devanagari (स्टोर, फोन नंबर), occasional mishearing ("nudges" → "Nagesh"). |
| Qwen3-ASR, language forced to Hindi | 8.2 | 18 | 96 % | **Don't use.** On silence it printed the hotword list back as the transcript. |
| **VibeVoice-ASR 9B** (4-bit) | 3.8 | 0 (tags `[Environmental Sounds]`) | 97 % | Only model that does speakers + timestamps + text in one pass over a whole hour. But not verbatim: romanises Hindi or drifts into English paraphrase ("In its bagal…"). Use for who-said-what, not for quotes. |
| whisper.cpp turbo, `-l auto` | 10.8 | 6 | 85 % | Romanised Hinglish ("bhot acha lag raha hai") on voice notes, Devanagari elsewhere; drops content. |
| whisper.cpp turbo, `-l hi` + Silero VAD | 9.4 | 0 | 85 % | Big improvement over the current default for two flags, but loops on music (28× "जार्ट") and loses the first seconds of clips. |
| whisper.cpp small, translate | 28.8 | 9 | 89 % | Readable English gist; not a transcript (lossy, invents: "Add the chopped onions" on silence). |
| whisper.cpp turbo, `-l hi` (**current default for calls**) | 7.8 | 5 | 77 % | Worst: Devanagari transliteration of English, 13× loops ("वाट्साप इस…"), drops ~25 % of speech, emits broken UTF-8 (`�`). |

Coverage = words relative to the longest output for that clip, averaged; a
proxy for dropped speech, not accuracy.

### Published numbers (for context, not measured here)

| model | benchmark | WER |
|---|---|---|
| Tara | Vistaar Hindi mean (7 sets) | 12.06 (Sarvam Saaras-v3 12.32, ElevenLabs Scribe-v2 13.79) |
| Tara | Code-switch FLEURS | 8.37 |
| Trelis whisper-hinglish-preview | CoSHE-500 code-switched | 13.67 (Sarvam 11.47, Scribe-v2 12.43, whisper-large-v3 29.74) |

## Root causes found in the old pipeline

1. **Volume.** One test call has stretches at −50 dB mean. Every Whisper-family
   model invents text on audio that quiet ("झाल झाल", "Thank you. Thank you.").
   Now: `highpass + dynaudnorm` on every input.
2. **No VAD.** Silence and screen-share music produce the loops that forced
   the manual chunk/pass workflow. Now: Silero VAD gates all ASR.
3. **`-l hi` on Hinglish.** Forces English words into Devanagari and,
   with context carry-over, triggers loops. Now: Tara mixed-code mode.
4. **No speakers.** Calls were hand-attributed from Google Meet captions.
   Now: pyannote segmentation-3.0 + speaker embeddings (sherpa-onnx), 17×
   realtime on CPU.

## What was added

| file | what |
|---|---|
| `bin/transcribe.py` | normalise → VAD → diarize → ASR per speaker turn → loop removal → name corrections → `transcript.txt/.srt/.json` |
| `bin/diarize.py` | who-spoke-when, standalone |
| `bin/ocr.swift` | on-screen text from frames (macOS Vision, no download) |
| `bin/ingest.sh` | now calls the above; `--engine tara\|qwen3\|vibevoice\|whisper`, `--speakers N`, `--ocr`; whisper fallback gained VAD + `-mc 0` |
| `models/hf/tara`, `qwen3-asr-1.7b-8bit`, `vibevoice-asr-4bit` | 2.9 + 2.3 + 5.3 GB |
| `models/diar/` | segmentation, speaker embedding, Silero VAD (≈50 MB) |
| `eval/` | clips, runners, per-model outputs, `score.py` |

## Full call, end to end

Oct 9 call, 22 min, `--speakers 3` (`eval/full/`):

| engine | wall time | words | notes |
|---|---|---|---|
| tara | 5 m 47 s (3.8× realtime) | 3,408 | clean Hinglish, speaker turns, product names spelled right |
| qwen3 | 6 m 01 s | 3,414 | most English rendered in Devanagari on short turns; once emitted Chinese (`嗯，社群。`) → now retried in Hindi |

Old way for the same call: two whisper.cpp passes over 44 hand-cut chunks
plus targeted re-runs (`out/call-2026-10-09/run.sh`, `tgt/`), no speakers.

Breakdown: VAD <5 s, diarization ~75 s (CPU), ASR ~270 s. Whisper pads every
speaker turn to 30 s, so ASR cost scales with the number of turns, not minutes.

## Tara on MLX (`bin/tara_mlx.py`)

Same weights, converted once (`bin/tara_mlx.py convert`, ~10 s, 3.1 GB);
mlx-audio's Whisper layers plus our own batched greedy decoder, because the
stock decoder can't send the `<|hi|><|mixedcode|>` prefix. Now the default
`tara` engine; PyTorch remains as `tara-torch`.

Oct 9 call, 22 min, nothing else on the GPU:

| | tara-torch | tara (MLX) |
|---|---|---|
| wall time, end to end | 266 s | **179 s** (7.4× realtime) |
| peak memory | 7.3 GB | **4.4 GB** |
| output | 3,408 words | 3,404 words — 99.94 % identical; 3 eval clips 100 % identical |

What mattered, measured:

- **Preallocated KV cache + fused attention** in the decoder. mlx-audio grows
  its cache by concatenation; at batch 8 a decode step went from 50 ms to
  320 ms by token 200. Fixed: 8 clips 45 s → 21 s.
- **`mx.set_cache_limit(512 MB)`.** Uncapped, MLX held 5 GB of freed buffers on
  top of a 6 GB working set; together with another transcription job that filled
  16 GB of swap.
- **Diarization on 8 threads** (the performance cores): 75 s → 58 s, same turns.

Tried and dropped: 8-bit weights (slower on every test), `mx.compile` of the
decode step (no change: decoding is bandwidth-bound, not launch-bound), fused
attention in the encoder (no change), batch 16 (12 % faster but 9.5 GB peak),
sorting turns by length (no change).

Where the 179 s goes now: diarization ~58 s (CPU), ASR ~110 s (GPU), load/VAD
~10 s. The decoder sits near the M4 Pro's memory-bandwidth floor (~12 ms per
step for 3 GB of weights), so the remaining big lever is running diarization
on the CPU *while* the GPU transcribes, not a faster ASR kernel.

## Diarization overlapped with ASR

`transcribe.py` now starts diarization in a child process (CPU) and
transcribes VAD units on the GPU at the same time; when both finish, each
line takes the speaker who talks most inside it. Units where a second speaker
has ≥ 1 s are re-transcribed turn by turn, so lines still never mix two
people's words.

On the Oct 9 call, merging VAD regions across pauses < 1 s gives 91 units
(vs 117 speaker turns), and only 4 of them span two speakers (~80 s redone).

| Oct 9 call, 22 min | sequential (diarize → ASR) | overlapped |
|---|---|---|
| wall time | 179 s | **125 s** (10.5× realtime) |
| peak memory | 4.4 GB | 3.9 GB |
| speaker label per word | — | 99.88 % the same (4 of 3,315 words differ) |
| text | — | 97.4 % word agreement; differences are mostly spelling variants (पहुँचिएगा/पहुंचिएगा) and punctuation; of 54 that change words, neither side is consistently better (overlap: "phone number", "purchase route"; sequential: "first autologin") |

Overall since the start of this work: 266 s (PyTorch, sequential) → 125 s,
7.3 GB → 3.9 GB, same model and same transcript quality.

Also changed: `--speakers N` is now trusted as given (tiny clusters are only
folded away when the count is auto-detected) — with folding, a 4-second
reply from a second person was being merged into
the main speaker. If diarization fails, the transcript is still written,
without speaker labels.

## Speaker names from voice prints (`bin/voices.py`)

Ground truth: Google Meet caption files name every cue ("Asha: …")
and line up with the recordings at 0 s offset (checked against VAD).

**Embedding model** (`eval/speakers/bench_embeddings.py`): prints from the
first half of the Oct 10 call, nearest-print accuracy on held-out caption
segments, per person:

| model | Oct 10 2nd half (Mee/Rav/Ash) | Oct 9 (Rav/Ash) | Oct 5 (Rav/Ash) |
|---|---|---|---|
| **NeMo TitaNet-large** | 100/100/100 | 100/100 | **96**/99 |
| 3D-Speaker CAM++ zh-en | 100/100/100 | 100/100 | 92/99 |
| 3D-Speaker ERes2Net zh-cn (was the diarization model) | 100/100/100 | 100/100 | 80/100 |
| WeSpeaker ResNet-34 / 293 | 75–88 / 75–92 / 98–100 | 73–80 / 98–100 | 64–72 / 99–100 |

Score distributions (TitaNet, single segments): own print median 0.73;
another enrolled person ≤ 0.48 (p99); never-enrolled voices ≤ 0.48. Rule:
name a cluster if score ≥ 0.55 and ≥ 0.10 above the runner-up.

**End to end** (prints from the whole Oct 10 call; Oct 9 and Oct 5 never
used for enrolment), share of captioned speech carrying the right name
(`eval/speakers/score_names.py`):

| | Ravi | Asha |
|---|---|---|
| Oct 9 | 92.3 % | 99.6 % |
| Oct 5 | 95.2 % | 95.2 % |

Getting there needed the diarizer switched to TitaNet too: with the old
Mandarin-trained embedding, the Oct 5 call's clusters merged Ravi's release
walkthrough into Asha (Ravi 52.8 %). Auto-detect over-splits Asha into
2–3 clusters; naming merges them back.

**Unknown voices** (Sept 28 Marathi call: Ravi + two people never
enrolled): clusters of only the unknown people scored 0.50–0.52 and stayed
`S1`/`S2`; Ravi's speech was 100 % named. **Limit:** one diarization
cluster held 27 s of Ravi and 11 s of a guest, and took Ravi's name as
a whole — names are only as pure as the diarization clusters. That call is
Marathi with sparse captions; the our team calls above did not show it.

## Naming refinements (2026-10-12)

Measured on the same three calls (share of captioned speech with the right name):

| | before | after |
|---|---|---|
| Oct 9 — Ravi / Asha | 92.3 / 99.6 % | 93.9 / 99.0 % |
| Oct 5 — Ravi / Asha | 95.2 / 95.2 % | 95.9 / 95.4 % |
| Marathi call — Guest 1 wrongly named "Ravi" | 13 s | 13 s (voices only) → **2 s** with captions in the job |
| Marathi call — Guest 2 named | 0 % (unknown) | **97.6 %** with captions in the job |

- **Wide-margin rule**: a cluster scoring ≥ 0.45 with a ≥ 0.20 lead is named
  (one 4 s reply scored 0.54 vs 0.29); unknown voices lead by ≤ 0.02.
- **Per-segment re-check**: segments ≥ 2 s move to another enrolled person on
  a clear lead, and become `S?` when they score < 0.35 against their own
  cluster's name. Data: true-speaker segments scored ≥ 0.48 except one 0.24;
  guests/other people merged into a cluster scored 0.0–0.33.
- **Meet captions** in the same job name groups still unnamed (`S1`, `S?`).
  Tried and dropped: captions as line-by-line speaker turns — Meet credited a
  4 s reply to the person talking around it, where diarization and the voice
  print both heard a second speaker.

## Language routing: English / Hindi / Marathi in one file (2026-10-12)

Files can mix languages, so the default `auto` engine picks a model per
speech unit (VAD piece, ≤ 28 s). Scored on 361 units of six full recordings
(the Sept 28 Marathi call, the Oct 9 and Oct 5 Hinglish calls, two
Hinglish screen videos, the English Video 3), share of speech time:

| detector | Marathi call | Hinglish ×4 | English Video 3 |
|---|---|---|---|
| Tara's language token (en/hi/mr) | 0 % Marathi (Hindi 100 %) | Hindi 93–100 % | English 100 % (clips) |
| VoxLingua107 ECAPA, per unit | Marathi 94 % | Marathi 0–1 % | Marathi 39 % (Indian-accented English) |
| + unsure/short units vote with ±90 s | Marathi 99 % | 0 % | 31 % |
| + only if Tara's text is < 60 % Latin script | **Marathi 97 %** | **0 %** | **0 %** |

- Tara can't hear Marathi at all: its Hindi fine-tune gives the `mr` token ~0.
  It does separate English from Hindi, so it keeps that call.
- VoxLingua107 separates Hindi from Marathi but hears Indian-accented English
  as Marathi. Tara writes such units 78–100 % in Latin script, genuine Marathi
  units mostly < 60 %, hence the script check. The 3 % of the Marathi call it
  keeps with Tara are units that are mostly English words.
- Stitched test (Marathi, Hinglish, English, Marathi, Hinglish, 90 s each):
  both Marathi parts 100 % to IndicWhisper, nothing English or Hindi to it.
  Hindi/English files take exactly the Tara path as before; the check costs
  ~0.05 s per unit on the CPU.

Not covered: Marathi said entirely with English loanwords stays in Tara's
Hinglish style; Hindi and Marathi mixed inside one unit goes to whichever
dominates. A Marathi gold transcript would give real WER for both paths.

## Marathi decoding speed (2026-10-12)

IndicWhisper Marathi ran on PyTorch/MPS with beam search (5), 0.9x realtime.
On 30 lines (413 s of speech) of the Marathi call:

| decoding | speed | notes |
|---|---|---|
| PyTorch, beam 5 (was) | 0.9x realtime | drops whole phrases, e.g. a line's opening sentence |
| PyTorch, beam 3 / beam 2 | 1.4x / 2.1x | 31 % / 37 % different from beam 5 |
| greedy, whole unit (≤ 28 s) | 16–18x | keeps more speech (+10 % words), but invents text in the second half of long units |
| **MLX greedy on ≤ 12 s pieces** (now) | **12.5x** | keeps the openings, coherent to the end, highest model confidence (mean log-prob −0.22 vs −0.28 whole-unit) |

Pieces are cut at the quietest 100 ms in the middle half of a unit. Model
confidence alone didn't flag the invented text reliably, so there's no
beam-search fallback. Full 23-min call through `auto`: 1,744 s → 284 s
(4.8x realtime), 2,511 → 2,926 words, same language split (Marathi 96 %).
Tara's output is unchanged by the shared MLX code (12/12 lines identical).
No Marathi gold transcript yet, so this is read, not scored as WER.

## Next steps, in order of payoff

1. **A 10-minute gold transcript.** Hand-correct `eval/full/call1009-tara/transcript.txt`
   for 10 min (≈1 h of work, starting from Tara's output) → real WER/CER and
   speaker-error numbers, so future model swaps are decided on data.
2. ~~Speed: Tara on MLX~~ — done, 1.5× end to end (see above; the 3–5×
   estimate was wrong: decoding is memory-bandwidth-bound on either framework).
   ~~Overlap diarization with ASR~~ — done, 179 s → 125 s (see above).
3. ~~**Name the speakers.**~~ — done, see above. Keep one 10 s voice sample each of Asha, Ravi,
   Meera; match diarization clusters to them by embedding similarity
   → `Asha:` instead of `S1:`.
4. ~~**Better speaker embeddings.**~~ — done (TitaNet-large). The current one is trained on Mandarin
   (3D-Speaker). Try `wespeaker_en_voxceleb_resnet34` or NeMo TitaNet
   (same sherpa-onnx API) once (1) gives a way to score it.
5. **Meeting notes from the transcript.** A hand-written
   style summary (decisions, owners, deadlines, Trello cards) can be generated
   straight from `transcript.json` by Claude, now that lines carry speakers and
   timestamps.
6. **Cloud fallback for hard audio.** ElevenLabs Scribe v2 / Sarvam Saaras v3
   are on par with Tara on Hindi benchmarks and include diarization; worth an
   API key for calls where local output looks bad. Not set up (no keys here).
7. **Video understanding.** Frames + OCR + transcript are already aligned by
   timestamp; a vision-language pass (Claude on the frames, or Qwen3-VL locally)
   could caption what's on screen at each step of a walkthrough.
