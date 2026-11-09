# Gold transcript (≈1 hour of work, once)

Every quality claim in `eval/REPORT.md` so far is a proxy (agreement between
models, speaker names vs Meet captions). One hand-corrected stretch turns
that into real error rates for every future change.

**Material** (not in git — company audio/text):

- `oct09-02m-12m.wav` — Oct 9 call, 02:00–12:00
- `oct09-draft.txt` — Tara's transcript of it, one line per segment, times from the full call

**To do:** copy the draft to `oct09-gold.txt`, play the wav, and fix each line:

- Write what was *said*: Hindi in Devanagari, English words in Latin script
  (as Tara does). Don't translate, don't tidy grammar; keep "right", "okay", repeats.
- Fix the speaker name after the time if it's wrong (`Asha`, `Ravi`, or `S1`…).
- Missing speech: add a line `[mm:ss] Name: text` at the right time.
- Can't make it out: put `[?]` in the line; those lines are left out of scoring.
- Leave times as they are unless a line is badly misplaced (±2 s doesn't matter).
  The wav starts at 02:00, so a line at `[05:30]` is 3:30 into the wav.

**Then score any transcript of the Oct 9 call:**

```bash
.venv/bin/python bin/transcribe.py out/call-2026-10-09/audio.wav --out /tmp/t
.venv/bin/python eval/gold/score_gold.py /tmp/t/transcript.json
```

Prints WER, CER (spelling variants like nukta/chandrabindu don't count) and
the share of speech with the right speaker name.
