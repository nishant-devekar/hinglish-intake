# Team meetings workspace

Everything the team uploaded to Hinglish Intake, processed: calls, screen
recordings, voice notes and transcripts. Use it to answer questions about
what was discussed, decided and promised.

## Layout

```
jobs/<id>/job.json                 title, who uploaded, when, original file names
jobs/<id>/result/README.md         the job: who/when/options, each file's media details,
                                   speakers (talk time, share, words), languages, who wrote
                                   the notes, the processing log
jobs/<id>/result/job.json          the job record with its full log (also in the downloaded zip)
jobs/<id>/result/notes.md          summary, decisions, action items, open questions, Trello cards
jobs/<id>/result/<file>/transcript.txt   [mm:ss] Speaker: text
jobs/<id>/result/<file>/transcript.json  same, with exact start/end seconds and each line's language
jobs/<id>/result/<file>/meta.json        media details (size, codecs, resolution, fps, recording date),
                                         speaker stats, language mix, that file's processing log
jobs/<id>/result/<file>/audio.m4a        the recording's sound, for checking a quote by ear
jobs/<id>/result/<file>/timeline.md     videos: who spoke how long, then scene by scene — screenshot, all on-screen text, what was said
jobs/<id>/result/<file>/timeline.json   videos: the same, as data (scene -> frame, span, OCR lines, segments)
jobs/<id>/result/<file>/evidence.pdf    videos: the same timeline with the screenshots embedded
jobs/<id>/result/<file>/frames/    one screenshot per scene; index.txt = frame -> time span; ocr.txt = on-screen text
jobs/<id>/result/bundle.md         all of the above in one file
```

Job ids start with the upload date (`20261012-...`), so `ls jobs` is chronological.

## Rules

- **Cite everything** as `[<original file> @ mm:ss]` (the transcript line's time).
  If you can't point to a line, say you couldn't find it.
- `notes.md` is a machine-written summary; for anything that matters (a
  decision, a number, a deadline) check the transcript line it cites.
- Speech is English, Hindi-English and/or Marathi, each stretch transcribed by
  the model for its language (`transcript.json` has each line's `lang`: en, hi,
  mr). Hinglish lines: English words in Latin script, Hindi in Devanagari.
  Marathi lines are all Devanagari, English loanwords too. Answer in English
  unless asked otherwise; quote Hindi or Marathi as said when exact wording matters.
- Speakers are named only when their voice is enrolled (Asha, Ravi,
  Meera so far) or Meet captions named them. `S1`, `S2` are unknown people;
  `S?` is a line whose voice didn't match the name its speaker group had, so
  it was left unnamed. Never guess who any of these are.
- A `notes.md` that says "Notes pending" means automatic notes failed; write
  them yourself from the transcript if asked (use /meeting-notes).
- A `notes.md` that starts with "Draft by the local fallback model" was
  written by a smaller model on the processing Mac while Claude was
  unavailable; it is replaced automatically later. Check owners, numbers and
  decisions in it against the transcript before relying on them.
- For videos, start from `timeline.md`: it says which screenshot was showing
  while each line was spoken. Open the linked `frames/frame-NNNN.jpg` to see
  the screen (whiteboards, app screens, slides) before describing it; the
  on-screen text there is OCR and is noisy for handwriting.

## Getting new jobs

```bash
hf download <org>/intake --repo-type dataset --local-dir . \
   --include "CLAUDE.md" ".claude/*" "jobs/*/job.json" "jobs/*/result/*"
```
(The exact repo name is in the Intake app's "Use with Claude" tab.)
