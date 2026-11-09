---
name: meeting-notes
description: Summarise a team meeting from this workspace — the latest job, or one named by title, date or id. Use when asked "what happened in…", "notes for…", or to (re)write notes for a job whose notes.md is pending.
---

1. Find the job: `ls jobs` (ids start with the date), match on `job.json` title/files.
   With no hint, use the newest job whose `job.json` status is `done`.
2. If `result/notes.md` exists and isn't "Notes pending", present it — but spot-check
   2-3 decisions/action items against the transcript lines they cite and fix anything wrong.
3. Otherwise write notes from `result/*/transcript.txt` with these sections: Summary,
   Decisions, Action items (Owner | Action | Due | Source), Open questions, Topics (mm:ss list).
   Every bullet ends with `[<file> @ mm:ss]`. Keep `S1`/`S2` labels; don't guess names.
4. If asked to save, write to `result/notes.md` (local copy only — tell the user it isn't
   synced back to the team).
