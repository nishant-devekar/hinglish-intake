---
name: actions
description: Action items from the team's meetings — everyone's, or one person's (e.g. "/actions Ravi") — with due dates and sources. Use for "what's on my plate", "what did X promise", "open action items".
---

1. Collect the `## Action items` tables from `jobs/*/result/notes.md`.
2. Filter by owner if a name is given (match first names; `S1`/`S2` are unknown people).
3. Check each against later meetings' transcripts for signs it was done or dropped
   ("done", "ho gaya", "release de diya", the item's keywords) and mark
   ✅ done / ⏳ open / ❓ unclear with the evidence line.
4. Output a table: Owner | Action | Due | Status | Source. Oldest open items first.
