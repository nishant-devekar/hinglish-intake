---
name: decisions
description: List decisions made across the team's meetings, optionally filtered by topic, person or date range, each with its source line. Use for "what did we decide about…", "decision log", "what changed since…".
---

1. Read the `## Decisions` section of every `jobs/*/result/notes.md` in range
   (job ids start with YYYYMMDD; filter by that for dates).
2. For a topic or person filter, also grep the transcripts
   (`grep -ri "<term>" jobs/*/result/*/transcript.txt`) — notes can miss things.
3. Output newest first, grouped by meeting:
   `- <decision> — <who> [<file> @ mm:ss]` under `### <meeting title> (<date>)`.
4. Flag decisions that a later meeting reversed or changed, citing both.
