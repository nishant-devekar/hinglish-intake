---
name: search
description: Find where something was said across all the team's meetings and recordings — a topic, feature, customer, number or phrase — and quote it with file and time. Use for "when did we talk about…", "who mentioned…", "find the part where…".
---

1. Search transcripts and on-screen text, case-insensitive, in both scripts where it matters
   (e.g. "WhatsApp" and "व्हाट्सएप"; "onboarding" and "ऑनबोर्डिंग"):
   `grep -rin "<term>" jobs/*/result/*/transcript.txt jobs/*/result/*/timeline.md`
2. For each hit, read a few lines around it so the quote makes sense.
3. Answer with the matches grouped by meeting (title + date from job.json):
   `[<file> @ mm:ss] Speaker: "<quote>"` plus a one-line English gloss for Hindi lines.
4. For a video, find the hit's scene in timeline.md and open its screenshot (`frames/frame-NNNN.jpg`)
   to say what was on screen at that moment.
