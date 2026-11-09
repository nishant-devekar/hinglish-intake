# How the demo was made

- `team-call.mp3` is a fictional 40-second call, synthesized with the macOS
  voices Lekha (Hindi) and Rishi (English, India). No real person's audio.
  Try it yourself: `bin/ingest.sh docs/team-call.mp3 --speakers 2`.
- The run was recorded with [vhs](https://github.com/charmbracelet/vhs)
  (`docs/demo.tape`) on a base M4 Mac (16 GB). The wait while it
  transcribes is shown at 6× speed; the real run took 28 s.
- The transcript frame is the unedited `transcript.txt` from that run,
  rendered in a browser with the terminal's colours, because terminal
  recorders don't shape Devanagari (vowel signs come apart). The errors are
  the model's: "नमस्ते" came out as "सस्ती", and "अच्छा" and "Perfect" were
  dropped.
