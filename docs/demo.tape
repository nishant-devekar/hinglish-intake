Output demo-run.mp4
Set Shell "bash"
Set FontSize 15
Set Width 1100
Set Height 420
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Hide
Type "export PS1='$ ' && clear" Enter
Sleep 300ms
Show

Type "# a 40 s Hinglish call, two speakers, all on this Mac" Enter
Sleep 600ms
Type "bin/ingest.sh team-call.mp3 --speakers 2" Enter
Wait+Screen@240s /speech:/
Sleep 2s


