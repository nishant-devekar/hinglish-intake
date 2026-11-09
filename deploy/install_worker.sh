#!/usr/bin/env bash
# Run the processing worker in the background on this Mac, started at login
# and restarted if it crashes (launchd LaunchAgent).
#
#   deploy/install_worker.sh <org>/intake     install / update and start
#   deploy/install_worker.sh --remove         stop and uninstall
#
# Logs: ~/Library/Logs/hinglish-intake.log
# Needs (once, as you): `hf auth login` and `claude` -> /login.
set -euo pipefail

LABEL=com.hinglish-intake.worker
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$HOME/Library/Logs/hinglish-intake.log"

if [ "${1:-}" = "--remove" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"; echo "removed $LABEL"; exit 0
fi
[ $# -eq 1 ] || { echo "usage: $0 <org>/intake | --remove" >&2; exit 1; }

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string>
    <string>$ROOT/bin/worker.py</string>
    <string>hf://$1</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <!-- ffmpeg/whisper (brew), claude (/usr/local/bin), swift (/usr/bin) -->
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HF_HUB_DISABLE_PROGRESS_BARS</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
EOF
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "started $LABEL for $1 — log: $LOG"
