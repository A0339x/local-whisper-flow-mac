#!/bin/bash
# Build "Voice To Text.app" — a thin, menu-bar-only (LSUIElement) launcher
# bundle that runs flow.py via uv. Permissions (Mic / Accessibility / Input
# Monitoring) attach to this app once it's ad-hoc signed.
#
# Re-run this any time; it rebuilds the bundle. Your code lives in flow.py /
# config.toml in this folder — the app just points at them, so edits take
# effect on next launch (no rebuild needed for code/config changes).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$PROJECT_DIR/Voice To Text.app"
UV_BIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
OLLAMA_BIN="$(command -v ollama || echo /usr/local/bin/ollama)"

echo "Project : $PROJECT_DIR"
echo "uv      : $UV_BIN"
echo "ollama  : $OLLAMA_BIN"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# App icon (built once by gen_icon; kept in the project root).
if [ -f "$PROJECT_DIR/AppIcon.icns" ]; then
  cp "$PROJECT_DIR/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>            <string>Voice To Text</string>
    <key>CFBundleDisplayName</key>     <string>Voice To Text</string>
    <key>CFBundleIdentifier</key>      <string>com.local.voicetotext</string>
    <key>CFBundleVersion</key>         <string>0.1.0</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundlePackageType</key>     <string>APPL</string>
    <key>CFBundleExecutable</key>      <string>launcher</string>
    <key>CFBundleIconFile</key>        <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>  <string>13.0</string>
    <key>LSUIElement</key>             <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Voice To Text records your microphone to transcribe speech to text.</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/launcher" <<LAUNCH
#!/bin/bash
# This launcher runs on EVERY click and exits fast, so each click either starts
# the background dictation agent (first time) or just pops the Settings window.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
PROJECT="$PROJECT_DIR"
PYTHON="\$PROJECT/.venv/bin/python"
LOG="\$PROJECT/voice-to-text.log"

# Make sure Ollama is running (no-op if already up).
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  ("$OLLAMA_BIN" serve >> "\$LOG" 2>&1 &)
  sleep 1
fi

# Make sure the venv exists (first run / after dependency changes).
if [ ! -x "\$PYTHON" ]; then
  echo "Creating environment…" >> "\$LOG"
  "$UV_BIN" sync --project "\$PROJECT" >> "\$LOG" 2>&1
fi

if pgrep -f "\$PROJECT/flow.py" >/dev/null 2>&1; then
  # Agent already running → this click means "open Settings".
  touch "\$PROJECT/.show_settings"
else
  # Cold start (e.g. at login) → just start the agent silently, no Settings
  # popup. Launching the agent as a child of this LaunchServices-started app
  # gives it the GUI context its windows need. Its own Python keeps the
  # keystroke/mic permissions on THIS interpreter.
  echo "---- starting agent \$(date) ----" >> "\$LOG"
  nohup "\$PYTHON" "\$PROJECT/flow.py" >> "\$LOG" 2>&1 &
fi
LAUNCH
chmod +x "$APP/Contents/MacOS/launcher"

# Ad-hoc code signature → stable identity so TCC permissions persist.
codesign --force --deep --sign - "$APP"

echo "Built: $APP"
echo "Verifying signature…"
codesign --verify --verbose=2 "$APP" 2>&1 | sed 's/^/  /' || true
