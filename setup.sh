#!/bin/bash
# One-command setup for Voice-To-Text on a fresh Mac.
# Run this from inside the project folder:  ./setup.sh
set -e
cd "$(dirname "$0")"
PROJECT="$(pwd)"
echo "── Setting up Voice-To-Text in: $PROJECT"

# 1) Apple Silicon required (MLX Whisper is Apple-Silicon only).
if [ "$(uname -m)" != "arm64" ]; then
  echo "✋ This needs an Apple Silicon Mac (M1/M2/M3/M4/M5). MLX Whisper won't run on Intel."
  exit 1
fi

# 2) uv (Python toolchain)
if ! command -v uv >/dev/null 2>&1; then
  echo "── Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 3) Ollama (local LLM for smart formatting)
if ! command -v ollama >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "── Installing Ollama via Homebrew…"
    brew install ollama
  else
    echo "✋ Please install Ollama from https://ollama.com/download , then re-run ./setup.sh"
    exit 1
  fi
fi
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "── Starting Ollama…"
  (ollama serve >/dev/null 2>&1 &)
  sleep 2
fi
echo "── Pulling the formatting model (llama3.1:8b, ~5 GB, one time)…"
ollama pull llama3.1:8b

# 4) Python dependencies (creates .venv)
echo "── Installing Python dependencies…"
uv sync

# 5) Build the app bundle (paths adapt to THIS machine automatically)
echo "── Building the app…"
./build_app.sh >/dev/null

# 6) Optional: install to /Applications + start at login
read -r -p "── Install to /Applications and start at login? [y/N] " yn
if [[ "$yn" =~ ^[Yy] ]]; then
  rm -rf "/Applications/Voice To Text.app"
  cp -R "Voice To Text.app" "/Applications/Voice To Text.app"
  codesign --force --deep --sign - "/Applications/Voice To Text.app" >/dev/null 2>&1 || true
  osascript -e 'tell application "System Events" to delete (every login item whose name is "Voice To Text")' 2>/dev/null || true
  osascript -e 'tell application "System Events" to make login item at end with properties {path:"/Applications/Voice To Text.app", hidden:true}' >/dev/null
  open "/Applications/Voice To Text.app"
fi

cat <<'NEXT'

✅ Build complete!

ONE manual step left — grant macOS permissions (required for any dictation app):
  System Settings ▸ Privacy & Security, and add the app's "Python" to:
    • Microphone        (you'll also get a prompt on first use)
    • Accessibility     (to paste with ⌘V)
    • Input Monitoring  (to hear the Right Option hotkey)
  Then QUIT and relaunch the app (click "Voice To Text").

Use it:  tap Right Option, talk, tap Right Option again → text pastes.
First dictation downloads the Whisper model (~3 GB, one time). See README.md.
NEXT
