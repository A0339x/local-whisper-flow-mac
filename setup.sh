#!/bin/bash
# One-command setup for Voice-To-Text on a fresh Mac.
# Run from inside the project folder:  ./setup.sh
# It ASKS which mode you want — no config editing required:
#   • Offline — 100% private, on-device (Whisper + Ollama). Needs model downloads.
#   • Online  — AssemblyAI streaming dictation + Groq writing. Runs on any
#              Apple-Silicon Mac, no downloads. Needs a free AssemblyAI + Groq key.
set -e
cd "$(dirname "$0")"
PROJECT="$(pwd)"
echo "── Setting up Voice-To-Text in: $PROJECT"

# Flags: --offline / --online pick the mode non-interactively; --yes / -y is
# unattended (defaults to OFFLINE — private, no key).
AUTO=0; EDITION=""
for a in "$@"; do case "$a" in
  --yes|-y) AUTO=1 ;;
  --offline) EDITION=offline ;;
  --online|--cloud) EDITION=online ;;
esac; done

# Collect an API key into a 0600 file (skip if already present).
collect_key() {  # NAME HINT URL FILE
  if [ -s "$4" ]; then echo "── $1 key already present."
  elif [ "$AUTO" = "1" ]; then echo "✋ No $1 key — put it at $4 and re-run."; exit 1
  else
    echo "   Get a free $1 key at $3"
    read -r -p "── Paste your $1 API key ($2): " _K
    printf '%s' "$_K" > "$4"; chmod 600 "$4"
  fi
}

# Apple Silicon required (macOS UI stack; local also needs MLX). Cloud needs NO
# powerful machine — a base M1 Air is plenty since the heavy models run remotely.
if [ "$(uname -m)" != "arm64" ]; then
  echo "✋ This needs an Apple-Silicon Mac (M1 or newer)."
  exit 1
fi

# uv (Python toolchain) — needed by both editions.
if ! command -v uv >/dev/null 2>&1; then
  echo "── Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ── Pick the mode (OFFLINE is the default — private, no key) ───────────────────
if [ -z "$EDITION" ]; then
  if [ "$AUTO" = "1" ]; then
    EDITION=offline
  else
    echo ""
    echo "Which mode do you want?"
    echo "  [1] Offline  — 100% on your Mac. Private, NO API key, works without"
    echo "                 internet. Dictation + AI writing run on-device.  (default)"
    echo "  [2] Online   — AssemblyAI streaming dictation + Groq AI writing. Fast on"
    echo "                 any Mac, no downloads. Needs a free AssemblyAI + Groq key."
    echo "  You can switch anytime later in the app (menu ▸ Offline mode)."
    read -r -p "── Choose 1 or 2 [1]: " ch
    case "$ch" in 2) EDITION=online ;; *) EDITION=offline ;; esac
  fi
fi
echo "── Mode: $EDITION"

mkdir -p "$HOME/.config/voice-to-text"
GROQ_FILE="$HOME/.config/voice-to-text/groq_key"
AAI_FILE="$HOME/.config/voice-to-text/assemblyai_key"

if [ "$EDITION" = "online" ]; then
  # AssemblyAI streaming dictation + Groq gpt-oss-120b writing. Two keys, no local models.
  collect_key AssemblyAI "your key" assemblyai.com "$AAI_FILE"
  collect_key Groq "gsk_…" console.groq.com "$GROQ_FILE"
  cp config.assemblyai.toml config.local.toml
  echo "── Enabled Online (AssemblyAI dictation + Groq writing) mode."
else
  # Offline (default): 100% on-device — local Whisper + local gpt-oss:20b write.
  EDITION=offline
  rm -f config.local.toml          # use the 100%-local base in config.toml
  echo "── Enabled Offline (on-device) mode."
  if ! command -v ollama >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then echo "── Installing Ollama…"; brew install ollama
    else echo "✋ Install Ollama from https://ollama.com/download , then re-run."; exit 1; fi
  fi
  curl -s http://localhost:11434/api/tags >/dev/null 2>&1 || { echo "── Starting Ollama…"; (ollama serve >/dev/null 2>&1 &); sleep 2; }
  MODEL=$(awk '/^\[/{s=$0} s=="[formatting]" && /^[[:space:]]*model[[:space:]]*=/{v=$0; sub(/^[^=]*=[[:space:]]*/,"",v); gsub(/"/,"",v); print v; exit}' config.toml)
  MODEL=${MODEL:-gpt-oss:20b}
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then echo "── On-device Write model present: $MODEL"
  else echo "── Pulling the on-device Write model: $MODEL (one time, ~13GB)…"; ollama pull "$MODEL"; fi
fi

# ── Common: deps, build, install ──────────────────────────────────────────────
echo "── Installing Python dependencies…"
uv sync
echo "── Building the app…"
./build_app.sh >/dev/null

if [ "$AUTO" = "1" ]; then yn="y"; else
  read -r -p "── Install to /Applications and start at login? [y/N] " yn
fi
if [[ "$yn" =~ ^[Yy] ]]; then
  rm -rf "/Applications/Voice To Text.app"
  cp -R "Voice To Text.app" "/Applications/Voice To Text.app"
  codesign --force --deep --sign - "/Applications/Voice To Text.app" >/dev/null 2>&1 || true
  osascript -e 'tell application "System Events" to delete (every login item whose name is "Voice To Text")' 2>/dev/null || true
  osascript -e 'tell application "System Events" to make login item at end with properties {path:"/Applications/Voice To Text.app", hidden:true}' >/dev/null
  open "/Applications/Voice To Text.app"
fi

echo ""
echo "✅ Build complete — edition: $EDITION"
echo ""
echo "ONE manual step left — grant macOS permissions (required for any dictation app):"
echo "  System Settings ▸ Privacy & Security, add the app's \"Python\" to:"
echo "    • Microphone        (you'll also get a prompt on first use)"
echo "    • Accessibility     (to paste with ⌘V)"
echo "    • Input Monitoring  (to hear the Right Option hotkey)"
echo "  Then QUIT and relaunch the app (click \"Voice To Text\")."
echo ""
echo "Use it:  Right Option → dictate.   Left Option → AI edit/write."
case "$EDITION" in
  online)  echo "🔵 Online: no downloads — dictation streams via AssemblyAI (~40ms), AI writing"
           echo "via Groq. Only what you dictate is sent. Switch to offline anytime (menu ▸ Offline mode)." ;;
  *)       echo "🟢 Offline: 100% on-device, no keys, no internet. First dictation downloads the"
           echo "Whisper model (~3 GB, once). Switch to online anytime (menu ▸ Offline mode)." ;;
esac
