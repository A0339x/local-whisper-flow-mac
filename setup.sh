#!/bin/bash
# One-command setup for Voice-To-Text on a fresh Mac.
# Run from inside the project folder:  ./setup.sh
# It ASKS which edition you want — no config editing required:
#   • Local  — 100% private, on-device (Whisper + Ollama). Needs model downloads.
#   • Cloud  — runs on any Apple-Silicon Mac, no downloads. Needs Groq + OpenAI keys.
set -e
cd "$(dirname "$0")"
PROJECT="$(pwd)"
echo "── Setting up Voice-To-Text in: $PROJECT"

# Flags: --local / --cloud pick the edition non-interactively; --yes / -y is
# unattended (defaults to local unless --cloud is also given).
AUTO=0; EDITION=""
for a in "$@"; do case "$a" in
  --yes|-y) AUTO=1 ;;
  --local) EDITION=local ;;
  --cloud) EDITION=cloud ;;
esac; done

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

# ── Pick the edition ──────────────────────────────────────────────────────────
if [ -z "$EDITION" ]; then
  if [ "$AUTO" = "1" ]; then
    EDITION=local
  else
    echo ""
    echo "Which edition do you want?"
    echo "  [1] Local  — 100% private, on-device. Downloads ~3 GB Whisper model + Ollama."
    echo "               Best on a capable Apple-Silicon Mac. Works offline, free."
    echo "  [2] Cloud  — runs on ANY Apple-Silicon Mac, no downloads, light + fast."
    echo "               Needs free Groq + OpenAI API keys. Only your dictations are sent."
    read -r -p "── Choose 1 or 2 [1]: " ch
    case "$ch" in 2) EDITION=cloud ;; *) EDITION=local ;; esac
  fi
fi
echo "── Edition: $EDITION"

if [ "$EDITION" = "cloud" ]; then
  # Cloud: collect keys, enable the full-cloud preset, skip all local models.
  mkdir -p "$HOME/.config/voice-to-text"
  GROQ_FILE="$HOME/.config/voice-to-text/groq_key"
  OPENAI_FILE="$HOME/.config/voice-to-text/openai_key"
  for spec in "Groq|gsk_…|console.groq.com|$GROQ_FILE" "OpenAI|sk-…|platform.openai.com|$OPENAI_FILE"; do
    IFS='|' read -r NAME HINT URL FILE <<<"$spec"
    if [ -s "$FILE" ]; then echo "── $NAME key already present."
    elif [ "$AUTO" = "1" ]; then echo "✋ No $NAME key. Put it at $FILE and re-run."; exit 1
    else
      echo "   Get a free $NAME key at $URL"
      read -r -p "── Paste your $NAME API key ($HINT): " K
      printf '%s' "$K" > "$FILE"; chmod 600 "$FILE"
    fi
  done
  if [ -s config.local.toml ]; then
    echo "── config.local.toml exists — leaving it (delete it to re-pick edition)."
  else
    cp config.cloud-full.toml config.local.toml
    echo "── Enabled full-cloud edition."
  fi
else
  # Local: Ollama + the Write-mode model. Remove any cloud override so the
  # 100%-local default in config.toml takes effect.
  [ -f config.local.toml ] && { rm -f config.local.toml; echo "── Removed cloud override (now 100% local)."; }
  if ! command -v ollama >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then echo "── Installing Ollama…"; brew install ollama
    else echo "✋ Install Ollama from https://ollama.com/download , then re-run ./setup.sh"; exit 1; fi
  fi
  if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "── Starting Ollama…"; (ollama serve >/dev/null 2>&1 &); sleep 2
  fi
  MODEL=$(awk '/^\[/{s=$0} s=="[formatting]" && /^[[:space:]]*model[[:space:]]*=/{v=$0; sub(/^[^=]*=[[:space:]]*/,"",v); gsub(/"/,"",v); print v; exit}' config.toml)
  MODEL=${MODEL:-llama3.1:8b}
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
    echo "── Write-mode model already present: $MODEL"
  else
    echo "── Pulling the Write-mode model: $MODEL (one time)…"; ollama pull "$MODEL"
  fi
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
if [ "$EDITION" = "cloud" ]; then
  echo "Cloud edition: no model download — first dictation is instant. Only your"
  echo "dictations are uploaded. Switch to local later: rm config.local.toml && ./setup.sh"
else
  echo "Local edition: first dictation downloads the Whisper model (~3 GB, one time)."
  echo "Switch to cloud later: ./setup.sh --cloud"
fi
