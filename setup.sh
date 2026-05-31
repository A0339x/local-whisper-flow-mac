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
  --offline) EDITION=offline ;;
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

# ── Pick the edition ──────────────────────────────────────────────────────────
# Write/Command mode runs on Groq's gpt-oss-120b in the two main editions
# (sharper than the local 8B). They differ by DICTATION: on-device Whisper vs
# cloud Groq. Both need just ONE free Groq key. --offline is the zero-key, fully
# on-device option (local 8B Write).
if [ -z "$EDITION" ]; then
  if [ "$AUTO" = "1" ]; then
    EDITION=local
  else
    echo ""
    echo "Which edition do you want?  (both use one free Groq key — console.groq.com)"
    echo "  [1] Local  — dictation on-device (Whisper) + AI write via Groq (best privacy)."
    echo "               Your dictation never leaves the Mac; only Write drafts go to Groq."
    echo "               Recommended."
    echo "  [2] Cloud  — dictation AND write via Groq. Runs on ANY Apple-Silicon Mac,"
    echo "               no downloads. Lightest setup."
    echo "  (100% offline, no key, local 8B write — lower quality: re-run with --offline)"
    read -r -p "── Choose 1 or 2 [1]: " ch
    case "$ch" in 2) EDITION=cloud ;; *) EDITION=local ;; esac
  fi
fi
echo "── Edition: $EDITION"

mkdir -p "$HOME/.config/voice-to-text"
GROQ_FILE="$HOME/.config/voice-to-text/groq_key"
OPENAI_FILE="$HOME/.config/voice-to-text/openai_key"

if [ "$EDITION" = "cloud" ]; then
  # Groq dictation + Groq write (gpt-oss-120b). One key, no local models.
  collect_key Groq "gsk_…" console.groq.com "$GROQ_FILE"
  [ -s config.local.toml ] && echo "── config.local.toml exists — leaving it (delete to re-pick)." \
    || { cp config.cloud-full.toml config.local.toml; echo "── Enabled full-cloud edition (Groq)."; }

elif [ "$EDITION" = "offline" ]; then
  # Fully on-device: local Whisper + local 8B write. No keys. Needs Ollama.
  [ -f config.local.toml ] && { rm -f config.local.toml; echo "── Removed override (now 100% offline)."; }
  if ! command -v ollama >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then echo "── Installing Ollama…"; brew install ollama
    else echo "✋ Install Ollama from https://ollama.com/download , then re-run."; exit 1; fi
  fi
  curl -s http://localhost:11434/api/tags >/dev/null 2>&1 || { echo "── Starting Ollama…"; (ollama serve >/dev/null 2>&1 &); sleep 2; }
  MODEL=$(awk '/^\[/{s=$0} s=="[formatting]" && /^[[:space:]]*model[[:space:]]*=/{v=$0; sub(/^[^=]*=[[:space:]]*/,"",v); gsub(/"/,"",v); print v; exit}' config.toml)
  MODEL=${MODEL:-llama3.1:8b}
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then echo "── Write model present: $MODEL"
  else echo "── Pulling the Write model: $MODEL (one time)…"; ollama pull "$MODEL"; fi

else
  # Local (default): on-device Whisper dictation + Groq write (gpt-oss-120b).
  # One Groq key, no Ollama.
  EDITION=local
  collect_key Groq "gsk_…" console.groq.com "$GROQ_FILE"
  [ -s config.local.toml ] && echo "── config.local.toml exists — leaving it (delete to re-pick)." \
    || { cp config.cloud.toml config.local.toml; echo "── Enabled local-dictation + Groq-write edition."; }
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
  cloud)   echo "Cloud: no downloads — first dictation is instant. Only your dictations are"
           echo "uploaded. Re-pick anytime: rm config.local.toml && ./setup.sh" ;;
  offline) echo "Offline: 100% on-device, no keys. First dictation downloads Whisper (~3 GB)."
           echo "Switch to OpenAI write / cloud: ./setup.sh  (and pick 1 or 2)" ;;
  *)       echo "Local: dictation on-device (first one downloads Whisper ~3 GB); write via OpenAI."
           echo "Re-pick anytime: rm config.local.toml && ./setup.sh" ;;
esac
