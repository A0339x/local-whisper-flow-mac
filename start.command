#!/bin/bash
# Double-click this file in Finder to launch Voice-To-Text.
cd "$(dirname "$0")" || exit 1
# Make sure Ollama is up (no-op if already running).
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama…"
  (ollama serve >/dev/null 2>&1 &)
  sleep 2
fi
echo "Launching Voice-To-Text… (look for 🎤 in your menu bar)"
exec uv run python flow.py
