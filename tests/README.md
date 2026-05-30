# Tests

Integration tests for the Voice-To-Text pipeline. They use real models and
macOS `say` as a synthetic voice, so they require:

- the app's environment (`uv sync` has been run)
- **Ollama running** with the configured formatting model pulled
- the Whisper model cached (run one dictation, or it downloads on first test)

Run:

```bash
uv run python tests/test_pipeline.py
```

Covers formatter behavior, prompt-injection resistance, replacements, paragraph
and spacing logic, the excitement detector (warm-up, adaptation, recovery), and
end-to-end audio. Asserts properties (the LLM is non-deterministic) plus
deterministic unit checks.
