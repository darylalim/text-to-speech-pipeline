# Kokoro Pipeline

Streamlit web app for generating multilingual speech using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), a lightweight text-to-speech model by [Hexgrad](https://github.com/hexgrad).

## Features

- 9 supported languages (American English, British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin Chinese)
- Voices discovered dynamically from HuggingFace Hub
- Voice comparison mode (up to 3 voices side by side)
- Adjustable speech speed (0.5x–2.0x)
- In-browser audio playback and WAV download
- Generation history with sidebar navigation and persistent output (max 20 entries)
- Generation metrics: model name, input characters, output duration, generation time

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng)

## Installation

```bash
uv sync --group dev
uv run streamlit run streamlit_app.py
```

> **Note:** The spaCy model `en_core_web_sm` (required for English G2P) is installed automatically via `uv sync`.

## Development

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`
