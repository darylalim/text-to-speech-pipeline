# Text to Speech Pipeline

Streamlit web app for generating multilingual speech using [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox), a text-to-speech model by [Resemble AI](https://www.resemble.ai/).

## Features

- 23 supported languages
- Optional voice cloning from a ~10s reference audio clip
- Adjustable speech style (CFG weight and exaggeration)
- In-browser audio playback and WAV download
- Generation metrics: model name, input characters, output duration, generation time

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv sync --group dev
uv run streamlit run streamlit_app.py
```

## Development

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`
