# Text to Speech Pipeline

Streamlit web app for generating multilingual speech using [Bark Small](https://huggingface.co/suno/bark-small), a transformer-based text-to-audio model by [Suno](https://www.suno.ai/).

## Features

- 13 supported languages with configurable voice selection (language, gender, speaker)
- In-browser audio playback and WAV download
- Generation metrics: model name, input tokens, output duration, generation time

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
