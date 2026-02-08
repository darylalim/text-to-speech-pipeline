# Text to Speech Pipeline

Streamlit web app for generating multilingual speech using [Bark Small](https://huggingface.co/suno/bark-small), a transformer-based text-to-audio model by [Suno](https://www.suno.ai/).

## Features

- 13 supported languages with configurable voice selection (language, gender, speaker)
- In-browser audio playback and WAV download
- Generation metrics: model name, input tokens, output duration, generation time

## Installation

```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Development

- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Typecheck**: `pyright`
- **Test**: `pytest`
