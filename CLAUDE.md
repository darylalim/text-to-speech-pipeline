# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for generating multilingual speech using [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox), a text-to-speech model by [Resemble AI](https://www.resemble.ai/).

## Setup

```bash
uv sync --group dev
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

**Runtime:** `chatterbox-tts`, `torch`, `torchaudio`, `numpy`, `streamlit`

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`).

## Architecture

### Files

- `streamlit_app.py` — single-file app with text input, language selection, voice cloning, style controls, and audio playback
- `tests/conftest.py` — mocks `streamlit` and `chatterbox` for import
- `tests/test_app.py` — unit tests

### Model

[Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (`ChatterboxMultilingualTTS` from `chatterbox-tts` package), 500M params.

### Supported Languages

ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh — 23 languages

### Performance

- Best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache model
- `time.perf_counter()` for timing

### UI

- Text input (max 300 characters)
- Language selection (23 languages via `SUPPORTED_LANGUAGES` from `chatterbox.mtl_tts`)
- Optional voice cloning via file upload (~10s WAV/MP3 reference audio)
- Style sliders: CFG Weight (0.0–1.0), Exaggeration (0.0–1.0)
- Generated audio displayed in browser player via `st.audio`
- WAV download via `st.download_button` (saved with `torchaudio.save`)
- Metrics via `st.metric`: model name, input characters, output duration, generation time
- Errors shown with `st.exception()`
- No session state for audio persistence

## Resources

- [GitHub Repo](https://github.com/resemble-ai/chatterbox)
- [Hugging Face](https://huggingface.co/ResembleAI/chatterbox)
- [PyPI](https://pypi.org/project/chatterbox-tts/)
