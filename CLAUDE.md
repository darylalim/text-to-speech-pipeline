# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for generating multilingual speech using [Bark Small](https://huggingface.co/suno/bark-small), a transformer-based text-to-audio model by [Suno](https://www.suno.ai/).

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
- `VoicePreset` TypedDict for voice preset structure
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

**Runtime:** `transformers`, `torch`, `accelerate`, `numpy`, `scipy`, `streamlit`

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`).

## Architecture

### Files

- `streamlit_app.py` — single-file app
- `voice_presets.json` — voice preset data (language, gender, speaker mappings)
- `tests/conftest.py` — mocks `streamlit` and `transformers` for import
- `tests/test_app.py` — unit tests

### Model

[Bark Small](https://huggingface.co/suno/bark-small) (`suno/bark-small`) exclusively, for local memory constraints.

### Supported Languages

en, de, es, fr, hi, it, ja, ko, pl, pt, ru, tr, zh — [voice prompt library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)

### Performance

- Best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache model and processor
- float16 precision for `dtype`
- `torch.inference_mode()` for generation
- `time.perf_counter()` for timing

### UI

- Text input with voice selection (language, gender, speaker)
- Voice presets loaded from `voice_presets.json` in format `v2/{language_code}_speaker_{N}`
- Generated audio displayed in browser player via `st.audio`
- WAV download via `st.download_button`
- Metrics via `st.metric`: model name, input tokens, output duration, generation time
- Errors shown with `st.exception()`
- No session state for audio persistence

## Bark Gotchas

- **Audio dtype** — float16 tensors must be converted to float32 (`.float()`) before `.numpy()`. `scipy.io.wavfile.write` rejects float16.
- **semantic_config** — `model.generation_config.semantic_config` is a dict, not an object. Use `[]` / `.get()`, not attribute access.
- **pad_token_id** — Set on `semantic_config` (dict), not top-level `generation_config`. Must not equal `eos_token_id` — use `eos_token_id + 1` (10001). Equal values trigger "attention mask not set" error.
- **tie_word_embeddings** — Set `model.config.tie_word_embeddings = False` after loading. Do not pass as kwarg to `from_pretrained()` (raises `TypeError`).
- **do_sample** — Set on `model.generation_config`, not as kwarg to `generate()`. Mixing config with kwargs is deprecated.
- **max_length** — Delete `model.generation_config.max_length` after loading. Bark sets both `max_length` (20) and `max_new_tokens` (768), which conflict. Only `max_new_tokens` should be used.
- **Device transfer** — Use `inputs.to(device)` (BatchFeature method), not dict comprehension. Handles nested structures and preserves `attention_mask` forwarding.

## Resources

- [GitHub Repo](https://github.com/suno-ai/bark)
- [Bark Colab](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)
- [Hugging Face Colab](https://colab.research.google.com/drive/1dWWkZzvu7L9Bunq9zvD-W02RFUXoW-Pd?usp=sharing)
- [Hugging Face Demo](https://huggingface.co/spaces/suno/bark)
- [Bark docs](https://huggingface.co/docs/transformers/model_doc/bark)
