# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for generating multilingual speech using [Bark Small](https://huggingface.co/suno/bark-small), a transformer-based text-to-audio model by [Suno](https://www.suno.ai/). Bark can also generate music, background noise, sound effects, and produce nonverbal communications like laughing, sighing and crying.

## Setup

```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Commands

- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Typecheck**: `pyright`
- **Test**: `pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- `VoicePreset` TypedDict for voice preset structure
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `transformers` - Hugging Face model loading
- `torch` - Tensor operations
- `accelerate` - Inference optimization
- `numpy` - Array operations
- `scipy` - Audio file processing
- `streamlit` - Web user interface

## Configuration

`pyproject.toml` — ruff isort (`combine-as-imports`), pytest (`pythonpath`), and pyright (`pythonVersion = "3.12"`).

## Architecture

### Entry Point

`streamlit_app.py` - single-file app.

### Tests

`tests/` directory with `conftest.py` (mocks streamlit and transformers for import) and `test_app.py`.

### Model

Uses [Bark Small](https://huggingface.co/suno/bark-small) (`suno/bark-small`) exclusively for local memory constraints.

### Supported Languages

- English (en)
- German (de)
- Spanish (es)
- French (fr)
- Hindi (hi)
- Italian (it)
- Japanese (ja)
- Korean (ko)
- Polish (pl)
- Portuguese (pt)
- Russian (ru)
- Turkish (tr)
- Chinese, simplified (zh)

[Voice prompt library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)

### Performance

- Use best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache model
- Use float16 precision for `dtype`
- `torch.inference_mode()` for generation
- Text input with voice selection (language → gender → speaker)
- Voice presets map languages to speaker options in format `v2/{language_code}_speaker_{N}`
- Generated audio displayed in browser player
- Do not use session state to persist audio
- `time.perf_counter()` for timing (fractional seconds)

### Error Handling

- Unexpected exceptions shown with `st.exception()` for debugging

### Audio Download

Audio file containing synthesized text-to-speech output downloadable via `st.download_button`.

Response headers:

- `model` (string) - model name
- `prompt_eval_count` (integer) - number of input tokens in the prompt
- `output_duration` (float) - generated audio duration in seconds
- `eval_duration` (float) - text-to-speech generation time in seconds (rounded to 2 decimal places)

### Metrics

`st.metric` displays all response headers.

## Bark Gotchas

- **Audio dtype**: Model outputs float16 tensors when loaded with `dtype=torch.float16`. Convert to float32 with `.float()` before `.numpy()` — `scipy.io.wavfile.write` does not support float16.
- **Sub-config access**: `model.generation_config.semantic_config` is a dict, not an object. Use dict access (`.get()`, `[]`), not attribute access.
- **pad_token_id**: Set on `generation_config.semantic_config` (dict), not on the top-level `generation_config`. The top-level setting has no effect on Bark's semantic sub-model. Must NOT equal `eos_token_id` — use `eos_token_id + 1` (10001, outside codebook range 0–10000). Setting them equal triggers an "attention mask not set" error because the model can't infer padding from input when the two tokens are identical.
- **Device transfer**: Use `inputs.to(device)` (BatchFeature method), not a dict comprehension. BatchFeature.to() handles nested structures like `history_prompt` and preserves `attention_mask` forwarding.

## Resources

- [GitHub Repo](https://github.com/suno-ai/bark)
- [Bark Colab](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)
- [Hugging Face Colab](https://colab.research.google.com/drive/1dWWkZzvu7L9Bunq9zvD-W02RFUXoW-Pd?usp=sharing)
- [Hugging Face Demo](https://huggingface.co/spaces/suno/bark)
- [Bark docs](https://huggingface.co/docs/transformers/model_doc/bark)
