# Chatterbox-Multilingual Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Bark Small with Chatterbox-Multilingual as the TTS engine, adding optional voice cloning and style control sliders.

**Architecture:** Single-file Streamlit app using `ChatterboxMultilingualTTS` from the `chatterbox-tts` package. Model loaded once via `@st.cache_resource`, generates audio via `model.generate(text, language_id=...)`. Optional voice cloning from uploaded audio. Style tuning via `cfg_weight` and `exaggeration` sliders.

**Tech Stack:** `chatterbox-tts`, `torchaudio`, `torch`, `numpy`, `streamlit`

---

### Task 1: Update dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Replace the dependencies and description:

```toml
[project]
name = "text-to-speech-pipeline"
version = "0.2.0"
description = "Streamlit web app for multilingual text-to-speech using Chatterbox"
requires-python = ">=3.12"
dependencies = [
    "chatterbox-tts",
    "numpy",
    "streamlit",
    "torch",
    "torchaudio",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "ty",
]

[tool.ruff.lint.isort]
combine-as-imports = true

[tool.pytest.ini_options]
pythonpath = ["."]

[tool.ty.environment]
python-version = "3.12"
```

Removed: `transformers`, `accelerate`, `scipy`
Added: `chatterbox-tts`, `torchaudio`

**Step 2: Sync dependencies**

Run: `uv sync --group dev`
Expected: resolves and installs `chatterbox-tts` and `torchaudio`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: replace Bark deps with chatterbox-tts and torchaudio"
```

---

### Task 2: Delete voice_presets.json

Chatterbox-Multilingual uses `language_id` codes and optional voice cloning — no numbered speaker presets.

**Files:**
- Delete: `voice_presets.json`

**Step 1: Delete the file**

```bash
git rm voice_presets.json
```

**Step 2: Commit**

```bash
git commit -m "chore: remove voice_presets.json (replaced by Chatterbox language_id)"
```

---

### Task 3: Rewrite streamlit_app.py

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Write the new app**

Replace the entire file with:

```python
import io
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchaudio

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS

MODEL_NAME = "Chatterbox Multilingual"

LANGUAGES: dict[str, str] = {name: code for code, name in SUPPORTED_LANGUAGES.items()}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(device: str) -> ChatterboxMultilingualTTS:
    return ChatterboxMultilingualTTS.from_pretrained(device=device)


def generate_speech(
    text: str,
    language_id: str,
    model: ChatterboxMultilingualTTS,
    audio_prompt_path: str | None = None,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
) -> tuple[int, np.ndarray]:
    wav = model.generate(
        text,
        language_id=language_id,
        audio_prompt_path=audio_prompt_path,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
    )
    return model.sr, wav.squeeze(0).numpy()


st.title("Text to Speech Pipeline")
st.write("Generate multilingual speech with Chatterbox by Resemble AI.")

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model = load_model(device)

st.subheader("Text")
text_input = st.text_area(
    "Text",
    placeholder="Enter text...",
    max_chars=300,
    height=150,
    help="Maximum 300 characters per generation.",
)

st.subheader("Voice")
voice_col1, voice_col2 = st.columns(2)

with voice_col1:
    language = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        help="Select a language for speech generation.",
    )

language_id = LANGUAGES[language]

with voice_col2:
    audio_file = st.file_uploader(
        "Reference Audio (optional)",
        type=["wav", "mp3"],
        help="Upload a ~10s audio clip to clone a voice. Leave empty for default voice.",
    )

st.subheader("Style")
style_col1, style_col2 = st.columns(2)

with style_col1:
    cfg_weight = st.slider(
        "CFG Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Classifier-free guidance strength. Lower values may improve pacing for fast speakers.",
    )

with style_col2:
    exaggeration = st.slider(
        "Exaggeration",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Speech expressiveness intensity. Higher values produce more expressive speech.",
    )

if st.button("Generate", type="primary"):
    if text_input.strip():
        try:
            audio_prompt_path: str | None = None
            if audio_file is not None:
                suffix = Path(audio_file.name).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(audio_file.read())
                    audio_prompt_path = tmp.name

            with st.spinner("Generating speech..."):
                start = time.perf_counter()
                sampling_rate, audio_array = generate_speech(
                    text_input,
                    language_id,
                    model,
                    audio_prompt_path=audio_prompt_path,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                )
                eval_duration = round(time.perf_counter() - start, 2)
                output_duration = len(audio_array) / sampling_rate

            st.audio(audio_array, sample_rate=sampling_rate)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model", "Chatterbox")
            col2.metric("Input Characters", len(text_input))
            col3.metric("Output Duration", f"{output_duration:.2f}s")
            col4.metric("Generation Time", f"{eval_duration}s")

            wav_buffer = io.BytesIO()
            torchaudio.save(
                wav_buffer,
                torch.tensor(audio_array).unsqueeze(0),
                sampling_rate,
                format="wav",
            )
            st.download_button(
                label="Download Audio",
                data=wav_buffer.getvalue(),
                file_name="speech.wav",
                mime="audio/wav",
            )

        except Exception as e:
            st.exception(e)
    else:
        st.warning("Enter text.")
```

**Step 2: Verify the app launches**

Run: `uv run streamlit run streamlit_app.py`
Expected: app loads without errors, model downloads on first run

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: replace Bark with Chatterbox-Multilingual TTS engine

Add 23-language support, optional voice cloning from uploaded audio,
and cfg_weight/exaggeration style sliders."
```

---

### Task 4: Rewrite test conftest

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Write the new conftest**

Replace the entire file with:

```python
import sys
from unittest.mock import MagicMock

# Mock streamlit to prevent UI initialization on import
_st = MagicMock()
_st.cache_resource = lambda f: f
_st.selectbox.side_effect = lambda label, **_kw: {
    "Language": "English",
}.get(label, MagicMock())
_st.slider.side_effect = lambda label, **_kw: {
    "CFG Weight": 0.5,
    "Exaggeration": 0.5,
}.get(label, MagicMock())
_st.button.return_value = False
_st.text_area.return_value = ""
_st.file_uploader.return_value = None
_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
sys.modules["streamlit"] = _st

# Mock chatterbox to prevent model downloads on import
_chatterbox = MagicMock()
sys.modules["chatterbox"] = _chatterbox
sys.modules["chatterbox.mtl_tts"] = _chatterbox.mtl_tts
_chatterbox.mtl_tts.SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}
_chatterbox.mtl_tts.ChatterboxMultilingualTTS = MagicMock()
```

**Step 2: Verify conftest loads**

Run: `uv run python -c "import tests.conftest; print('OK')"`
Expected: prints "OK"

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: update conftest mocks for chatterbox"
```

---

### Task 5: Rewrite tests

**Files:**
- Modify: `tests/test_app.py`

**Step 1: Write the failing tests**

Replace the entire file with:

```python
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from streamlit_app import (
    LANGUAGES,
    MODEL_NAME,
    ChatterboxMultilingualTTS,
    generate_speech,
    get_device,
    load_model,
)

EXPECTED_LANGUAGES = [
    "Arabic",
    "Chinese",
    "Danish",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Malay",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Turkish",
]

EXPECTED_CODES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
}


class TestLanguages:
    def test_all_languages_present(self) -> None:
        assert sorted(LANGUAGES.keys()) == EXPECTED_LANGUAGES

    def test_language_codes(self) -> None:
        codes = set(LANGUAGES.values())
        assert codes == EXPECTED_CODES

    def test_language_count(self) -> None:
        assert len(LANGUAGES) == 23


class TestModelName:
    def test_model_name(self) -> None:
        assert MODEL_NAME == "Chatterbox Multilingual"


class TestGetDevice:
    def test_mps_preferred(self) -> None:
        with patch("torch.backends.mps.is_available", return_value=True):
            assert get_device() == "mps"

    def test_cuda_fallback(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            assert get_device() == "cuda"

    def test_cpu_fallback(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            assert get_device() == "cpu"


class TestLoadModel:
    def test_calls_from_pretrained_with_device(self) -> None:
        load_model("cpu")
        ChatterboxMultilingualTTS.from_pretrained.assert_called_with(device="cpu")

    def test_returns_model(self) -> None:
        model = load_model("cpu")
        assert model is ChatterboxMultilingualTTS.from_pretrained.return_value


class TestGenerateSpeech:
    def _mock_model(
        self, *, sample_rate: int = 24000, audio_length: int = 48000
    ) -> MagicMock:
        model = MagicMock()
        model.sr = sample_rate
        model.generate.return_value = torch.randn(1, audio_length)
        return model

    def test_returns_sampling_rate_and_audio(self) -> None:
        model = self._mock_model()

        sampling_rate, audio = generate_speech("hello", "en", model)

        assert sampling_rate == 24000
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (48000,)

    def test_calls_generate_with_correct_args(self) -> None:
        model = self._mock_model()

        generate_speech(
            "test text",
            "fr",
            model,
            audio_prompt_path="/tmp/ref.wav",
            cfg_weight=0.3,
            exaggeration=0.7,
        )

        model.generate.assert_called_once_with(
            "test text",
            language_id="fr",
            audio_prompt_path="/tmp/ref.wav",
            cfg_weight=0.3,
            exaggeration=0.7,
        )

    def test_default_parameters(self) -> None:
        model = self._mock_model()

        generate_speech("test", "en", model)

        model.generate.assert_called_once_with(
            "test",
            language_id="en",
            audio_prompt_path=None,
            cfg_weight=0.5,
            exaggeration=0.5,
        )

    def test_output_is_float32_numpy(self) -> None:
        model = self._mock_model()
        model.generate.return_value = torch.randn(1, 100, dtype=torch.float16)

        _, audio = generate_speech("test", "en", model)

        assert audio.dtype == np.float32
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py -v`
Expected: all tests pass

**Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test: rewrite tests for Chatterbox-Multilingual API"
```

---

### Task 6: Run lint, format, and typecheck

**Files:** All modified files

**Step 1: Format**

Run: `uv run ruff format .`

**Step 2: Lint**

Run: `uv run ruff check .`
Expected: no errors

**Step 3: Typecheck**

Run: `uv run ty check`
Expected: no errors (may have chatterbox stub warnings — acceptable)

**Step 4: Fix any issues found, then commit**

```bash
git add -u
git commit -m "style: fix lint and formatting"
```

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Key changes:
- Project description: "Chatterbox Multilingual" instead of "Bark Small"
- Dependencies: replace `transformers`, `accelerate`, `scipy` with `chatterbox-tts`, `torchaudio`
- Architecture section: update model info, remove voice_presets.json, update file descriptions
- Remove Bark Gotchas section entirely
- Update Supported Languages to list all 23
- Update UI section: language selector, optional voice cloning uploader, style sliders
- Update Performance section: remove float16 mention, update cache description
- Update Resources section with Chatterbox links

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Chatterbox migration"
```

---

### Task 8: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update README.md**

```markdown
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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for Chatterbox migration"
```
