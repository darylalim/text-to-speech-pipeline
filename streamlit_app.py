import io
import tempfile
import time
import warnings
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Generator

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.attention as _torch_attention
import scipy.io.wavfile as wavfile

import diffusers.models.lora as _diffusers_lora

# Patch: chatterbox-tts pins diffusers 0.29 which exposes deprecated LoRACompatibleLinear.
# Remove this patch (and the peft dependency) when chatterbox upgrades diffusers.
_diffusers_lora.LoRACompatibleLinear = nn.Linear  # type: ignore[attr-defined]

# Patch: chatterbox uses deprecated torch.backends.cuda.sdp_kernel() context manager.
# Replace with a wrapper around torch.nn.attention.sdpa_kernel().
# Remove when chatterbox updates to the new API.
_SDP_BACKEND_MAP = {
    "enable_flash": _torch_attention.SDPBackend.FLASH_ATTENTION,
    "enable_math": _torch_attention.SDPBackend.MATH,
    "enable_mem_efficient": _torch_attention.SDPBackend.EFFICIENT_ATTENTION,
}


@contextmanager
def _sdp_kernel_compat(**kwargs: bool) -> Generator[None, None, None]:
    backends = [b for k, b in _SDP_BACKEND_MAP.items() if kwargs.get(k, False)]
    with _torch_attention.sdpa_kernel(backends):
        yield


torch.backends.cuda.sdp_kernel = _sdp_kernel_compat  # type: ignore[assignment]

# Patch: chatterbox sets output_attentions=True on the LlamaConfig for attention alignment,
# which propagates to GenerationConfig and triggers a spurious warning about
# return_dict_in_generate. Chatterbox uses manual forward passes, not model.generate(),
# so the GenerationConfig value is never used. Remove when chatterbox fixes this.
warnings.filterwarnings(
    "ignore",
    message=r"`return_dict_in_generate` is NOT set to `True`, but `output_attentions` is",
    category=UserWarning,
    module=r"transformers\.generation\.configuration_utils",
)

# Patch: chatterbox passes output_attentions=True to every LlamaModel forward call, which
# forces SDPA attention to fall back to eager on each step. Default LlamaConfig to eager so
# the model uses it directly without the fallback warning. This also avoids a breaking change
# in transformers v5.0.0 where the automatic fallback will be removed.
# Remove when chatterbox passes attn_implementation="eager" itself.
from transformers import LlamaConfig as _LlamaConfig  # noqa: E402

_original_llama_config_init = _LlamaConfig.__init__


def _llama_config_eager_attn(
    self: _LlamaConfig, *args: object, **kwargs: object
) -> None:
    kwargs.setdefault("attn_implementation", "eager")
    _original_llama_config_init(self, *args, **kwargs)


_LlamaConfig.__init__ = _llama_config_eager_attn  # type: ignore[method-assign]

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS  # noqa: E402

import torchaudio.backend._no_backend  # noqa: E402
import torchaudio.backend._sox_io_backend  # noqa: E402
import torchaudio.backend.no_backend  # noqa: E402
import torchaudio.backend.soundfile_backend  # noqa: E402
import torchaudio.backend.sox_io_backend  # noqa: E402
from torchaudio._backend import soundfile_backend as _ta_soundfile_backend  # noqa: E402

# Patch: torchaudio 2.x backend stub modules define __getattr__ that emits a deprecation
# warning on any attribute access. Streamlit's file watcher triggers this via hasattr checks.
# Replace with silent delegators. Remove when torchaudio drops these stub modules.
torchaudio.backend.no_backend.__getattr__ = lambda name: getattr(  # type: ignore[attr-defined]
    torchaudio.backend._no_backend, name
)
torchaudio.backend.soundfile_backend.__getattr__ = lambda name: getattr(  # type: ignore[attr-defined]
    _ta_soundfile_backend, name
)
torchaudio.backend.sox_io_backend.__getattr__ = lambda name: getattr(  # type: ignore[attr-defined]
    torchaudio.backend._sox_io_backend, name
)

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
    _torch_load = torch.load
    torch.load = partial(_torch_load, map_location=torch.device(device))  # type: ignore[assignment]
    try:
        return ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
    finally:
        torch.load = _torch_load


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
    return model.sr, wav.squeeze(0).float().numpy()


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
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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
            wavfile.write(wav_buffer, sampling_rate, audio_array)
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
