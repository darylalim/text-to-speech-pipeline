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
