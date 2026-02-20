import io
import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
import torch
from transformers import BarkModel, BarkProcessor


class VoicePreset(TypedDict):
    code: str
    male: list[int]
    female: list[int]


VOICE_PRESETS: dict[str, VoicePreset] = json.loads(
    (Path(__file__).parent / "voice_presets.json").read_text()
)

MODEL_CHECKPOINT = "suno/bark-small"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(device: str) -> tuple[BarkModel, BarkProcessor]:
    model = BarkModel.from_pretrained(
        MODEL_CHECKPOINT, device_map=device, dtype=torch.float16
    )
    model.config.tie_word_embeddings = False
    processor = BarkProcessor.from_pretrained(MODEL_CHECKPOINT)

    semantic_config = model.generation_config.semantic_config
    if semantic_config.get("pad_token_id") is None:
        semantic_config["pad_token_id"] = semantic_config["eos_token_id"] + 1

    return model, processor


def generate_speech(
    text: str,
    voice_preset: str,
    model: BarkModel,
    processor: BarkProcessor,
    device: str,
) -> tuple[int, np.ndarray, int]:
    inputs = processor(
        text=[text],
        return_attention_mask=True,
        return_tensors="pt",
        voice_preset=voice_preset,
    )

    prompt_eval_count = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        speech_values = model.generate(
            **inputs.to(device),
            do_sample=True,
        )

    sampling_rate = model.generation_config.sample_rate
    return (
        sampling_rate,
        speech_values.cpu().float().numpy().squeeze(),
        prompt_eval_count,
    )


st.title("Text to Speech Pipeline")
st.write("Generate speech from text with Suno Bark models.")

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, processor = load_model(device)

st.subheader("Text")
text_input = st.text_area(
    "Text",
    placeholder="Enter text...",
    height=150,
    help="Bark models support emotional cues like [laughs], [sighs], [music], etc.",
)

st.subheader("Voice")
voice_col1, voice_col2, voice_col3 = st.columns(3)

with voice_col1:
    language = st.selectbox(
        "Language",
        options=list(VOICE_PRESETS.keys()),
        help="Select a language for the voice preset.",
    )

preset = VOICE_PRESETS[language]

with voice_col2:
    available_genders: list[str] = []
    if preset["male"]:
        available_genders.append("Male")
    if preset["female"]:
        available_genders.append("Female")
    gender = st.selectbox(
        "Gender",
        options=available_genders,
        help="Select a gender for the voice preset.",
    )

with voice_col3:
    speakers = preset["male"] if gender == "Male" else preset["female"]

    speaker_number = st.selectbox(
        "Speaker",
        options=speakers,
        format_func=lambda x: f"Speaker {x}",
        help="Select a speaker for the voice preset.",
    )

voice_preset = f"v2/{preset['code']}_speaker_{speaker_number}"

if st.button("Generate", type="primary"):
    if text_input.strip():
        try:
            with st.spinner("Generating speech..."):
                start = time.perf_counter()
                sampling_rate, audio_array, prompt_eval_count = generate_speech(
                    text_input, voice_preset, model, processor, device
                )
                eval_duration = round(time.perf_counter() - start, 2)
                output_duration = len(audio_array) / sampling_rate

            st.audio(audio_array, sample_rate=sampling_rate)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model", "Bark Small")
            col2.metric("Input Tokens", prompt_eval_count)
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
