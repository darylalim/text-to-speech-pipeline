import io
import time
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


VOICE_PRESETS: dict[str, VoicePreset] = {
    "English": {
        "code": "en",
        "male": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "female": [9],
    },
    "Chinese (Simplified)": {
        "code": "zh",
        "male": [0, 1, 2, 3, 5, 8],
        "female": [4, 6, 7, 9],
    },
    "French": {
        "code": "fr",
        "male": [0, 3, 4, 6, 7, 8, 9],
        "female": [1, 2, 5],
    },
    "German": {
        "code": "de",
        "male": [0, 1, 2, 4, 5, 6, 7, 9],
        "female": [3, 8],
    },
    "Hindi": {
        "code": "hi",
        "male": [2, 5, 6, 7, 8],
        "female": [0, 1, 3, 4, 9],
    },
    "Italian": {
        "code": "it",
        "male": [0, 1, 3, 4, 5, 6, 8],
        "female": [2, 7, 9],
    },
    "Japanese": {
        "code": "ja",
        "male": [2, 6],
        "female": [0, 1, 3, 4, 5, 7, 8, 9],
    },
    "Korean": {
        "code": "ko",
        "male": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "female": [0],
    },
    "Polish": {
        "code": "pl",
        "male": [0, 1, 2, 3, 5, 7, 8],
        "female": [4, 6, 9],
    },
    "Portuguese": {
        "code": "pt",
        "male": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "female": [],
    },
    "Russian": {
        "code": "ru",
        "male": [0, 1, 2, 3, 4, 7, 8],
        "female": [5, 6, 9],
    },
    "Spanish": {
        "code": "es",
        "male": [0, 1, 2, 3, 4, 5, 6, 7],
        "female": [8, 9],
    },
    "Turkish": {
        "code": "tr",
        "male": [0, 1, 2, 3, 6, 7, 8, 9],
        "female": [4, 5],
    },
}

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
    available_genders = [g for g in ("Male", "Female") if preset[g.lower()]]
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

            metrics = {
                "Model": "Bark Small",
                "Input Tokens": prompt_eval_count,
                "Output Duration": f"{output_duration:.2f}s",
                "Generation Time": f"{eval_duration}s",
            }
            for col, (label, value) in zip(st.columns(4), metrics.items()):
                col.metric(label, value)

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
