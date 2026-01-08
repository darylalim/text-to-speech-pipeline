import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
import torch
from transformers import BarkModel, BarkProcessor

VOICE_PRESETS = {
    "English": {
        "code": "en",
        "male": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "female": [9]
    },
    "Chinese (Simplified)": {
        "code": "zh",
        "male": [0, 1, 2, 3, 5, 8],
        "female": [4, 6, 7, 9]
    },
    "French": {
        "code": "fr",
        "male": [0, 3, 4, 6, 7, 8, 9],
        "female": [1, 2, 5]
    },
    "German": {
        "code": "de",
        "male": [0, 1, 2, 4, 5, 6, 7, 9],
        "female": [3, 8]
    },
    "Hindi": {
        "code": "hi",
        "male": [2, 5, 6, 7, 8],
        "female": [0, 1, 3, 4, 9]
    },
    "Italian": {
        "code": "it",
        "male": [0, 1, 3, 4, 5, 6, 8],
        "female": [2, 7, 9]
    },
    "Japanese": {
        "code": "ja",
        "male": [2, 6],
        "female": [0, 1, 3, 4, 5, 7, 8, 9]
    },
    "Korean": {
        "code": "ko",
        "male": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "female": [0]
    },
    "Polish": {
        "code": "pl",
        "male": [0, 1, 2, 3, 5, 7, 8],
        "female": [4, 6, 9]
    },
    "Portuguese": {
        "code": "pt",
        "male": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "female": []
    },
    "Russian": {
        "code": "ru",
        "male": [0, 1, 2, 3, 4, 7, 8],
        "female": [5, 6, 9]
    },
    "Spanish": {
        "code": "es",
        "male": [0, 1, 2, 3, 4, 5, 6, 7],
        "female": [8, 9]
    },
    "Turkish": {
        "code": "tr",
        "male": [0, 1, 2, 3, 6, 7, 8, 9],
        "female": [4, 5]
    }
}

def get_device():
    """Automatically detect the best available device in order of priority: MPS, CUDA, CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@st.cache_resource
def load_model(device):
    """Load model and processor at application startup."""
    model = BarkModel.from_pretrained("suno/bark-small", device_map=device, dtype=torch.float16)
    processor = BarkProcessor.from_pretrained("suno/bark-small")
    
    # Set pad_token_id to avoid warnings during generation
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    return model, processor

def generate_speech(text, voice_preset, model, processor, device):
    """Generate speech from text input with voice preset and attention mask"""
    inputs = processor(
        text=[text],
        return_attention_mask=True,
        return_tensors="pt",
        voice_preset=voice_preset
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        speech_values = model.generate(
            **inputs,
            do_sample=True,
            pad_token_id=model.generation_config.pad_token_id
        )
    
    sampling_rate = model.generation_config.sample_rate
    
    audio_array = speech_values.cpu().numpy().squeeze()

    return sampling_rate, audio_array

st.title("Text to Speech Pipeline")
st.markdown("Generate speech from text with Suno Bark models.")

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, processor = load_model(device)

# Initialize session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sampling_rate' not in st.session_state:
    st.session_state.sampling_rate = None

st.subheader("Text")
text_input = st.text_area(
    "Text",
    placeholder="Enter text...",
    height=150,
    help="Bark models support emotional cues like [laughs], [sighs], [music], etc."
)

st.subheader("Voice")
col1, col2, col3 = st.columns(3)

with col1:
    language = st.selectbox(
        "Language",
        options=list(VOICE_PRESETS.keys()),
        help="Select a language for the voice preset."
    )

with col2:
    available_genders = []
    if VOICE_PRESETS[language]["male"]:
        available_genders.append("Male")
    if VOICE_PRESETS[language]["female"]:
        available_genders.append("Female")
    
    gender = st.selectbox(
        "Gender",
        options=available_genders,
        help="Select a gender for the voice preset."
    )

with col3:
    gender_key = gender.lower()
    available_speakers = VOICE_PRESETS[language][gender_key]
    
    speaker_number = st.selectbox(
        "Speaker",
        options=available_speakers,
        format_func=lambda x: f"Speaker {x}",
        help="Select a speaker for the voice preset."
    )

# Generate voice preset string
language_code = VOICE_PRESETS[language]["code"]
voice_preset = f"v2/{language_code}_speaker_{speaker_number}"

if st.button("Generate", type="primary"):
    if text_input.strip() is not None:
        try:
            with st.spinner("Generating speech..."):
                sampling_rate, audio_array = generate_speech(text_input, voice_preset, model, processor, device)

                st.session_state.sampling_rate = sampling_rate
                st.session_state.audio_data = audio_array

            st.success(f"Done. Speech generated with {language} {gender} Speaker {speaker_number}.")

        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
    else:
        st.warning("Enter text.")

# Display audio player
if st.session_state.audio_data is not None:
    st.write("Audio")
    st.audio(st.session_state.audio_data, sample_rate=st.session_state.sampling_rate)
