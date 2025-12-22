import streamlit as st
import torch
from transformers import BarkModel, BarkProcessor
import scipy.io.wavfile as wavfile
import numpy as np

# Voice presets organized by language and gender
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

# Set page configuration
st.set_page_config(
    page_title="Text to Speech Pipeline",
    page_icon="🎙️",
    layout="centered"
)

# Title and description
st.title("🎙️ Text to Speech Pipeline")
st.markdown("Convert text to speech using the Suno Bark Small model.")

# Device selection
@st.cache_resource
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Load model and processor with caching
@st.cache_resource
def load_model():
    device = get_device()
    st.info(f"Loading model on device: {device}")
    
    model = BarkModel.from_pretrained("suno/bark-small", dtype=torch.float16).to(device)
    processor = BarkProcessor.from_pretrained("suno/bark-small")
    
    # Set pad_token_id to avoid warnings during generation
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    return processor, model, device

# Initialize session state for generated audio
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sampling_rate' not in st.session_state:
    st.session_state.sampling_rate = None

# Load model
try:
    with st.spinner("Loading model..."):
        processor, model, device = load_model()
    st.success(f"✓ Model loaded successfully on {device}")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Text input
st.markdown("### Enter text")
text_input = st.text_area(
    "Text to convert to speech:",
    placeholder="Type your text here. You can include emotions like [laughs], [sighs], etc.",
    height=150,
    help="The Bark model supports emotional cues like [laughs], [sighs], [music], etc."
)

# Voice preset selection
st.markdown("### Select Voice")
col1, col2, col3 = st.columns(3)

with col1:
    language = st.selectbox(
        "Language",
        options=list(VOICE_PRESETS.keys()),
        help="Select the language for the voice"
    )

with col2:
    # Get available genders for selected language
    available_genders = []
    if VOICE_PRESETS[language]["male"]:
        available_genders.append("Male")
    if VOICE_PRESETS[language]["female"]:
        available_genders.append("Female")
    
    gender = st.selectbox(
        "Gender",
        options=available_genders,
        help="Select the gender of the voice"
    )

with col3:
    # Get available speakers for selected language and gender
    gender_key = gender.lower()
    available_speakers = VOICE_PRESETS[language][gender_key]
    
    speaker_number = st.selectbox(
        "Speaker",
        options=available_speakers,
        format_func=lambda x: f"Speaker {x}",
        help="Select a specific speaker voice"
    )

# Generate the voice preset string
language_code = VOICE_PRESETS[language]["code"]
voice_preset = f"v2/{language_code}_speaker_{speaker_number}"

# Generate button
if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("⚠️ Please enter text!")
    else:
        try:
            with st.spinner("Generating speech..."):
                # Process input with voice preset and explicit attention mask
                inputs = processor(
                    text=[text_input],
                    return_tensors="pt",
                    voice_preset=voice_preset,
                    return_attention_mask=True
                )
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate speech with explicit parameters
                with torch.no_grad():
                    speech_values = model.generate(
                        **inputs,
                        do_sample=True,
                        pad_token_id=model.generation_config.pad_token_id
                    )
                
                # Get sampling rate
                sampling_rate = model.generation_config.sample_rate
                
                # Convert to numpy and move to CPU
                audio_array = speech_values.cpu().numpy().squeeze()
                
                # Store in session state
                st.session_state.audio_data = audio_array
                st.session_state.sampling_rate = sampling_rate
                
            st.success(f"✓ Speech generated successfully using {language} - {gender} - Speaker {speaker_number}")
            
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")

# Display audio player if audio exists
if st.session_state.audio_data is not None:
    st.markdown("### 🔊 Generated Audio")
    
    # Display audio player
    st.audio(st.session_state.audio_data, sample_rate=st.session_state.sampling_rate)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Suno Bark • Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
