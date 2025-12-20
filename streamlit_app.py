import streamlit as st
import torch
from transformers import AutoProcessor, AutoModel
import scipy.io.wavfile as wavfile
import numpy as np
import io

# Set page configuration
st.set_page_config(
    page_title="Text to Speech Pipeline",
    page_icon="🎙️",
    layout="centered"
)

# Title and description
st.title("🎙️ Text to Speech Pipeline")
st.markdown("Convert text to speech using the Suno Bark model.")

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
    
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = AutoModel.from_pretrained("suno/bark")
    model = model.to(device)
    
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

# Generate button
if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("⚠️ Please enter text!")
    else:
        try:
            with st.spinner("Generating speech..."):
                # Process input
                inputs = processor(
                    text=[text_input],
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate speech
                with torch.no_grad():
                    speech_values = model.generate(**inputs, do_sample=True)
                
                # Get sampling rate
                sampling_rate = model.generation_config.sample_rate
                
                # Convert to numpy and move to CPU
                audio_array = speech_values.cpu().numpy().squeeze()
                
                # Store in session state
                st.session_state.audio_data = audio_array
                st.session_state.sampling_rate = sampling_rate
                
            st.success("✓ Speech generated successfully")
            
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")

# Display audio player and download button if audio exists
if st.session_state.audio_data is not None:
    st.markdown("### 🔊 Generated Audio")
    
    # Display audio player
    st.audio(st.session_state.audio_data, sample_rate=st.session_state.sampling_rate)
    
    # Create WAV file for download
    st.markdown("### 💾 Download Audio")
    
    # Convert to WAV format in memory
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, st.session_state.sampling_rate, st.session_state.audio_data)
    wav_buffer.seek(0)
    
    # Download button
    st.download_button(
        label="⬇️ Download as WAV",
        data=wav_buffer,
        file_name="generated_speech.wav",
        mime="audio/wav",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Suno Bark • Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
