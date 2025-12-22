# Text to Speech Pipeline

## Overview
A Streamlit web application that converts text to speech using the Suno Bark Small model. Features automatic device detection (MPS/CUDA/CPU), voice selection across 13 languages, text input, audio playback, and WAV file downloads.

**Supported Languages:** Chinese (Simplified), English, French, German, Hindi, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, and Turkish.

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Launch the app with:
```bash
streamlit run streamlit_app.py
```

The app will:
- Automatically detect and use MPS acceleration on Apple Silicon Macs
- Download the Bark Small model on first run (~2GB, one-time)
- Open in your default browser

## How to Use

1. Enter text in the input box
2. Select your preferred language, gender, and speaker
3. Click "Generate Speech" 
4. Play the generated audio in the browser
5. Download as a WAV file

## Tips for Using Bark

Bark supports special tokens for expressive speech:
- `[laughs]` - Adds laughter
- `[sighs]` - Adds a sigh
- `[music]` - Adds musical tones
- `[gasps]` - Adds a gasp
- `♪` - Indicates singing

Example: `"Hello! [laughs] I'm having a great day. ♪ La la la ♪"`
