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
