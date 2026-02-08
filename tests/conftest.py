import sys
from unittest.mock import MagicMock

# Mock streamlit to prevent UI initialization on import
_st = MagicMock()
_st.cache_resource = lambda f: f
_st.selectbox.side_effect = lambda label, **_kw: {
    "Language": "English",
    "Gender": "Male",
    "Speaker": 0,
}.get(label, MagicMock())
_st.button.return_value = False
_st.text_area.return_value = ""
_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
sys.modules["streamlit"] = _st

# Mock transformers to prevent model downloads on import
sys.modules["transformers"] = MagicMock()
