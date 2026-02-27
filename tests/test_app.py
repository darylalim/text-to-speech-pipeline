from unittest.mock import MagicMock, patch

import numpy as np
import torch

from streamlit_app import (
    LANGUAGES,
    MODEL_NAME,
    ChatterboxMultilingualTTS,
    generate_speech,
    get_device,
    load_model,
)

EXPECTED_LANGUAGES = [
    "Arabic",
    "Chinese",
    "Danish",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Malay",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Turkish",
]

EXPECTED_CODES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
}


class TestLanguages:
    def test_all_languages_present(self) -> None:
        assert sorted(LANGUAGES.keys()) == EXPECTED_LANGUAGES

    def test_language_codes(self) -> None:
        codes = set(LANGUAGES.values())
        assert codes == EXPECTED_CODES

    def test_language_count(self) -> None:
        assert len(LANGUAGES) == 23


class TestModelName:
    def test_model_name(self) -> None:
        assert MODEL_NAME == "Chatterbox Multilingual"


class TestGetDevice:
    def test_mps_preferred(self) -> None:
        with patch("torch.backends.mps.is_available", return_value=True):
            assert get_device() == "mps"

    def test_cuda_fallback(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            assert get_device() == "cuda"

    def test_cpu_fallback(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            assert get_device() == "cpu"


class TestLoadModel:
    def test_calls_from_pretrained_with_device(self) -> None:
        load_model("cpu")
        ChatterboxMultilingualTTS.from_pretrained.assert_called_with(device="cpu")

    def test_returns_model(self) -> None:
        model = load_model("cpu")
        assert model is ChatterboxMultilingualTTS.from_pretrained.return_value


class TestGenerateSpeech:
    def _mock_model(
        self, *, sample_rate: int = 24000, audio_length: int = 48000
    ) -> MagicMock:
        model = MagicMock()
        model.sr = sample_rate
        model.generate.return_value = torch.randn(1, audio_length)
        return model

    def test_returns_sampling_rate_and_audio(self) -> None:
        model = self._mock_model()

        sampling_rate, audio = generate_speech("hello", "en", model)

        assert sampling_rate == 24000
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (48000,)

    def test_calls_generate_with_correct_args(self) -> None:
        model = self._mock_model()

        generate_speech(
            "test text",
            "fr",
            model,
            audio_prompt_path="/tmp/ref.wav",
            cfg_weight=0.3,
            exaggeration=0.7,
        )

        model.generate.assert_called_once_with(
            "test text",
            language_id="fr",
            audio_prompt_path="/tmp/ref.wav",
            cfg_weight=0.3,
            exaggeration=0.7,
        )

    def test_default_parameters(self) -> None:
        model = self._mock_model()

        generate_speech("test", "en", model)

        model.generate.assert_called_once_with(
            "test",
            language_id="en",
            audio_prompt_path=None,
            cfg_weight=0.5,
            exaggeration=0.5,
        )

    def test_output_is_float32_numpy(self) -> None:
        model = self._mock_model()
        model.generate.return_value = torch.randn(1, 100, dtype=torch.float16)

        _, audio = generate_speech("test", "en", model)

        assert audio.dtype == np.float32
