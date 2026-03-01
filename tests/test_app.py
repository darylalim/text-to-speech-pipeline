import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import diffusers.models.lora
import torch.nn as nn
import torchaudio.backend.no_backend
import torchaudio.backend.soundfile_backend
import torchaudio.backend.sox_io_backend

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
    "ar",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "sw",
    "tr",
    "zh",
}


class TestLanguages:
    def test_all_languages_present(self) -> None:
        assert sorted(LANGUAGES.keys()) == EXPECTED_LANGUAGES

    def test_language_codes(self) -> None:
        codes = set(LANGUAGES.values())
        assert codes == EXPECTED_CODES

    def test_language_count(self) -> None:
        assert len(LANGUAGES) == 23


class TestDependencyPatches:
    def test_lora_compatible_linear_replaced_with_nn_linear(self) -> None:
        assert diffusers.models.lora.LoRACompatibleLinear is nn.Linear

    def test_torchaudio_no_backend_getattr_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hasattr(torchaudio.backend.no_backend, "__path__")

    def test_torchaudio_soundfile_backend_getattr_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hasattr(torchaudio.backend.soundfile_backend, "__path__")

    def test_torchaudio_sox_io_backend_getattr_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hasattr(torchaudio.backend.sox_io_backend, "__path__")

    def test_sdp_kernel_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                pass

    def test_generation_config_output_attentions_no_warning(self) -> None:
        from transformers import GenerationConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings(
                "ignore",
                message=r"`return_dict_in_generate` is NOT set to `True`, but `output_attentions` is",
                category=UserWarning,
            )
            GenerationConfig(output_attentions=True)
        assert not any("output_attentions" in str(w.message) for w in caught)


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
        ChatterboxMultilingualTTS.from_pretrained.assert_called_with(  # type: ignore[union-attribute]
            device=torch.device("cpu")
        )

    def test_returns_model(self) -> None:
        model = load_model("cpu")
        assert model is ChatterboxMultilingualTTS.from_pretrained.return_value  # type: ignore[union-attribute]

    def test_patches_torch_load_with_map_location(self) -> None:
        captured_load = {}

        def fake_from_pretrained(device: torch.device) -> MagicMock:
            captured_load["fn"] = torch.load
            return MagicMock()

        ChatterboxMultilingualTTS.from_pretrained.side_effect = fake_from_pretrained  # type: ignore[union-attribute]
        try:
            load_model("cpu")
            # During from_pretrained, torch.load should have map_location baked in
            assert captured_load["fn"].keywords["map_location"] == torch.device("cpu")
        finally:
            ChatterboxMultilingualTTS.from_pretrained.side_effect = None  # type: ignore[union-attribute]

    def test_restores_torch_load_after_loading(self) -> None:
        original = torch.load
        load_model("cpu")
        assert torch.load is original


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
