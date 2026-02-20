import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from streamlit_app import (
    MODEL_CHECKPOINT,
    VOICE_PRESETS,
    BarkModel,
    generate_speech,
    get_device,
    load_model,
)

EXPECTED_LANGUAGES = [
    "Chinese (Simplified)",
    "English",
    "French",
    "German",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Turkish",
]

EXPECTED_CODES = {
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "ko",
    "pl",
    "pt",
    "ru",
    "tr",
    "zh",
}


class TestVoicePresets:
    def test_all_languages_present(self) -> None:
        assert sorted(VOICE_PRESETS.keys()) == EXPECTED_LANGUAGES

    def test_each_preset_has_required_keys(self) -> None:
        for language, preset in VOICE_PRESETS.items():
            assert "code" in preset, f"{language} missing 'code'"
            assert "male" in preset, f"{language} missing 'male'"
            assert "female" in preset, f"{language} missing 'female'"

    def test_language_codes(self) -> None:
        codes = {preset["code"] for preset in VOICE_PRESETS.values()}
        assert codes == EXPECTED_CODES

    def test_speaker_numbers_are_valid(self) -> None:
        for language, preset in VOICE_PRESETS.items():
            for speaker in preset["male"]:
                assert isinstance(speaker, int) and 0 <= speaker <= 9, (
                    f"{language} male speaker {speaker} out of range"
                )
            for speaker in preset["female"]:
                assert isinstance(speaker, int) and 0 <= speaker <= 9, (
                    f"{language} female speaker {speaker} out of range"
                )

    def test_each_language_has_at_least_one_gender(self) -> None:
        for language, preset in VOICE_PRESETS.items():
            assert preset["male"] or preset["female"], f"{language} has no speakers"


class TestVoicePresetsJson:
    def test_json_file_is_valid(self) -> None:
        path = Path(__file__).parent.parent / "voice_presets.json"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        assert len(data) == 13

    def test_json_matches_loaded_data(self) -> None:
        path = Path(__file__).parent.parent / "voice_presets.json"
        data = json.loads(path.read_text())
        assert data == VOICE_PRESETS


class TestModelCheckpoint:
    def test_checkpoint_is_bark_small(self) -> None:
        assert MODEL_CHECKPOINT == "suno/bark-small"


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
    def _mock_model(self, pad_token_id: int | None = None) -> MagicMock:
        mock = MagicMock()
        semantic_config: dict[str, int] = {"eos_token_id": 10000}
        if pad_token_id is not None:
            semantic_config["pad_token_id"] = pad_token_id
        mock.generation_config.semantic_config = semantic_config
        BarkModel.from_pretrained.return_value = mock
        return mock

    def test_configures_model(self) -> None:
        self._mock_model()
        model, _ = load_model("cpu")

        assert model.config.tie_word_embeddings is False
        assert model.generation_config.do_sample is True

    def test_removes_max_length(self) -> None:
        mock = self._mock_model()
        mock.generation_config.max_length = 20
        load_model("cpu")

        assert not hasattr(mock.generation_config, "max_length")

    def test_sets_pad_token_id_from_eos(self) -> None:
        self._mock_model()
        model, _ = load_model("cpu")

        assert model.generation_config.semantic_config["pad_token_id"] == 10001

    def test_preserves_existing_pad_token_id(self) -> None:
        self._mock_model(pad_token_id=99)
        model, _ = load_model("cpu")

        assert model.generation_config.semantic_config["pad_token_id"] == 99


class TestGenerateSpeech:
    def _mock_model_and_processor(
        self, *, sample_rate: int = 24000, audio_length: int = 48000, tokens: int = 5
    ) -> tuple[MagicMock, MagicMock]:
        model = MagicMock()
        model.generation_config.sample_rate = sample_rate
        model.generate.return_value = torch.randn(1, audio_length)

        processor = MagicMock()
        inputs = MagicMock()
        inputs.__getitem__.return_value = torch.tensor([[0] * tokens])
        inputs.to.return_value = {"input_ids": torch.tensor([[0] * tokens])}
        processor.return_value = inputs

        return model, processor

    def test_returns_sampling_rate_audio_and_token_count(self) -> None:
        model, processor = self._mock_model_and_processor()

        sampling_rate, audio, token_count = generate_speech(
            "hello", "v2/en_speaker_0", model, processor, "cpu"
        )

        assert sampling_rate == 24000
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (48000,)
        assert token_count == 5

    def test_calls_processor_with_correct_args(self) -> None:
        model, processor = self._mock_model_and_processor()

        generate_speech("test text", "v2/fr_speaker_3", model, processor, "cpu")

        processor.assert_called_once_with(
            text=["test text"],
            return_attention_mask=True,
            return_tensors="pt",
            voice_preset="v2/fr_speaker_3",
        )

    def test_converts_output_to_float32_numpy(self) -> None:
        model, processor = self._mock_model_and_processor()
        model.generate.return_value = torch.randn(1, 100, dtype=torch.float16)

        _, audio, _ = generate_speech(
            "test", "v2/en_speaker_0", model, processor, "cpu"
        )

        assert audio.dtype == np.float32
