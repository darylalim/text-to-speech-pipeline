from unittest.mock import patch

from streamlit_app import MODEL_CHECKPOINT, VOICE_PRESETS, get_device

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
