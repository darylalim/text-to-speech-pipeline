# Chatterbox-Multilingual Migration Design

## Summary

Replace Bark Small with Chatterbox-Multilingual (500M params, 23 languages) as the TTS engine. Full replacement — no side-by-side support.

## Decisions

- **Model:** `ChatterboxMultilingualTTS` from `chatterbox-tts` package
- **Voice cloning:** Optional — users can upload a reference audio file (~10s WAV/MP3)
- **Style controls:** Expose `cfg_weight` and `exaggeration` sliders (0.0–1.0, default 0.5)
- **Approach:** Full replacement of Bark, not side-by-side

## Architecture

Single-file Streamlit app (`streamlit_app.py`). One model object handles loading and generation.

**API change from Bark:**

```python
# Bark (old)
model = BarkModel.from_pretrained(checkpoint, device_map=device, dtype=torch.float16)
processor = BarkProcessor.from_pretrained(checkpoint)
inputs = processor(text=[text], voice_preset=preset, ...)
speech_values = model.generate(**inputs.to(device))

# Chatterbox-Multilingual (new)
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
wav = model.generate(text, language_id=lang_code, audio_prompt_path=ref_audio, cfg_weight=cfg, exaggeration=exag)
```

Key simplifications:
- No separate processor
- No voice presets — default voice or cloned from uploaded audio
- Output is a tensor, saved via `torchaudio.save()`
- Sample rate via `model.sr`
- No Bark gotchas (pad_token_id, tie_word_embeddings, max_length, etc.)

## Dependencies

| Action | Package |
|--------|---------|
| Remove | `transformers`, `accelerate`, `scipy` |
| Add | `chatterbox-tts`, `torchaudio` |
| Keep | `torch`, `numpy`, `streamlit` |

## Supported Languages (23)

Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish

## UI Changes

Replace three-column voice selector (language/gender/speaker) with:
1. **Language** dropdown — 23 languages
2. **Voice cloning** — optional `st.file_uploader` (WAV/MP3)
3. **Style** — two sliders: cfg_weight (0.0–1.0) and exaggeration (0.0–1.0)

## Metrics

- Model name: "Chatterbox Multilingual"
- Input tokens → Input characters (token count not available)
- Output duration and generation time: unchanged calculation

## Files Changed

| File | Action |
|------|--------|
| `streamlit_app.py` | Rewrite model loading, generation, and UI |
| `voice_presets.json` | Delete |
| `pyproject.toml` | Update dependencies and description |
| `tests/conftest.py` | Update mocks |
| `tests/test_app.py` | Rewrite tests |
| `CLAUDE.md` | Update architecture, gotchas, dependencies |
| `README.md` | Update description |
