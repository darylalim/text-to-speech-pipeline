# Streaming Progress & Phoneme Token Display

## Overview

Add two features to the Kokoro TTS Streamlit app:

1. **Streaming progress** — real-time feedback during speech generation, showing chunk-by-chunk progress via `st.status`
2. **Phoneme token display** — show the phonetic tokens used for generation, both post-generation (expander) and on-demand (Tokenize button)

## Context

The app currently blocks with a spinner during generation and provides no visibility into phoneme tokenization. The official HF demo (Gradio-based) offers streaming audio and a Tokenize button. Streamlit does not support progressive audio playback, so we use a progress indicator approach instead.

## Design

### 1. `generate_speech` refactor

Change from returning a single `np.ndarray` to a **generator** that yields `(audio_chunk, phonemes)` tuples.

**Before:**

```python
def generate_speech(
    text: str,
    voice: str,
    pipeline: KPipeline,
    speed: float = 1.0,
) -> np.ndarray:
    chunks = list(pipeline(text, voice=voice, speed=speed))
    if not chunks:
        raise ValueError("No audio generated. Check your input text.")
    audio = np.concatenate([c.audio for c in chunks])
    return audio.astype(np.float32)
```

**After:**

```python
def generate_speech(
    text: str,
    voice: str,
    pipeline: KPipeline,
    speed: float = 1.0,
) -> Generator[tuple[np.ndarray, str], None, None]:
    generated = False
    for result in pipeline(text, voice=voice, speed=speed):
        if result.audio is not None:
            generated = True
            yield result.audio.numpy().astype(np.float32), result.phonemes
    if not generated:
        raise ValueError("No audio generated. Check your input text.")
```

Callers collect chunks, concatenate audio, and join phonemes.

### 2. `load_tokenizer` — new cached function

```python
@st.cache_resource
def load_tokenizer(lang_code: str) -> KPipeline:
    return KPipeline(lang_code=lang_code, model=False)
```

A lightweight pipeline with no model weights. Used exclusively by the Tokenize button for fast phoneme extraction.

### 3. `tokenize_text` — new function

```python
def tokenize_text(text: str, lang_code: str) -> str:
    tokenizer = load_tokenizer(lang_code)
    phonemes = []
    for result in tokenizer(text):
        if result.phonemes:
            phonemes.append(result.phonemes)
    return " ".join(phonemes)
```

Returns the joined phoneme string for display. No audio inference is performed.

### 4. Progress feedback during generation

Replace `st.spinner` with `st.status` in the Generate button handler:

```python
with st.status("Generating speech...", expanded=True) as status:
    chunks = []
    for i, (audio_chunk, phonemes) in enumerate(generate_speech(...), 1):
        chunks.append((audio_chunk, phonemes))
        st.write(f"Generated chunk {i}...")
    status.update(label="Generation complete!", state="complete")
```

**Compare mode:** Status updates include the voice name — "Generating af_heart, chunk 2..."

After status completes, audio is concatenated and displayed as before. The `st.status` widget collapses automatically.

### 5. Phoneme token display in `render_output`

Add an expander below the audio output.

**Single mode:** One expander after the audio player, metrics, and download button:

```python
with st.expander("Phoneme Tokens"):
    st.code(result["phonemes"])
```

**Compare mode:** One shared expander at the bottom (phonemes are determined by language + text, not by voice, so they are identical across voices of the same language):

```python
with st.expander("Phoneme Tokens"):
    st.code(results[0]["phonemes"])
```

`st.code` provides monospace, copyable display suited for phoneme strings.

### 6. Tokenize button

Placed next to the Generate button using columns:

```python
col1, col2 = st.columns(2)
with col1:
    generate_clicked = st.button("Generate", type="primary")
with col2:
    tokenize_clicked = st.button("Tokenize")
```

When clicked:
- Validates that text is non-empty
- Calls `tokenize_text(text_input, lang_code)`
- Displays result in `st.expander("Phoneme Tokens")` with `st.code`
- Does not generate audio or affect session history

### 7. Result dict changes

Each result dict gains a `"phonemes"` key:

```python
results.append({
    "audio": concatenated_audio,
    "voice": v,
    "text": text_input,
    "speed": speed,
    "duration": len(concatenated_audio) / SAMPLE_RATE,
    "generation_time": gen_time,
    "phonemes": all_phonemes,  # new
})
```

Phonemes are persisted in history and restored when a history entry is loaded.

### 8. Test updates

**Modified tests:**
- `TestGenerateSpeech` — update all tests for generator behavior (iterate instead of direct return, check yielded tuples)

**New tests:**
- `TestTokenizeText` — verify phoneme extraction without audio, empty text handling
- `TestRenderOutput` — verify `st.expander` and `st.code` are called with phoneme data
- `TestGenerateSpeech` — verify phonemes are yielded alongside audio chunks

**Unchanged tests:**
- `TestLanguages`, `TestModelConstants`, `TestGetVoices`, `TestLoadPipeline`, `TestAddToHistory` — no changes needed

## Files changed

- `streamlit_app.py` — all feature changes (single file app)
- `tests/test_app.py` — test updates and additions
- `tests/conftest.py` — may need mock updates for `KPipeline(model=False)` if tokenizer tests require it

## No new dependencies

All functionality uses existing `kokoro` and `streamlit` APIs.
