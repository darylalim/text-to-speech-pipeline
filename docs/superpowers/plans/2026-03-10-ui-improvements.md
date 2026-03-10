# UI Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the Streamlit TTS app with voice comparison, session state persistence, and sidebar generation history.

**Architecture:** Single-file app evolution. Extract `add_to_history` and `render_output` helper functions. Add `st.session_state` for output persistence. Add `st.sidebar` for generation history. `generate_speech` is unchanged — multi-voice looping at call site.

**Tech Stack:** Streamlit (widgets, session_state, sidebar), NumPy, SciPy

**Spec:** `docs/superpowers/specs/2026-03-10-ui-improvements-design.md`

---

## File Structure

- **Modify:** `streamlit_app.py` — all app changes
- **Modify:** `tests/conftest.py` — add mocks for new Streamlit widgets
- **Modify:** `tests/test_app.py` — add tests for new helper functions
- **Modify:** `CLAUDE.md` — update architecture documentation

---

## Chunk 1: Foundation & Text Input

### Task 1: Update conftest with new Streamlit mocks

**Files:**
- Modify: `tests/conftest.py:15-18`

The new UI features use `st.toggle`, `st.multiselect`, `st.session_state`, `st.sidebar`, and `st.rerun`. While `MagicMock` auto-creates most attributes, three need explicit configuration:

- `st.toggle` — must return `False` so the non-compare code path runs during import
- `st.multiselect` — must return `[]` (not a truthy MagicMock)
- `st.session_state` — must be a real `dict` (not MagicMock) because the app code uses the `in` operator and `.get()` method

The remaining widgets (`st.caption`, `st.sidebar`, `st.rerun`, `st.header`, `st.markdown`) work fine as auto-created MagicMock attributes. Note: `st.sidebar` is used as a context manager (`with st.sidebar:`), which MagicMock supports. Inside the `with` block, `st.*` calls go to `_st` (not the sidebar mock), so existing mocks like `_st.button.return_value = False` apply correctly.

- [ ] **Step 1: Add new mocks to conftest**

In `tests/conftest.py`, add after line 17 (`_st.columns.side_effect = ...`):

```python
_st.toggle.return_value = False
_st.multiselect.return_value = []
_st.session_state = {}
```

These defaults ensure the module-level code takes the non-compare path during import (toggle False → selectbox used, not multiselect; session_state empty → initialized to defaults).

Note: `st.session_state` is shared across all tests in a session. The module-level code mutates it during import (adding `current_output` and `history` keys). Since no existing tests depend on session state being empty, this is fine. If future tests need a clean session state, add a pytest fixture that resets it.

- [ ] **Step 2: Run tests to verify nothing breaks**

Run: `uv run pytest -v`
Expected: All 19 existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add mocks for toggle, multiselect, session_state"
```

### Task 2: Text input changes

**Files:**
- Modify: `streamlit_app.py:65-72`

Remove the 300-character hard limit, increase text area height from 150px to 200px, add a live character count via `st.caption`.

- [ ] **Step 1: Update text_area and add caption**

Replace lines 65–72:

```python
st.subheader("Text")
text_input = st.text_area(
    "Text",
    placeholder="Enter text...",
    max_chars=300,
    height=150,
    help="Maximum 300 characters per generation.",
)
```

With:

```python
st.subheader("Text")
text_input = st.text_area(
    "Text",
    placeholder="Enter text...",
    height=200,
    help="Enter text for speech generation.",
)
st.caption(f"{len(text_input)} characters")
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest -v`
Expected: All 19 tests PASS

- [ ] **Step 3: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: remove character limit, add live char count, increase text area height"
```

### Task 3: Add HISTORY_MAX constant

**Files:**
- Modify: `streamlit_app.py:15`
- Modify: `tests/test_app.py:6-14,43-51`

- [ ] **Step 1: Write failing test**

In `tests/test_app.py`, add `HISTORY_MAX` to the import:

```python
from streamlit_app import (
    HISTORY_MAX,
    LANGUAGES,
    MODEL_NAME,
    REPO_ID,
    SAMPLE_RATE,
    generate_speech,
    get_voices,
    load_pipeline,
)
```

Add test to `TestModelConstants`:

```python
    def test_history_max(self) -> None:
        assert HISTORY_MAX == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py::TestModelConstants::test_history_max -v`
Expected: FAIL with `ImportError` (HISTORY_MAX not defined)

- [ ] **Step 3: Add constant to streamlit_app.py**

After line 15 (`REPO_ID = "hexgrad/Kokoro-82M"`), add:

```python
HISTORY_MAX = 20
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py::TestModelConstants::test_history_max -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "feat: add HISTORY_MAX constant"
```

---

## Chunk 2: Helper Functions

### Task 4: add_to_history — write tests

**Files:**
- Modify: `tests/test_app.py`

Write tests for `add_to_history` before implementing it. The function inserts an entry at the front of the history list (newest first) and drops the oldest entry when the cap is exceeded.

- [ ] **Step 1: Add import and test class**

Add `add_to_history` to the import block:

```python
from streamlit_app import (
    HISTORY_MAX,
    LANGUAGES,
    MODEL_NAME,
    REPO_ID,
    SAMPLE_RATE,
    add_to_history,
    generate_speech,
    get_voices,
    load_pipeline,
)
```

Add the test class at the end of the file:

```python
class TestAddToHistory:
    def test_adds_entry_to_empty_history(self) -> None:
        history: list[list[dict[str, object]]] = []
        entry: list[dict[str, object]] = [{"voice": "af_heart", "text": "hello"}]
        add_to_history(history, entry)
        assert len(history) == 1
        assert history[0] is entry

    def test_newest_first(self) -> None:
        old: list[dict[str, object]] = [{"voice": "af_bella"}]
        history: list[list[dict[str, object]]] = [old]
        new: list[dict[str, object]] = [{"voice": "af_heart"}]
        add_to_history(history, new)
        assert history[0] is new
        assert history[1] is old

    def test_caps_at_max_entries(self) -> None:
        history: list[list[dict[str, object]]] = [
            [{"voice": f"v{i}"}] for i in range(20)
        ]
        new: list[dict[str, object]] = [{"voice": "new"}]
        add_to_history(history, new, max_entries=20)
        assert len(history) == 20

    def test_drops_oldest_when_full(self) -> None:
        history: list[list[dict[str, object]]] = [
            [{"voice": f"v{i}"}] for i in range(20)
        ]
        oldest = history[-1]
        new: list[dict[str, object]] = [{"voice": "new"}]
        add_to_history(history, new, max_entries=20)
        assert oldest not in history
        assert history[0] is new

    def test_custom_max_entries(self) -> None:
        history: list[list[dict[str, object]]] = [
            [{"voice": f"v{i}"}] for i in range(3)
        ]
        new: list[dict[str, object]] = [{"voice": "new"}]
        add_to_history(history, new, max_entries=3)
        assert len(history) == 3
        assert history[0] is new

    def test_default_max_uses_history_max(self) -> None:
        history: list[list[dict[str, object]]] = [
            [{"voice": f"v{i}"}] for i in range(HISTORY_MAX)
        ]
        new: list[dict[str, object]] = [{"voice": "new"}]
        add_to_history(history, new)
        assert len(history) == HISTORY_MAX
        assert history[0] is new
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestAddToHistory -v`
Expected: FAIL with `ImportError` (add_to_history not defined)

- [ ] **Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test: add failing tests for add_to_history"
```

### Task 5: add_to_history — implement

**Files:**
- Modify: `streamlit_app.py` (after `generate_speech` function, before module-level UI code)

- [ ] **Step 1: Implement add_to_history**

Add after the `generate_speech` function (after line 59):

```python


def add_to_history(
    history: list[list[dict[str, object]]],
    entry: list[dict[str, object]],
    max_entries: int = HISTORY_MAX,
) -> None:
    history.insert(0, entry)
    if len(history) > max_entries:
        history.pop()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_app.py::TestAddToHistory -v`
Expected: All 6 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: implement add_to_history helper"
```

### Task 6: render_output function

**Files:**
- Modify: `streamlit_app.py` (after `add_to_history`)

This function renders generation results. If there's one result, it shows the existing 4-metric row layout. If there are multiple results (compare mode), it shows shared metrics (Model, Input Characters) once, then per-voice blocks with audio player, duration, generation time, and download button.

No unit tests for this function — it only calls `st.*` widgets, matching the existing pattern where UI rendering is not unit tested.

- [ ] **Step 1: Add render_output function**

Add after `add_to_history`:

```python


def render_output(results: list[dict[str, object]]) -> None:
    if len(results) > 1:
        col1, col2 = st.columns(2)
        col1.metric("Model", MODEL_NAME)
        col2.metric("Input Characters", len(str(results[0]["text"])))
        for result in results:
            st.markdown(f"### {result['voice']}")
            st.audio(result["audio"], sample_rate=SAMPLE_RATE)
            mc1, mc2 = st.columns(2)
            mc1.metric("Output Duration", f"{result['duration']:.2f}s")
            mc2.metric("Generation Time", f"{result['generation_time']}s")
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, SAMPLE_RATE, result["audio"])  # type: ignore[arg-type]
            st.download_button(
                label=f"Download {result['voice']}",
                data=wav_buffer.getvalue(),
                file_name=f"speech_{result['voice']}.wav",
                mime="audio/wav",
                key=f"download_{result['voice']}",
            )
    else:
        result = results[0]
        st.audio(result["audio"], sample_rate=SAMPLE_RATE)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", MODEL_NAME)
        col2.metric("Input Characters", len(str(result["text"])))
        col3.metric("Output Duration", f"{result['duration']:.2f}s")
        col4.metric("Generation Time", f"{result['generation_time']}s")
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, SAMPLE_RATE, result["audio"])  # type: ignore[arg-type]
        st.download_button(
            label="Download Audio",
            data=wav_buffer.getvalue(),
            file_name="speech.wav",
            mime="audio/wav",
        )
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 3: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add render_output helper for single and compare modes"
```

---

## Chunk 3: Session State, Voice Comparison & Generation

### Task 7: Session state initialization and output persistence

**Files:**
- Modify: `streamlit_app.py` (module-level UI code)

Initialize session state keys and replace the inline output rendering with `render_output` called from session state. After generation, store results in session state and call `st.rerun()` to update the sidebar.

- [ ] **Step 1: Add session state initialization**

Add after the `render_output` function, before `st.title(...)`:

```python


if "current_output" not in st.session_state:
    st.session_state["current_output"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
```

- [ ] **Step 2: Replace the generate block and output rendering**

Replace the entire generate block (lines 107–136 in the original, starting with `if st.button("Generate", type="primary"):`) with:

```python
if st.button("Generate", type="primary"):
    if text_input.strip():
        try:
            results = []
            with st.spinner("Generating speech..."):
                start = time.perf_counter()
                audio_array = generate_speech(text_input, voice, pipeline, speed=speed)
                gen_time = round(time.perf_counter() - start, 2)
                results.append({
                    "audio": audio_array,
                    "voice": voice,
                    "text": text_input,
                    "speed": speed,
                    "duration": len(audio_array) / SAMPLE_RATE,
                    "generation_time": gen_time,
                })
            st.session_state["current_output"] = results
            add_to_history(st.session_state["history"], results)
            st.rerun()
        except Exception as e:
            st.exception(e)
    else:
        st.warning("Enter text.")

if st.session_state["current_output"] is not None:
    render_output(st.session_state["current_output"])
```

Note: This step uses the single-voice `voice` variable. Multi-voice comparison is added in Task 8.

- [ ] **Step 3: Run tests**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: persist output in session state, render from state"
```

### Task 8: Voice comparison toggle and multiselect

**Files:**
- Modify: `streamlit_app.py` (voice selection section)

Add a Compare Voices toggle. Read its state from `st.session_state` before rendering the voice column so the toggle can appear below the columns while its value is available above. When enabled, replace the single voice `st.selectbox` with `st.multiselect` (max 3).

- [ ] **Step 1: Replace voice selection code**

Replace the voice selection section (from `st.subheader("Voice")` through the voice selectbox) with:

```python
st.subheader("Voice")
compare_mode = st.session_state.get("compare_mode", False)
voice_col1, voice_col2 = st.columns(2)

with voice_col1:
    language = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        help="Select a language for speech generation.",
    )

lang_code = LANGUAGES[language]

with voice_col2:
    voices = get_voices(lang_code)
    if compare_mode:
        selected_voices = st.multiselect(
            "Voices",
            options=voices,
            max_selections=3,
            help="Select up to 3 voices to compare.",
        )
    else:
        voice = st.selectbox(
            "Voice",
            options=voices,
            help="The second letter indicates gender: 'f' for female, 'm' for male.",
        )
        selected_voices = [voice]

st.toggle("Compare Voices", key="compare_mode")
```

The `st.toggle` widget uses `key="compare_mode"` which stores its value in `st.session_state["compare_mode"]`. The `st.session_state.get("compare_mode", False)` at the top reads this value on rerun, making it available before the toggle is rendered.

- [ ] **Step 2: Update generate block for multi-voice**

Replace the generate block (from Task 7) with:

```python
if st.button("Generate", type="primary"):
    if not text_input.strip():
        st.warning("Enter text.")
    elif compare_mode and not selected_voices:
        st.warning("Select at least one voice.")
    else:
        try:
            results = []
            with st.spinner("Generating speech..."):
                for v in selected_voices:
                    start = time.perf_counter()
                    audio_array = generate_speech(text_input, v, pipeline, speed=speed)
                    gen_time = round(time.perf_counter() - start, 2)
                    results.append({
                        "audio": audio_array,
                        "voice": v,
                        "text": text_input,
                        "speed": speed,
                        "duration": len(audio_array) / SAMPLE_RATE,
                        "generation_time": gen_time,
                    })
            st.session_state["current_output"] = results
            add_to_history(st.session_state["history"], results)
            st.rerun()
        except Exception as e:
            st.exception(e)

if st.session_state["current_output"] is not None:
    render_output(st.session_state["current_output"])
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add compare voices toggle with multi-voice generation"
```

---

## Chunk 4: Sidebar History & Documentation

### Task 9: Sidebar generation history

**Files:**
- Modify: `streamlit_app.py` (add sidebar section)

Add a sidebar that displays past generations (newest first). Each entry shows a truncated text preview, voice names, audio players, and a Load button that copies the entry into `current_output`.

The sidebar code must run BEFORE the main area output rendering so that Load button clicks update `current_output` before it is rendered.

- [ ] **Step 1: Add sidebar section**

Add after session state initialization (after the `st.session_state["history"] = []` line), before `st.title(...)`:

```python

with st.sidebar:
    st.header("Generation History")
    history = st.session_state["history"]
    if not history:
        st.caption("No generations yet.")
    for i, entry in enumerate(history):
        text = str(entry[0]["text"])
        text_preview = text[:50] + ("..." if len(text) > 50 else "")
        voice_names = ", ".join(str(r["voice"]) for r in entry)
        st.markdown(f"**{text_preview}**")
        st.caption(voice_names)
        for result in entry:
            st.audio(result["audio"], sample_rate=SAMPLE_RATE)
        if st.button("Load", key=f"load_{i}"):
            st.session_state["current_output"] = entry
```

The sidebar runs before the output rendering section, so clicking Load sets `current_output` before `render_output` is called. No `st.rerun()` needed for Load.

- [ ] **Step 2: Run tests**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 3: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add sidebar generation history with load buttons"
```

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Update the Architecture section to reflect the new features and helper functions.

- [ ] **Step 1: Update the UI section**

Replace the UI section in CLAUDE.md with:

```markdown
### UI

- Text input (no character limit, live character count)
- Language selection (9 languages via `LANGUAGES` dict)
- Voice selector (dynamically populated from HuggingFace Hub)
- Compare Voices toggle: switches voice selector to multiselect (max 3 voices)
- Speed slider (0.5–2.0, default 1.0)
- Generated audio displayed in browser player via `st.audio`
- WAV download via `st.download_button` (saved with `scipy.io.wavfile.write`)
- Metrics via `st.metric`: model name, input characters, output duration, generation time
- Compare mode: shared Model + Input Characters metrics, per-voice Duration + Generation Time
- Errors shown with `st.exception()`
- Session state (`st.session_state`) persists current output and generation history across reruns
- Sidebar generation history (max 20 entries, newest first) with Load buttons
```

- [ ] **Step 2: Add helper functions to Architecture > Files**

In the Architecture > Files section of CLAUDE.md, update the `streamlit_app.py` bullet:

Replace:
```markdown
- `streamlit_app.py` — single-file app with text input, language/voice selection, speed control, and audio playback
```

With:
```markdown
- `streamlit_app.py` — single-file app with text input, language/voice selection, speed control, audio playback, voice comparison, and session-based history. Helper functions: `add_to_history` (history management), `render_output` (output rendering for single and compare modes)
```

Note: The old UI section's last bullet (`- No session state for audio persistence`) is removed by the full replacement in Step 1.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for UI improvements"
```

### Task 11: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Run lint, format, typecheck**

Run: `uv run ruff check . && uv run ruff format . && uv run ty check`
Expected: No errors (ty may show pre-existing warnings — only check for new ones)

- [ ] **Step 3: Manual smoke test**

Run: `uv run streamlit run streamlit_app.py`

Verify:
- Text area has no character limit and shows live count
- Compare Voices toggle switches between selectbox and multiselect
- Single voice generation shows 4-metric row and audio player
- Compare mode generates for multiple voices with stacked results
- Output persists when changing slider/language (no disappearing audio)
- Sidebar shows generation history with Load buttons
- Load button restores previous generation to main output area
