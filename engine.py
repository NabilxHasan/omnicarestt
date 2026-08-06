"""STT engine — audio file in, Bangla text out. Nothing else.

Public surface is a single function: `transcribe(audio_path) -> str`.
Everything else in the project (evaluation, the FastAPI server, future
integrations) talks through this one call. If we ever swap the model, add
streaming, or change the runtime, only this file moves.

Why the design looks the way it does
------------------------------------

1. **Lazy module-level cache.** The pipeline object is created on the FIRST
   `transcribe()` call, not at import time. Reason: `import engine` should be
   instant so `api.py` can import it during FastAPI startup without blocking
   the event loop, and so unit tests can import it without loading 1.5 GB of
   weights they'll never use. Once loaded, the pipeline stays cached — every
   subsequent `transcribe()` reuses it.

2. **Hugging Face `pipeline` instead of manual generate().** The pipeline
   handles three annoyances for us for free:
     - Resampling: audio at any sample rate is auto-resampled to 16 kHz (what
       Whisper was trained on). If we passed 44.1 kHz raw waveform to the
       model we'd get garbage.
     - Chunking: Whisper's window is 30 s. A 2-minute recording would silently
       truncate. `chunk_length_s=30, stride_length_s=(6, 0)` splits long audio
       into 30 s chunks with 6 s of left-overlap between them, transcribes
       each, and deduplicates words at the boundary. Without this, our
       service breaks the moment someone speaks for more than half a minute.
     - Language: base multilingual Whisper needs `language="bn"` in
       generate_kwargs to prevent cross-language drift. Our bengaliai
       checkpoint was fine-tuned Bangla-only and its generation_config
       predates newer transformers' language-arg schema, so passing the arg
       raises. We rely on the fine-tuned default instead, which is stable
       for this specific model. If we ever swap to a multilingual base, add
       a generation_config patch here.

3. **Model in HF-transformers format, not CTranslate2.** The vendored
   checkpoint at `models/bengaliai-asr-whisper-medium/` is safetensors +
   tokenizer.json (HF format). `faster-whisper` would be 3-5× quicker but
   needs the CTranslate2 format, which is a one-time conversion step. We're
   not blocked on speed yet — the CPU pipeline runs ~1 s per 10 s of audio on
   the M5, which is fine for eval and the API — so we skip the conversion
   until deploy day.

4. **No CLI, no batching, no post-processing.** Deliberate. `evaluate.py` and
   `api.py` build on top. Keeping this file to one function keeps the surface
   easy to reason about and easy to swap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import config

# Cached pipeline object. `None` = not yet loaded. First transcribe() call
# populates it; every call after that reuses it.
_asr: Optional[Any] = None


def _get_pipeline() -> Any:
    """Build the ASR pipeline on first call, return the cached one after that.

    Imports are inside the function so that `import engine` is cheap: torch +
    transformers together are ~500 ms of import time and 200 MB of RAM before
    they've done anything. Deferring them keeps startup fast.
    """
    global _asr
    if _asr is not None:
        return _asr

    import torch
    from transformers import pipeline

    _asr = pipeline(
        task="automatic-speech-recognition",
        model=str(config.STT_MODEL_DIR),
        device=config.DEVICE,           # "cpu" locally; "cuda:0" on Modal/RunPod
        torch_dtype=torch.float32,      # CPU has no fp16; GPU deploy will switch to torch.float16
        chunk_length_s=30,              # Whisper's native context window
        stride_length_s=(6, 0),         # 6 s left overlap catches words spanning chunk boundaries
        return_timestamps=False,        # we return plain text; timestamps add cost + complexity
        # NOTE: no `generate_kwargs={"language": ..., "task": ...}` here. The
        # bengaliai checkpoint's generation_config is older than what current
        # transformers expects, and passing the args raises ValueError. The
        # model is Bangla-only anyway (fine-tuned on ~1200h Bangladeshi audio),
        # so the default decode is what we want. See docstring point 2.
    )
    return _asr


def transcribe(audio_path: str | Path) -> str:
    """Transcribe an audio file to Bangla text.

    Accepts any format ffmpeg can read (wav, mp3, m4a, flac, ogg). Audio is
    auto-resampled to 16 kHz and chunked if longer than 30 s. Returns the
    transcript with leading / trailing whitespace stripped; internal
    punctuation and casing are left as the model emitted them (normalization
    is bnorm's job, done at scoring time — not here).

    Raises FileNotFoundError if the path doesn't exist. Any other failure
    (corrupt audio, model error) propagates as-is: callers get a real
    traceback instead of a silent empty string.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    asr = _get_pipeline()
    result = asr(str(audio_path))
    # `pipeline` returns {"text": "..."} for a single input and a list of such
    # dicts for a list of inputs. We only ever pass one, so index into the dict.
    return result["text"].strip()
