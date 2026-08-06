"""Second STT engine — base Whisper large-v3-turbo via MLX (Apple Silicon GPU).

Same public surface as engine.py:
    transcribe(audio_path) -> str

Why a second engine at all
--------------------------
The two models fail in *different* places, which is the whole point:

  engine.py  (tugstugi regional, Whisper-medium fine-tuned on 10 BD dialects)
      strong: Bangla orthography, conjuncts, dialect vocabulary
              (আঁই / ফোয়া / ছওল survive instead of being normalised away)
      weak:   English loanwords and code-switching, music beds, heavy noise
              — its fine-tune corpus was almost entirely clean Bangla speech

  engine2.py (base whisper-large-v3-turbo, 680k h multilingual pretraining)
      strong: English inside Bangla sentences, noise robustness, longer context
      weak:   Bangla spelling — it never learned the conjuncts properly and
              collapses them (ব্যথা → বেথা), which is exactly the failure the
              regional model was built to fix

So: where they AGREE, confidence is high. Where they DISAGREE, the
disagreement itself is the signal — usually one of them is in its known weak
spot, and which one tells you what to trust.

Runtime notes
-------------
- MLX runs on the M5's GPU, so this engine and engine.py (CPU/transformers)
  can genuinely run at the same time instead of fighting for the same cores.
- language="bn" is forced on purpose. Left to auto-detect, base Whisper often
  flips to Hindi or English on code-switched audio and emits Devanagari or
  Latin script, which makes the two engines' outputs incomparable. Forcing bn
  makes it transliterate English words into Bangla script — the same
  convention the regional model uses, and how Bangla speakers actually write
  code-switched text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# MLX-converted base Whisper. Not in config.py because config tracks the
# PRIMARY model; this one is a fixed second opinion.
MLX_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
MODEL_NAME = "whisper-large-v3-turbo (mlx)"

_loaded: bool = False


def _ensure_loaded() -> Any:
    """Import mlx_whisper on first use.

    mlx_whisper has no explicit load step — it caches the model internally
    after the first transcribe call. We just defer the import so that
    `import engine2` stays cheap and the server can start without it.
    """
    global _loaded
    import mlx_whisper
    if not _loaded:
        print(f"[engine2] using {MLX_MODEL_REPO} (loads on first transcribe)")
        _loaded = True
    return mlx_whisper


def transcribe(audio_path: str | Path) -> str:
    """Audio file in, Bangla text out. Any format ffmpeg can read."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    mlx_whisper = _ensure_loaded()
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=MLX_MODEL_REPO,
        language="bn",       # see module docstring — do not switch to auto
        task="transcribe",   # not "translate" (that would emit English)
    )
    return result["text"].strip()
