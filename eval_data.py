"""Build the evaluation set: FLEURS bn_in + your own medical clips.

Returns a uniform list of `EvalItem(audio_path, reference, source)` so
`evaluate.py` doesn't care where a clip came from.

Two sources, on purpose:
  * FLEURS bn_in  — public, ~40 clips of general Bangla. Comparable to what
                    everyone else reports. Downloaded on demand from the
                    Hugging Face Hub, cached in audio/eval_fleurs/.
  * audio/medical/ — your own clips (drop `.wav` + `.reference.txt` pairs in).
                     Empty on day one; grows as you record real conversations.
                     The medical number is the one that actually matters —
                     FLEURS is a sanity check that we haven't regressed on
                     the general case.

Download is cached: FLEURS clips already on disk are not re-fetched. This
lets `evaluate.py` be re-run cheaply while you iterate on other layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import config


@dataclass(frozen=True)
class EvalItem:
    audio: Path
    reference: str
    source: str          # "fleurs" or "medical" — surfaces in the report table


# --- FLEURS --------------------------------------------------------------

def load_fleurs(n: int = config.FLEURS_SAMPLE_COUNT) -> list[EvalItem]:
    """Return `n` FLEURS bn_in items. Downloads any missing ones on first call.

    We stream the dataset (`streaming=True`) instead of downloading the whole
    thing — a full FLEURS test split is ~1 GB and we only need N clips.
    """
    config.EVAL_FLEURS_DIR.mkdir(parents=True, exist_ok=True)

    # Fast path: everything already on disk.
    cached = _load_cached_fleurs(n)
    if len(cached) >= n:
        return cached[:n]

    # Slow path: fetch what's missing. Import inside the function so
    # `import eval_data` doesn't pull in `datasets` (200 MB of deps) just
    # to enumerate what's already cached.
    import soundfile as sf
    from datasets import load_dataset

    print(f"[eval_data] downloading {n - len(cached)} FLEURS bn_in clips...")
    ds = load_dataset("google/fleurs", "bn_in", split="test", streaming=True)

    for i, row in enumerate(ds):
        if i >= n:
            break
        stem = f"fleurs_bn_{i:02d}"
        wav_path = config.EVAL_FLEURS_DIR / f"{stem}.wav"
        ref_path = config.EVAL_FLEURS_DIR / f"{stem}.reference.txt"
        if wav_path.exists() and ref_path.exists():
            continue
        audio = row["audio"]
        sf.write(wav_path, audio["array"], audio["sampling_rate"])
        ref_path.write_text(row["transcription"], encoding="utf-8")

    return _load_cached_fleurs(n)[:n]


def _load_cached_fleurs(n: int) -> list[EvalItem]:
    """Read whatever FLEURS clips are already on disk. Sorted for reproducibility."""
    items = []
    for wav in sorted(config.EVAL_FLEURS_DIR.glob("fleurs_bn_*.wav")):
        ref_path = wav.with_suffix(".reference.txt")
        if not ref_path.exists():
            continue
        items.append(EvalItem(
            audio=wav,
            reference=ref_path.read_text(encoding="utf-8").strip(),
            source="fleurs",
        ))
        if len(items) >= n:
            break
    return items


# --- Medical -------------------------------------------------------------

def load_medical() -> list[EvalItem]:
    """Every `.wav` in audio/medical/ that has a matching `.reference.txt`.

    Silently skips any wav without a reference — those are recordings you
    haven't transcribed yet, not eval material.
    """
    items = []
    for wav in sorted(config.MEDICAL_DIR.glob("*.wav")):
        ref_path = wav.with_suffix(".reference.txt")
        if not ref_path.exists():
            continue
        items.append(EvalItem(
            audio=wav,
            reference=ref_path.read_text(encoding="utf-8").strip(),
            source="medical",
        ))
    return items


# --- Combined ------------------------------------------------------------

def load_all(fleurs_n: int = config.FLEURS_SAMPLE_COUNT) -> list[EvalItem]:
    """FLEURS clips followed by medical clips. The order `evaluate.py` reports in."""
    return load_fleurs(fleurs_n) + load_medical()
