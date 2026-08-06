"""One place to change paths, model IDs, and runtime settings.

Why this file exists: in the old lab we hardcoded model paths in 4 different
files. Swapping the model meant grep-and-pray. Every other module in this
project imports from here, so the whole system re-points with one edit.

Rule: no logic in this file. Only constants. If a value could change (a path,
a model name, a language code, an int threshold), it belongs here.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Everything is anchored on this file's location so the project works no matter
# what directory you launch python from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

AUDIO_DIR: Path       = PROJECT_ROOT / "audio"
EVAL_FLEURS_DIR: Path = AUDIO_DIR / "eval_fleurs"   # public general-Bangla eval
MEDICAL_DIR: Path     = AUDIO_DIR / "medical"       # your own medical clips

MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- Model -----------------------------------------------------------------
# The Bengali.AI Kaggle competition winner (~1200h Bangladeshi training data).
# On our earlier eval: 13.2% WER / 3.2% CER on FLEURS, 3.1% / 0.6% on medical.
# That's the baseline the whole design is built around. If Layer 5 fine-tuning
# ever produces a better checkpoint, drop it into MODELS_DIR and change this
# one line — nothing else in the system needs to move.
# --- ACTIVE MODEL: tugstugi regional (Whisper) ---
# Temporarily active while Qwen3-ASR base finishes downloading. When it lands
# we swap back to Qwen (STT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf") and change
# api.py's import back to qwen_engine.
STT_MODEL_ID: str   = "Qwen/Qwen3-ASR-1.7B-hf"       # target for after download
STT_MODEL_NAME: str = "tugstugi_bengaliai-regional-asr_whisper-medium (active)"

# STT_MODEL_DIR kept for compatibility with older scripts that still expect it.
# Points at the vendored Whisper regional model (unused while Qwen is active).
STT_MODEL_DIR: Path = MODELS_DIR / "tugstugi_bengaliai-regional-asr_whisper-medium"

# --- Model history (kept for quick revert if Qwen underperforms) ---
# Whisper base (bengaliai-asr-whisper-medium):
#   13.2% WER on FLEURS bn_in, 3.1% on synthetic medical. General Bangla.
# Whisper regional (tugstugi_bengaliai-regional-asr_whisper-medium):
#   72.1% WER on ben10 regional test set. 10 dialects, worse on standard.
# Qwen3-ASR-Bengali-FT (active):
#   23.9% WER / 9.6% CER on SUBAK.KO valid. Different eval set — numbers
#   NOT directly comparable to Whisper's FLEURS scores.

# --- Runtime ---------------------------------------------------------------
LANGUAGE: str = "bn"              # ISO 639-1 for Bangla; Whisper's token
TASK: str     = "transcribe"      # "transcribe" (Bangla out) vs "translate" (English out)

# M5 Mac: CPU is the honest default. GPU-serve config lives in run_server.sh
# and will override this at deploy time via env var (see engine.py Layer 1).
DEVICE: str = "cpu"

# --- Eval ------------------------------------------------------------------
FLEURS_SAMPLE_COUNT: int = 40     # 40 clips is where FLEURS numbers stabilize;
                                  # 5 gave us 72% and 40 gave 85% for the same
                                  # model, so small sets lie. Never go below 20.
