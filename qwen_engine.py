"""STT engine — Qwen3-ASR-Bengali-FT (Alibaba Qwen3-ASR-1.7B fine-tuned on SUBAK.KO 241h).

Replaces the Whisper-based engine.py. Same public surface:
    transcribe(audio_path) -> str

Why this file exists separately: Qwen3-ASR has a custom architecture (not the
Whisper class in transformers), so it needs different loading code:
  - AutoModel + AutoProcessor with trust_remote_code=True
  - A hand-rolled chat-style prompt with <|audio_pad|>
  - Manual dtype casting because inputs must match model weight dtype

Design decisions
----------------

1. **Lazy module-level cache.** Same pattern as the Whisper engine — pipeline
   only loads on first transcribe() call so `import qwen_engine` stays cheap.
   Model is ~4 GB in fp16, so this matters.

2. **Device selection**:
     - Apple MPS if available (much faster than CPU for a 2 B model)
     - CUDA if on a GPU box
     - CPU last resort (works, but each transcribe() is ~10-30 s per clip)

3. **Dtype selection**:
     - MPS  → float16 (MPS doesn't support bfloat16 as of torch 2.x)
     - CUDA → bfloat16 (matches the model card's training precision)
     - CPU  → float32 (fp16/bf16 on CPU is emulated and slower than fp32)

4. **trust_remote_code=True**. Qwen3-ASR ships Python module files inside its
   HF repo that transformers loads to instantiate the custom architecture.
   This is standard for research-grade models, not a red flag.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import config

_model: Optional[Any] = None
_processor: Optional[Any] = None
_device: Optional[str] = None
_torch: Optional[Any] = None

# Explicit Bangla instruction. Without a language hint, the base
# Qwen3-ASR-1.7B misidentifies Chittagonian / Sylheti as Hindi and emits
# Devanagari script. Naming Bengali + Bangla script forces the correct
# output alphabet.
TRANSCRIBE_INSTRUCTION = (
    "Transcribe the following Bengali (Bangla) speech. "
    "Output MUST be in Bangla script (বাংলা), not Devanagari or Latin. "
    "Preserve the speaker's regional dialect words verbatim."
)


def _pick_device_and_dtype():
    """Return (device_str, torch_dtype) tuned to the host."""
    import torch
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


def _load():
    """Load model + processor once. Cached for the process lifetime."""
    global _model, _processor, _device, _torch
    if _model is not None:
        return _model, _processor, _device

    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    _torch = torch
    _device, dtype = _pick_device_and_dtype()
    print(f"[qwen_engine] loading {config.STT_MODEL_ID} on {_device} ({dtype})...")

    # Use the concrete class (not AutoModel) because the fine-tune's config.json
    # has an auto_map pointing at .py files that don't exist in the repo.
    # AutoModel would honor auto_map and fail; the concrete class ignores it
    # and uses the built-in transformers implementation.
    _processor = AutoProcessor.from_pretrained(config.STT_MODEL_ID)
    # The fine-tune's uploaded processor is missing the chat template that the
    # base Qwen3-ASR ships. Fetch it from the base repo and attach it here so
    # apply_chat_template works without depending on cache-file placement.
    if not getattr(_processor, "chat_template", None):
        import json
        from huggingface_hub import hf_hub_download
        ct_path = hf_hub_download("Qwen/Qwen3-ASR-1.7B", "chat_template.json")
        with open(ct_path) as f:
            _processor.chat_template = json.load(f)["chat_template"]

    _model = Qwen3ASRForConditionalGeneration.from_pretrained(
        config.STT_MODEL_ID,
        dtype=dtype,
    ).to(_device)
    _model.eval()
    print(f"[qwen_engine] ready")
    return _model, _processor, _device


def transcribe(audio_path: str | Path) -> str:
    """Audio file in, Bangla text out.

    Accepts any format librosa (via soundfile/ffmpeg) can read — wav, mp3,
    m4a, flac, ogg, webm.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    import librosa
    import numpy as np
    model, processor, device = _load()

    # 16 kHz mono — what the model was trained on. librosa resamples silently.
    audio, _sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Qwen3-ASR's encoder requires mel-feature length to be a multiple of 100
    # (n_window * 2). Mel hop is 10 ms → 100 frames/sec → the audio has to be
    # a whole number of seconds. Round UP with silence.
    SAMPLES_PER_SEC = 16000
    remainder = len(audio) % SAMPLES_PER_SEC
    if remainder:
        audio = np.concatenate([audio, np.zeros(SAMPLES_PER_SEC - remainder, dtype=audio.dtype)])

    # apply_chat_template expands <|audio_pad|> into the right token count for
    # this specific audio length. Hand-typing the prompt only inserts one pad
    # token and fails for anything longer than a few seconds.
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio},
            {"type": "text",  "text":  TRANSCRIBE_INSTRUCTION},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        sampling_rate=16000,
    ).to(device)

    # Model weights are in a specific dtype (fp16/bf16/fp32 depending on host).
    # Input float tensors must match — otherwise generate() raises.
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (v.to(model_dtype) if v.is_floating_point() else v)
        for k, v in inputs.items()
    }

    with _torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    # generate() returns [prompt tokens + generated tokens]. Slice off the
    # prompt so batch_decode returns only the model's answer.
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
    )[0]
    return text.strip()
