"""Small local LLM helper — cleans noisy auto-captions using description as ground truth.

Uses Ollama at localhost:11434 (default). Prefers qwen2.5:7b (strong Bangla,
proven in earlier tests). Falls through gracefully if Ollama isn't running:
callers get back their input unchanged, no exceptions.

Public surface
--------------
- ollama_available() -> bool
- pick_model() -> str | None                  # first preferred model that is installed
- clean_captions_batch(caps, description) -> list[str]
    caps: list of (start_s, end_s, text) — the noisy auto-captions
    description: full video description text (used as lyric-bank context)
    returns: list of cleaned text strings, one per input caption
             empty string means "background music, drop this chunk"

Why cleanup at all
------------------
YouTube's Bangla auto-captions are ~30-50% WER on their own. Feeding them
into training as ground truth would poison the model. But paired with the
description (uploaders paste real lyrics there ~70% of the time on lyrics
videos), an LLM can do a decent job of aligning noisy timing with clean text.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# In preference order — first one installed wins. qwen2.5:7b is the memory-
# noted default; others are reasonable fallbacks.
PREFERRED_MODELS = [
    "qwen2.5:7b",
    "qwen3-vl:8b",     # text side works fine even though it's vision-multimodal
    "qwen2.5:14b",
    "llama3.1:8b",
    "gemma3:12b",
    "gemma2:9b",
]


def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def installed_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pick_model() -> Optional[str]:
    """Return the first preferred model that's actually installed."""
    installed = installed_models()
    for pref in PREFERRED_MODELS:
        for m in installed:
            if m == pref or m.startswith(pref + ":") or m == pref.split(":")[0]:
                return m
    return installed[0] if installed else None


def _ollama_generate(model: str, prompt: str, timeout: int = 180) -> str:
    """One-shot generation, JSON API. Low temperature for consistent output."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 2048},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def clean_captions_batch(
    caps: list[tuple[float, float, str]],
    description: str,
) -> list[str]:
    """Return one cleaned Bangla-script string per input caption.

    If Ollama is unavailable, no model is installed, or the LLM output can't
    be parsed, we return the original caption texts unchanged. That's safer
    than dropping lines — the reviewer will still catch bad ones in the UI.
    """
    if not caps:
        return []

    model = pick_model()
    if not model:
        return [c[2] for c in caps]

    # Trim description — LLMs get worse with 5 KB+ context and this is Ollama-local.
    desc_trimmed = (description or "").strip()
    if len(desc_trimmed) > 3000:
        desc_trimmed = desc_trimmed[:3000] + " ... [truncated]"

    lines_json = json.dumps(
        [{"i": i, "text": c[2]} for i, c in enumerate(caps)],
        ensure_ascii=False,
    )

    prompt = f"""You are helping clean noisy auto-generated Bangla captions from a lyrics YouTube video.

The song's real lyrics, pasted by the uploader in the video description (use this as your ground-truth lyric bank; may include non-lyric text like credits and links which you should IGNORE):
\"\"\"
{desc_trimmed}
\"\"\"

The noisy auto-captions to correct (one per line, indexed):
{lines_json}

For EACH caption line, output the CORRECT Bangla text (Bangla script, not romanized) that best matches the lyric being sung at that moment. Rules:
1. Match each auto-caption to the closest line from the real lyrics above.
2. Output in Bangla script only. Do not include romanization, English translation, or explanations.
3. If a line is clearly not lyrics (music-only, chatter), output empty string "".
4. Keep the SAME number of items as the input (one output per input line).
5. Output ONLY a JSON array like: [{{"i":0,"text":"..."}},{{"i":1,"text":"..."}},...]
No prose. No markdown fences. Just the JSON array."""

    try:
        raw = _ollama_generate(model, prompt)
    except Exception:
        return [c[2] for c in caps]

    # Extract the JSON array. LLMs sometimes wrap in ```json or add explanation.
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return [c[2] for c in caps]

    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [c[2] for c in caps]

    # Rebuild in input order; missing indexes fall back to original.
    by_idx: dict[int, str] = {}
    for entry in parsed:
        if isinstance(entry, dict) and "i" in entry and "text" in entry:
            by_idx[int(entry["i"])] = str(entry["text"]).strip()

    return [by_idx.get(i, c[2]) for i, c in enumerate(caps)]


# =======================================================================
# SECOND REFERENCE — an independent text track to cross-check STT against
# =======================================================================
#
# Why a second reference at all
# -----------------------------
# One reference tells you what the STT said differently. TWO independent
# references tell you WHO is wrong. Agreement between them is strong evidence
# the text is right; disagreement flags the chunk for human review. That is
# the difference between "training data with unknown noise" and "training
# data you can grade".
#
# Where reference #2 comes from
# -----------------------------
# The uploader's own description. On lyrics/karaoke videos the description
# usually holds the written text of what is sung — a source that is entirely
# independent of the caption track (different author, different process, no
# shared error mode). We extract it from the fetched description text; the
# model is never asked to recall anything from memory, because a 7B model
# "remembering" Bangla song text would hallucinate confidently and poison the
# very data we are trying to clean.


def extract_reference_lines(description: str) -> list[str]:
    """Pull the sung-text portion out of a video description.

    Descriptions mix the actual written lyrics with credits, social links,
    hashtags, and boilerplate. This asks the LLM to keep only the former.

    Returns [] when Ollama is unavailable or nothing lyric-like is found —
    callers should treat empty as "no second reference for this video".
    """
    desc = (description or "").strip()
    if not desc:
        return []

    model = pick_model()
    if not model:
        return []

    if len(desc) > 6000:
        desc = desc[:6000] + " ... [truncated]"

    prompt = f"""Below is the description text of a Bangla song video, exactly as the uploader wrote it.

\"\"\"
{desc}
\"\"\"

Extract ONLY the lines that are the written text of what is sung in the song. Discard everything else: channel names, artist/composer credits, social media links, subscribe requests, hashtags, copyright notices, timestamps, and any English boilerplate.

Rules:
- Keep the lines in the order they appear.
- Keep them in Bangla script exactly as written. Do not translate, romanize, reword, or "improve" anything.
- If the description contains no sung text at all, output an empty array.
- Output ONLY a JSON array of strings, e.g. ["line one","line two"]
No prose. No markdown fences. Just the JSON array."""

    try:
        raw = _ollama_generate(model, prompt)
    except Exception:
        return []

    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []

    return [str(x).strip() for x in parsed if isinstance(x, (str, int, float)) and str(x).strip()]


def _char_similarity(a: str, b: str) -> float:
    """0..1 similarity on character bigrams. Cheap, no deps, order-tolerant.

    Bigrams (not raw equality) so that a partly-misheard line still scores
    above an unrelated one — which is exactly the signal we need when
    matching a reference line to a noisy caption.
    """
    a, b = a.strip(), b.strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    def bigrams(s: str) -> set[str]:
        s = re.sub(r"\s+", "", s)
        return {s[i:i + 2] for i in range(len(s) - 1)} or {s}

    ba, bb = bigrams(a), bigrams(b)
    inter = len(ba & bb)
    union = len(ba | bb)
    return inter / union if union else 0.0


def align_reference_to_captions(
    reference_lines: list[str],
    captions: list[tuple[float, float, str]],
    min_similarity: float = 0.25,
) -> list[str]:
    """Map each caption slot to its best-matching reference line.

    Walks forward through `reference_lines` rather than searching the whole
    list each time: songs are sung in order, so a monotonic pointer both
    matches better and prevents a repeated chorus from stealing matches from
    later verses. The pointer may look a few lines ahead (to survive caption
    lines that have no reference counterpart) but never jumps backwards.

    Returns one string per caption — "" where nothing matched well enough.
    """
    if not reference_lines or not captions:
        return ["" for _ in captions]

    LOOKAHEAD = 4
    out: list[str] = []
    cursor = 0

    for _start, _end, cap_text in captions:
        best_idx, best_score = -1, 0.0
        for offset in range(LOOKAHEAD + 1):
            idx = cursor + offset
            if idx >= len(reference_lines):
                break
            score = _char_similarity(cap_text, reference_lines[idx])
            if score > best_score:
                best_idx, best_score = idx, score

        if best_idx >= 0 and best_score >= min_similarity:
            out.append(reference_lines[best_idx])
            cursor = best_idx + 1  # consume it; never rewind
        else:
            out.append("")

    return out


def agreement_score(a: str, b: str) -> float:
    """0..1 agreement between two candidate transcripts of the same audio.

    Used to bucket chunks into high/medium/low confidence in the review UI.
    """
    return _char_similarity(a, b)
