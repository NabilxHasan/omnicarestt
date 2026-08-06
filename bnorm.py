"""Canonicalize Bangla text so WER/CER comparisons measure real errors, not cosmetic differences.

The problem this solves:
  reference:   "খাবার গুলিতে"     (with space)
  hypothesis:  "খাবারগুলিতে।"     (joined, plus danda)
  Both read identically to a Bangla reader. Raw string compare counts them as
  100% different. That inflates WER and hides real signal.

What we strip / unify:
  1. Unicode NFC             — ড়/ঢ়/য় have a composed form (single codepoint)
                                and a base+nukta form (two codepoints). They
                                render the same but compare unequal. NFC picks
                                one canonical form for both sides.
  2. Zero-width chars        — ZWJ/ZWNJ/ZWSP/BOM affect conjunct rendering but
                                not meaning. Different tools insert them
                                differently; strip for scoring.
  3. Dashes                  — replaced with a space so "ক-খ" scores as two
                                tokens, matching how references write it.
  4. Punctuation             — danda (।), double danda (॥), plus every ASCII
                                and smart-quote punctuation the model emits.
                                References have none; keeping them would mark
                                every sentence-end as an insertion error.
  5. Digits                  — Bangla ০-৯ mapped to ASCII 0-9 so "১২" and "12"
                                match. The direction doesn't matter as long as
                                both sides go through the same map.
  6. Whitespace              — collapsed to single spaces, then trimmed.

Rule: this file only NORMALIZES. Scoring (WER/CER) lives in metrics.py
(Layer 2). Splitting them means the fine-tune notebook can normalize
predictions before pushing to Hugging Face without pulling in jiwer.
"""
from __future__ import annotations

import re
import unicodedata

# Punctuation the model tends to emit but that reference transcripts never
# include. Hyphen is handled separately (turned into a space, not removed).
_PUNCT = "।॥,;:!?\"'`()[]{}<>«»“”‘’…–—./\\|@#$%^&*_+=~"
_PUNCT_RE = re.compile(f"[{re.escape(_PUNCT)}]")

# All three dash variants collapse to a space so "ক-খ" tokenizes as two words.
_DASH_RE  = re.compile(r"[-‐‑]")

# ZWSP, ZWNJ, ZWJ, BOM. Kept invisible in-text but they break equality.
_ZW_RE    = re.compile("[​‌‍﻿]")

_WS_RE    = re.compile(r"\s+")

# Bangla digits -> ASCII digits. `str.translate` needs a codepoint-keyed dict.
_BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_DIGIT_MAP = {ord(b): str(i) for i, b in enumerate(_BN_DIGITS)}


def normalize(text: str) -> str:
    """Return the canonical form of `text` for scoring.

    Idempotent — running it twice gives the same result as running it once,
    which matters because metrics.py may call this on already-normalized text.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZW_RE.sub("", text)
    text = _DASH_RE.sub(" ", text)
    text = _PUNCT_RE.sub("", text)
    text = text.translate(_DIGIT_MAP)
    return _WS_RE.sub(" ", text).strip()
