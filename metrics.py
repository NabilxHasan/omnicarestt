"""WER and CER for Bangla ASR — normalized before comparison, pooled at corpus level.

Two metrics, on purpose:
  * WER (word error rate)  — Levenshtein at word tokens, / reference word count.
                             Reader-friendly. But for Bangla it's noisy:
                             `খাবার গুলিতে` vs `খাবারগুলিতে` counts as 2 errors
                             even though bnorm collapses whitespace, because
                             the join changes tokenization on the RAW string
                             before bnorm... wait, it doesn't, we normalize
                             first. Still: a single-letter spelling slip inside
                             a word is scored as 1 whole word wrong.
  * CER (character error rate) — Levenshtein at characters. Robust to word
                             boundary noise. The honest signal for Bangla,
                             where the failure mode is misspelling
                             a heard-correctly word.

Report BOTH. WER is what people expect to see; CER is what you actually trust.

Corpus vs sentence aggregation
------------------------------
When you have N clips, there are two ways to aggregate:
  (a) Average of per-clip WERs.  Wrong: a 3-word clip and a 30-word clip get
                                 equal weight, so a single bad short clip can
                                 wreck the number.
  (b) Sum of edits / sum of reference tokens. Right: each token contributes
                                 exactly once. This is what jiwer.wer(list, list)
                                 does when given parallel lists.

We do (b). All the numbers in the memory note (13.2% WER etc.) are computed
this way, so aggregation stays comparable across runs.

Empty edges
-----------
  * empty reference    → NaN (nothing to compare, don't dilute the corpus)
  * empty hypothesis on non-empty reference → 1.0 WER/CER (full miss)
  * both empty         → NaN
jiwer handles the second case correctly (all-deletions = 1.0), so we only
need to filter empty references before passing them in.
"""
from __future__ import annotations

from math import nan

import jiwer

from bnorm import normalize


def sentence_wer(reference: str, hypothesis: str) -> float:
    """WER for a single pair. NaN if reference is empty."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return nan
    return jiwer.wer(ref, hyp)


def sentence_cer(reference: str, hypothesis: str) -> float:
    """CER for a single pair. NaN if reference is empty."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return nan
    return jiwer.cer(ref, hyp)


def corpus_scores(pairs: list[tuple[str, str]]) -> dict:
    """WER + CER pooled across all pairs. Pairs with empty references are dropped.

    Returns `{"wer": float, "cer": float, "n": int}` where `n` is the number
    of pairs that actually contributed (empty refs excluded).
    """
    refs, hyps = [], []
    for r, h in pairs:
        r_n, h_n = normalize(r), normalize(h)
        if not r_n:
            continue          # can't score against an empty reference
        refs.append(r_n)
        hyps.append(h_n)

    if not refs:
        return {"wer": nan, "cer": nan, "n": 0}

    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "n": len(refs),
    }
