"""Correct STT output for Bangla/Banglish medical speech — lexicon first, LLM second, verified always.

What this solves
----------------
The STT engine hears correctly and spells wrong. PLAN.md documents the pattern:
conjuncts collapse (ব্যথা → বেথা), spurious ও-কার appears (ধরে → ধোরে), and
loanwords mangle (প্যারাসিটামল → পেরাসিটমো). Those are orthographic errors on
words the model *did* hear, which means they are recoverable from text alone.

Why this is not just "ask an LLM to fix it"
-------------------------------------------
PLAN.md rejects free-form LLM post-correction for a specific reason: in earlier
tests an LLM given a medical glossary confidently mapped garbled audio to the
WRONG medical term. A doctor spots visible garbage; they silently trust a
confident wrong drug name. So the LLM here is never trusted on its own word.

The pipeline is three stages, and the third is the one that matters:

  1. LEXICON  — deterministic fuzzy match against a closed vocabulary of drugs,
                symptoms, and tests. Normalized edit distance, with a margin
                requirement so an ambiguous match is reported as ambiguous
                rather than silently resolved. This stage cannot invent a term
                that is not in the vocabulary, because it only ever copies from
                the vocabulary.

  2. LLM      — qwen2.5:7b via local Ollama, for what the lexicon can't do:
                conjunct spelling inside ordinary words, word segmentation, and
                code-switch normalization. It is asked to correct spelling only.

  3. VERIFY   — the LLM's output is checked before it is allowed through:
                  a) it may not rewrite more than MAX_REWRITE_RATIO of the text
                  b) it may not introduce a medical term that wasn't already
                     in the input (or a near-match of something in the input)
                If either check fails, the LLM output is DISCARDED and the
                lexicon result is returned instead.

That last stage is the whole point. It makes "the LLM hallucinated a drug name"
a structurally detectable event rather than something you find out about in a
clinic. Anything the pipeline is unsure of lands in `flags` for human review
instead of being quietly resolved.

Public surface
--------------
    correct(text) -> dict
    lexicon_pass(text) -> (corrected_text, changes, flags)
    llm_available() -> bool
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Optional

import requests

from medical_llm import DRUG_MAP, SYMPTOM_MAP, TEST_MAP

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Same preference order as llm_helper.py — qwen2.5:7b is the project's proven
# Bangla model. Kept as its own list so the two callers can diverge later
# without surprising each other.
PREFERRED_MODELS = ["qwen2.5:7b", "qwen2.5:14b", "qwen3-vl:8b", "llama3.1:8b", "gemma3:12b"]

# --- Tuning knobs ----------------------------------------------------------

# A candidate must be at least this similar to a vocabulary entry to be
# considered a match at all.
#
# 0.57 is measured, not guessed. Against this vocabulary:
#   worst true positive   পেরাসিটমো → প্যারাসিটামল   0.583
#   worst false-positive  মিগ্রা      → অমিপ্রাজল      0.556   (an ordinary word!)
#   worst adversarial     ডাক্তারবাবু → রক্তচাপ        0.364
# The gap between the last true positive and the first false positive is only
# 0.027, which is far too tight to rely on alone — MIN_MARGIN below is what
# actually carries the safety, not this number. Re-measure both if you add
# vocabulary; scripts/threshold diagnostics live in the test file.
MIN_SIMILARITY = 0.57

# Wider windows need a HIGHER bar, and this is not a fudge factor — it follows
# from how ASR actually fails. A real recognition error inside a multi-word
# phrase corrupts one word and leaves the rest intact, so a genuine two-word
# match scores high (মাথা বেথা → মাথা ব্যথা = 0.80). A *marginal* multi-word
# match is therefore almost always coincidence, and there is far more surface
# for coincidence once you slide two- and three-token windows across ordinary
# speech.
#
# Measured consequence of getting this wrong: at a flat 0.57 the phrase
# টেস্ট করাতে ("to get a test done") matched চেস্ট এক্সরে ("Chest X-Ray") at
# 0.583 and the pipeline silently invented a radiology order. That is the exact
# confident-wrong-term failure PLAN.md refuses to ship, produced by the
# supposedly safe deterministic stage.
MIN_SIMILARITY_BY_WIDTH: dict[int, float] = {1: 0.57, 2: 0.75, 3: 0.80}


def _threshold_for(width: int) -> float:
    return MIN_SIMILARITY_BY_WIDTH.get(width, 0.85)

# The best match must beat the runner-up by this much, otherwise the span is
# ambiguous and goes to `flags` instead of being corrected.
#
# This is the load-bearing guard. মিগ্রা (Bangla for "mg", a perfectly correct
# word) scores 0.556 against অমিপ্রাজল — but its margin is 0.000 because two
# vocabulary entries tie for it, so it is rejected as ambiguous rather than
# silently "corrected" into a drug name. Real corrections have wide margins:
# পেরাসিটমো 0.250, মেটফরমিন 0.616. The tightest genuine one observed is
# ওমিপ্রাজোল at 0.100, which is why this sits at 0.08 and not higher.
MIN_MARGIN = 0.08

# Tokens shorter than this are never fuzzy-matched — short Bangla words collide
# with drug-name prefixes far too easily.
MIN_TOKEN_LEN = 4

# If the LLM rewrites more than this fraction of the text, we assume it stopped
# correcting and started composing. Discard and fall back.
MAX_REWRITE_RATIO = 0.40


# --- Text utilities --------------------------------------------------------

def _nfc(text: str) -> str:
    """Canonical unicode form. Bangla ড়/ঢ়/য় have composed and base+nukta forms
    that render identically but compare unequal — this picks one."""
    return unicodedata.normalize("NFC", text)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Two-row DP — we only ever need the previous row."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """1.0 = identical, 0.0 = nothing in common. Normalized by the longer string
    so that a short word and a long word can never score spuriously high."""
    a, b = _nfc(a), _nfc(b)
    if not a or not b:
        return 0.0
    longest = max(len(a), len(b))
    return 1.0 - (_edit_distance(a, b) / longest)


# --- Vocabulary ------------------------------------------------------------

def _build_vocab() -> dict[str, str]:
    """Closed vocabulary: Bangla surface form -> category.

    Built from medical_llm's maps so there is exactly one place to add a drug.
    The LLM is shown this list but can never add to it — that asymmetry is what
    makes stage 3's containment check meaningful.
    """
    vocab: dict[str, str] = {}
    for term in DRUG_MAP:
        vocab[_nfc(term)] = "drug"
    for term in SYMPTOM_MAP:
        vocab[_nfc(term)] = "symptom"
    for term in TEST_MAP:
        vocab[_nfc(term)] = "test"
    return vocab


VOCAB = _build_vocab()

# Bangla + Latin word characters, digits, and the internal marks that belong
# inside a Bangla word. Punctuation and whitespace are the separators.
_TOKEN_RE = re.compile(r"[ঀ-৿A-Za-z0-9]+")

# Vocabulary bucketed by word count. 15 of the 49 entries are multi-word
# (গলা ব্যথা, ব্লাড সুগার, প্রস্রাবে জ্বালাপোড়া …) and a single-token matcher can
# never reach them — it would compare one token against a two-word phrase and
# score it far too low. Matching therefore runs over token *windows*, and a
# window of N words is only ever compared against vocabulary entries of N
# words. That keeps the normalized distance meaningful on both sides.
VOCAB_BY_WORDS: dict[int, list[str]] = {}
for _term in VOCAB:
    VOCAB_BY_WORDS.setdefault(len(_term.split()), []).append(_term)

MAX_VOCAB_WORDS = max(VOCAB_BY_WORDS) if VOCAB_BY_WORDS else 1


def _best_match(phrase: str, n_words: Optional[int] = None) -> tuple[Optional[str], float, float]:
    """Return (best_vocab_term, best_score, margin_over_runner_up).

    Only entries with the same word count as `phrase` are considered, so a
    one-word token is never "corrected" into a two-word phrase.
    """
    if n_words is None:
        n_words = len(phrase.split())
    candidates = VOCAB_BY_WORDS.get(n_words, [])
    if not candidates:
        return None, 0.0, 0.0

    scored = sorted(
        ((term, _similarity(phrase, term)) for term in candidates),
        key=lambda kv: kv[1],
        reverse=True,
    )
    best_term, best_score = scored[0]
    runner_up = scored[1][1] if len(scored) > 1 else 0.0
    return best_term, best_score, best_score - runner_up


# --- Stage 1: lexicon ------------------------------------------------------

def lexicon_pass(text: str) -> tuple[str, list[dict], list[dict]]:
    """Deterministic closed-vocabulary correction over token windows.

    Scans left to right trying the LONGEST window first (3 tokens down to 1),
    so that "পেট বেথা" is matched as the two-word symptom পেট ব্যথা rather than
    having বেথা corrected in isolation. A matched window is consumed whole and
    the scan resumes after it.

    Three outcomes per window:
      * exact vocabulary hit          -> left alone, scan advances past it
      * clears similarity AND margin  -> corrected, recorded in `changes`
      * clears similarity, not margin -> left alone, recorded in `flags`

    That third case is the important one. An ambiguous drug name is precisely
    where guessing costs more than abstaining, so the span is returned
    unmodified with a note rather than resolved.

    Returns (corrected_text, changes, flags).
    """
    text = _nfc(text)
    tokens = list(_TOKEN_RE.finditer(text))
    changes: list[dict] = []
    flags: list[dict] = []
    out: list[str] = []
    last_end = 0
    i = 0

    while i < len(tokens):
        matched = False

        for size in range(min(MAX_VOCAB_WORDS, len(tokens) - i), 0, -1):
            span_tokens = tokens[i:i + size]
            phrase = " ".join(t.group(0) for t in span_tokens)
            span_start, span_end = span_tokens[0].start(), span_tokens[-1].end()

            # Exact hit: nothing to do, but consume the window so a longer
            # correct phrase isn't re-examined piecemeal.
            if phrase in VOCAB:
                out.append(text[last_end:span_start])
                out.append(text[span_start:span_end])
                last_end = span_end
                i += size
                matched = True
                break

            # Single short tokens collide with drug-name prefixes far too
            # easily to be worth fuzzy-matching.
            if size == 1 and len(phrase) < MIN_TOKEN_LEN:
                continue

            best, score, margin = _best_match(phrase, size)
            if best is None or score < _threshold_for(size):
                continue

            if margin < MIN_MARGIN:
                flags.append({
                    "token": phrase,
                    "reason": "ambiguous_match",
                    "detail": f"closest vocabulary entry '{best}' scored {score:.2f} "
                              f"but the runner-up was within {margin:.2f} — left uncorrected",
                })
                continue

            out.append(text[last_end:span_start])
            out.append(best)
            last_end = span_end
            changes.append({
                "from": phrase,
                "to": best,
                "kind": VOCAB[best],
                "stage": "lexicon",
                "confidence": round(score, 3),
            })
            i += size
            matched = True
            break

        if not matched:
            i += 1

    out.append(text[last_end:])
    return "".join(out), changes, flags


# --- Ollama plumbing -------------------------------------------------------

def _installed_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pick_model() -> Optional[str]:
    installed = _installed_models()
    for pref in PREFERRED_MODELS:
        for m in installed:
            if m == pref or m.startswith(pref.split(":")[0] + ":"):
                return m
    return installed[0] if installed else None


def llm_available() -> bool:
    return pick_model() is not None


def _generate(model: str, prompt: str, timeout: int = 180) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Near-greedy. This is a correction task, not a creative one —
            # sampling diversity here would just mean less reproducible output.
            "options": {"temperature": 0.1, "num_predict": 1024},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


# --- Stage 3: verification -------------------------------------------------

def _medical_terms_present(text: str) -> set[str]:
    """Vocabulary terms that appear in `text`, exactly or as a close variant.

    'Close variant' matters: the input contains the *misspelled* form, so an
    exact-substring test would report Paracetamol as absent from the input and
    then flag the corrected output as a hallucination. Fuzzy membership keeps
    the check honest in both directions.
    """
    text = _nfc(text)
    tokens = _TOKEN_RE.findall(text)
    found: set[str] = set()

    for term in VOCAB:
        if term in text:
            found.add(term)
            continue
        # Compare against same-width windows, matching lexicon_pass, so a
        # two-word symptom is tested against two-word spans and not against
        # single tokens it could never resemble.
        width = len(term.split())
        for start in range(0, max(len(tokens) - width + 1, 0)):
            window = " ".join(tokens[start:start + width])
            if width == 1 and len(window) < MIN_TOKEN_LEN:
                continue
            if _similarity(window, term) >= _threshold_for(width):
                found.add(term)
                break

    return found


def _verify(original: str, candidate: str) -> tuple[bool, Optional[str]]:
    """Gate the LLM's output. Returns (accepted, rejection_reason)."""
    if not candidate.strip():
        return False, "empty output"

    # (a) Did it rewrite rather than correct?
    ratio = _edit_distance(_nfc(original), _nfc(candidate)) / max(len(_nfc(original)), 1)
    if ratio > MAX_REWRITE_RATIO:
        return False, (f"rewrote {ratio:.0%} of the text (cap {MAX_REWRITE_RATIO:.0%}) "
                       f"— treated as composition, not correction")

    # (b) Did it invent a medical term that was never in the input?
    invented = _medical_terms_present(candidate) - _medical_terms_present(original)
    if invented:
        return False, f"introduced medical term(s) absent from input: {', '.join(sorted(invented))}"

    return True, None


# --- Stage 2: LLM ----------------------------------------------------------

_PROMPT = """You are correcting the SPELLING of a Bangla speech-to-text transcript from a medical consultation. The speech recognizer heard the words correctly but spelled some of them wrong.

Typical errors you must fix:
- collapsed conjuncts, e.g. বেথা should be ব্যথা
- spurious ও-কার, e.g. ধোরে should be ধরে
- mangled medicine names, e.g. পেরাসিটমো should be প্যারাসিটামল
- words wrongly split or joined

Medical vocabulary that may appear (use these exact spellings if you recognize a garbled version of one):
{vocab}

The transcript to correct:
\"\"\"
{text}
\"\"\"

HARD RULES — breaking any of these makes your answer useless:
1. Correct SPELLING ONLY. Do not rephrase, translate, summarize, expand, or explain.
2. Do NOT add any medicine, symptom, dosage, or test that is not already spoken in the transcript. If you are unsure what a garbled word is, LEAVE IT EXACTLY AS IT IS.
3. Keep English words the speaker used in Latin script (e.g. "500mg", "BP", "test").
4. Keep the same number of words wherever possible. Never invent content to fill a gap.
5. Output Bangla in Bangla script.

Return ONLY this JSON object, no prose and no markdown fences:
{{"corrected": "...", "uncertain": ["any word you were not confident about"]}}"""


def _llm_pass(text: str, model: str) -> tuple[Optional[str], list[str], Optional[str]]:
    """Run the LLM correction. Returns (corrected_or_None, uncertain_words, error)."""
    vocab_lines = "\n".join(f"- {term} ({kind})" for term, kind in sorted(VOCAB.items()))
    prompt = _PROMPT.format(vocab=vocab_lines, text=text)

    try:
        raw = _generate(model, prompt)
    except Exception as e:
        return None, [], f"ollama call failed: {e}"

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None, [], "no JSON object in model output"
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, [], f"unparseable JSON: {e}"

    corrected = str(parsed.get("corrected", "")).strip()
    uncertain = [str(u) for u in parsed.get("uncertain", []) if str(u).strip()]
    return (corrected or None), uncertain, None


# --- Public entry point ----------------------------------------------------

def correct(text: str, use_llm: bool = True) -> dict:
    """Correct a Bangla/Banglish medical transcript.

    Always runs the lexicon pass. Runs the LLM pass on top only if Ollama is up
    and `use_llm` is set, and only keeps its output if verification passes.

    The return value deliberately exposes HOW the result was reached — which
    stage produced it, what changed, and what was rejected — because a
    correction you can't audit is not usable in a clinical pipeline.
    """
    raw = _nfc(text or "").strip()
    if not raw:
        return {
            "raw": "", "corrected": "", "method": "empty",
            "changes": [], "flags": [], "uncertain": [],
            "llm_model": None, "llm_rejected": None,
        }

    lex_text, changes, flags = lexicon_pass(raw)

    result = {
        "raw": raw,
        "corrected": lex_text,
        "method": "lexicon" if changes else "unchanged",
        "changes": changes,
        "flags": flags,
        "uncertain": [],
        "llm_model": None,
        "llm_rejected": None,
    }

    if not use_llm:
        return result

    model = pick_model()
    if model is None:
        result["flags"] = flags + [{
            "token": None,
            "reason": "llm_unavailable",
            "detail": "Ollama is not reachable or has no model installed — "
                      "lexicon-only result returned",
        }]
        return result

    result["llm_model"] = model

    # The LLM sees the lexicon-corrected text, not the raw text: the drug names
    # are already fixed deterministically, so the LLM's remaining job is the
    # ordinary-word spelling it is actually good at.
    llm_text, uncertain, error = _llm_pass(lex_text, model)

    if error is not None:
        result["llm_rejected"] = error
        return result
    if llm_text is None:
        result["llm_rejected"] = "model returned no correction"
        return result

    accepted, reason = _verify(lex_text, llm_text)
    if not accepted:
        result["llm_rejected"] = reason
        result["flags"] = result["flags"] + [{
            "token": None,
            "reason": "llm_output_rejected",
            "detail": reason,
        }]
        return result

    if _nfc(llm_text) != _nfc(lex_text):
        result["changes"] = changes + [{
            "from": lex_text,
            "to": llm_text,
            "kind": "orthography",
            "stage": "llm",
            "confidence": round(_similarity(lex_text, llm_text), 3),
        }]
        result["method"] = "lexicon+llm" if changes else "llm"

    result["corrected"] = llm_text
    result["uncertain"] = uncertain
    return result
