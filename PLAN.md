# Omnicare STT — clean rebuild plan

## What we're building (and why, in one paragraph)

Omnicare needs a Bangla speech-to-text stage good enough for medical
conversations. In earlier experiments we learned three things that shape the
whole design:

1. **Model choice matters more than fine-tuning.** The Bengali.AI Kaggle
   competition winner (`bengaliai-asr-whisper-medium`, ~1200h Bangladeshi
   training data) hit **13.2% WER / 3.2% CER** on FLEURS and **3.1% / 0.6%** on
   synthetic medical audio. Raw Whisper-large-v3-turbo was 5-10× worse. So the
   base engine is decided — we vendor and use this model.
2. **Errors are orthographic, not acoustic.** The model hears correctly and
   spells wrong (conjunct collapses like `ব্যথা→বেথা`, spurious ও-কার like
   `ধরে→ধোরে`, mangled loanwords like `প্যারাসিটামল→পেরাসিটমো`). The right
   fix is a better-spelling model, not a downstream patch that guesses.
3. **LLM post-correction is off the table for the main build.** In earlier
   tests, an LLM given a medical glossary confidently mapped garbled audio to
   the wrong medical term. In clinical use, a *confident wrong word* is worse
   than *visible garbage* — a doctor spots the latter, silently trusts the
   former. So the rebuild does not depend on any LLM cleanup layer. If the
   baseline isn't good enough, we fix it at the source (fine-tune the STT
   model), not by adding a second model that can hallucinate.

The rebuild is a small, layered, single-purpose set of files. Each layer works
alone and can be understood before the next is written. No file does two things.
No LLM in the serving path — one model, one output, one thing to reason about.

---

## The layers (build in this order — each depends on the ones above)

```
Layer 0 — foundation
  ├─ config.py               single source of truth for paths, model IDs, thresholds
  └─ bnorm.py                Bangla text normalization (for fair WER/CER comparison)

Layer 1 — the STT engine (one model, one wrapper)
  └─ engine.py               load bengaliai model once, transcribe(wav_path) -> text

Layer 2 — evaluation (know how good we are before touching anything else)
  ├─ eval_data.py            build a small local eval set (FLEURS + your medical samples)
  ├─ metrics.py              WER and CER (word / char Levenshtein), using bnorm
  └─ evaluate.py             run engine on eval set, print per-file + aggregate

Layer 3 — serving (the layer Omnicare app actually calls)
  ├─ api.py                  FastAPI: POST /transcribe (audio file) -> {text, duration_s, model}
  └─ run_server.sh           uvicorn launcher (dev + prod flags documented inside)

Layer 4 — integration hook
  └─ client_example.py       reference: how the Railway app calls the API

Layer 5 — improvement at the source (only if Layer 2 shows we need it)
  └─ finetune_kaggle.ipynb   LoRA fine-tune on Bengali.AI + medical data (Kaggle notebook)
```

**Explicitly not in this folder** (and why):
- **No LLM post-correction.** Hallucination risk on medical terms outweighs
  the WER it could shave off. If we ever want to revisit, it lives as a
  separate optional service — see "Deferred experiments" below.
- **No TTS.** Unsolved licensing gap — separate problem.
- **No RAG, no chat UI, no Railway deployment.** This folder is the STT
  service only.

---

## Files, one at a time — what each does and why

### Layer 0

**`config.py`** — every path, model ID, and threshold in one place. Why:
in the old lab we had model paths hardcoded in 4 files; changing the model
meant a grep-and-pray. Here, one constant.

**`bnorm.py`** — Bangla text normalization. Why: raw WER on Bangla is
misleading because `খাবারগুলিতে` vs `খাবার গুলিতে` reads identically but
counts as 2 wrong words. Also normalizes punctuation (দাঁড়ি ।), NFC unicode,
digit forms (১২ vs 12), and collapses whitespace. Every WER/CER comparison
goes through this first.

### Layer 1

**`engine.py`** — the whole STT surface. Loads the vendored bengaliai model
**once** (module-level, lazy), exposes `transcribe(audio_path) -> str`.
No CLI, no batching, no post-processing — just clean audio-in / text-out.
Everything else in the system talks through this one function. Why: if we
later swap the model or add streaming, only this file changes.

### Layer 2 — evaluation

You need this **before** any change (fine-tuning included), so you can tell if
a change helped or hurt.

**`eval_data.py`** — pulls FLEURS bn_in samples and optionally reads your
`audio/medical/*.wav` folder with matching `.reference.txt` files. Returns a
list of `(audio_path, reference_text)`. Why separate: eval sets grow — you'll
add real recordings later, so this is the one place to register them.

**`metrics.py`** — WER (word-level Levenshtein / ref length) and CER
(character-level). Applies `bnorm` to both sides before comparing. Why
separate from `evaluate.py`: the fine-tune notebook re-uses these to score
its own checkpoints.

**`evaluate.py`** — glues it all: for each `(audio, ref)` pair, run
`engine.transcribe`, compute WER + CER, print table + aggregate. The one
command you run to answer "is this any good?".

### Layer 3 — serving

**`api.py`** — FastAPI app, one endpoint: `POST /transcribe`
(multipart audio upload → JSON `{text, duration_s, model}`).
Loads engine once at startup. Why FastAPI: it's what the Railway app already
speaks; it gives us OpenAPI docs for free; async support is there if we later
need streaming.

**`run_server.sh`** — `uvicorn api:app --host 0.0.0.0 --port 8000`, with
comments explaining `--workers` and GPU-serve flags for when we move this to
Modal/RunPod. Why a script, not a bare command: prod deploy config lives in
one file, not in someone's terminal history.

### Layer 4

**`client_example.py`** — a 20-line Python script showing how the Omnicare
Railway app calls the STT API. Not part of the service — it's a *reference*
for the app-side dev (you) so integration doesn't guess at the contract.

### Layer 5 — improvement at the source (conditional)

**`finetune_kaggle.ipynb`** — Kaggle notebook that LoRA-fine-tunes the
bengaliai model on (Bengali.AI Speech + synthetic medical). Only built if the
Layer 2 numbers say the baseline isn't good enough. Kaggle-hosted so the
1200h Bengali.AI dataset never leaves their disk (no download, no storage
cost). The fine-tuned checkpoint drops back into `models/`, and `engine.py`
is pointed at it with a one-line change in `config.py`. No other file
changes — that's the whole reason `config.py` and `engine.py` exist.

---

## Model / dependency decisions (locked)

| decision              | value                                                    | why |
|-----------------------|----------------------------------------------------------|-----|
| STT model             | `bengaliai-asr-whisper-medium` (vendored locally)        | Winner on both FLEURS and medical eval; 4× lower WER than base Whisper |
| Runtime               | `faster-whisper` (CTranslate2 int8)                      | 3-5× faster than HF `transformers`, runs CPU fine, ready for GPU when we move to Modal |
| Post-correction       | **None in serving path** (deferred experiment)           | Hallucination risk in medical > WER improvement it offers |
| Serving framework     | FastAPI + uvicorn                                        | Matches the Railway app; async-ready |
| Eval dataset          | FLEURS bn_in (~40 clips) + your `audio/medical/*`        | FLEURS = general, medical = the actual use case |
| Text metric           | WER + CER, both bnorm-normalized                         | CER is the honest number for Bangla (WER inflated by segmentation) |
| Fine-tune platform    | Kaggle free tier (T4×2) with Bengali.AI Speech           | Dataset is hosted natively there — zero data transfer |

---

## Folder layout when finished

```
new-stt/
├── PLAN.md                    (this file)
├── config.py
├── bnorm.py
├── engine.py
├── eval_data.py
├── metrics.py
├── evaluate.py
├── api.py
├── run_server.sh
├── client_example.py
├── finetune_kaggle.ipynb      (only if Layer 2 says we need it)
├── audio/
│   ├── eval_fleurs/           (downloaded FLEURS samples + .reference.txt)
│   └── medical/               (drop your own medical .wav + .reference.txt here)
├── models/
│   └── bengaliai-asr-whisper-medium/   (symlinked or copied from bangla-stt/models/)
└── requirements.txt
```

---

## What "done" looks like at each layer

- **Layer 0 done**: `python -c "from bnorm import normalize; print(normalize('খাবার গুলিতে ।'))"` returns the joined, punctuation-stripped form.
- **Layer 1 done**: `python -c "from engine import transcribe; print(transcribe('audio/eval_fleurs/fleurs_bn_00.wav'))"` prints Bangla text in <5s.
- **Layer 2 done**: `python evaluate.py` prints a table with per-file WER/CER + aggregate, within a few points of the baseline in memory (~13% WER FLEURS, ~3% medical).
- **Layer 3 done**: `curl -F audio=@somefile.wav http://localhost:8000/transcribe` returns JSON with the transcript.
- **Layer 4 done**: `client_example.py` runs end-to-end against a running server.
- **Layer 5 done** (if built): notebook runs on Kaggle, produces a checkpoint that beats the baseline on our eval set.

---

## The build sequence you'll follow

Each of your next prompts = "build the next file, explain it as you go".
Order is fixed by dependencies:

1. `config.py`
2. `bnorm.py`
3. `engine.py`
4. `eval_data.py`
5. `metrics.py`
6. `evaluate.py`   ← **first checkpoint: baseline is measured. Stop and read the numbers.**
7. `api.py`
8. `run_server.sh`
9. `client_example.py`  ← **second checkpoint: Omnicare can call the service.**
10. `finetune_kaggle.ipynb`  ← **only built if checkpoint 1 said the baseline is too weak for medical use.**

The plan is deliberately conservative: 9 files get us a working, deployable,
one-model STT service with no LLM in the loop. The 10th file only exists if
the data demands it.

---

## Deferred experiments (not part of the main build)

These are ideas we've explicitly *not* included in the main service. They're
listed here so we don't forget them, and so future us knows they were
considered and rejected on purpose.

- **LLM post-correction with strict guardrails.** Would only be reconsidered
  if (a) Layer 5 fine-tuning is exhausted and the medical WER is still too
  high, and (b) we can prove on our eval set that guardrails (edit-distance
  cap, closed-vocab whitelist) prevent every hallucination case. Even then it
  runs as a *separate* optional service, not in the STT path.
- **Streaming / partial transcripts.** Nice for live conversations but adds a
  lot of complexity. Wait until the batch API is proven in production.
- **Speaker diarization.** Useful for multi-party consults. Separate concern,
  separate model, not in this folder.
