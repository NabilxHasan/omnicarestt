"""Run the engine over the eval set, print per-clip and corpus WER/CER.

  python evaluate.py                    # all sources, default FLEURS N
  python evaluate.py --fleurs-only
  python evaluate.py --medical-only
  python evaluate.py --n 20             # limit FLEURS to 20 clips (quick smoke)

This is the one command that answers "is the STT any good?". It should be run
after every change that could affect quality: swapping the model, converting to
CTranslate2, deploying to a GPU box, or (later) a fine-tuned checkpoint.

What it prints
--------------
  * Per-clip table   — filename, source, seconds-to-transcribe, WER%, CER%.
                       Sorted worst-first so bad clips are the first thing you see.
  * Aggregate per source (fleurs vs medical) — where the model actually stands.
  * Overall corpus  — the single number to remember for the run.

Transcripts are written next to the audio as `<name>.hyp.txt` so you can diff
against the reference later without re-transcribing.
"""
from __future__ import annotations

import argparse
import time
from math import isnan

import config
import eval_data
from engine import transcribe
from metrics import corpus_scores, sentence_wer, sentence_cer


def _fmt_pct(x: float) -> str:
    return "  n/a " if isnan(x) else f"{x*100:5.1f}%"


def _run_one(item: eval_data.EvalItem) -> tuple[str, float, float, float]:
    """Transcribe one clip, save the hypothesis, return (hyp, seconds, wer, cer)."""
    t0 = time.time()
    hyp = transcribe(item.audio)
    secs = time.time() - t0

    # Persist the hypothesis alongside the audio for later diffing / debugging.
    (item.audio.with_suffix(".hyp.txt")).write_text(hyp, encoding="utf-8")

    return hyp, secs, sentence_wer(item.reference, hyp), sentence_cer(item.reference, hyp)


def _print_table(rows: list[dict]) -> None:
    """Per-clip table, sorted worst-first (highest WER at top)."""
    rows = sorted(rows, key=lambda r: (-1 if isnan(r["wer"]) else -r["wer"]))
    print(f"\n{'file':<24} {'source':<8} {'sec':>6}  {'WER':>7} {'CER':>7}")
    print("-" * 60)
    for r in rows:
        print(f"{r['file']:<24} {r['source']:<8} {r['sec']:>6.1f}  "
              f"{_fmt_pct(r['wer'])} {_fmt_pct(r['cer'])}")


def _print_corpus(label: str, pairs: list[tuple[str, str]]) -> None:
    s = corpus_scores(pairs)
    print(f"  {label:<12}  n={s['n']:<3}  WER={_fmt_pct(s['wer'])}  CER={_fmt_pct(s['cer'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs-only",  action="store_true")
    ap.add_argument("--medical-only", action="store_true")
    ap.add_argument("--n", type=int, default=config.FLEURS_SAMPLE_COUNT,
                    help="Max FLEURS clips (default from config).")
    args = ap.parse_args()

    if args.fleurs_only and args.medical_only:
        print("Pass only one of --fleurs-only / --medical-only.")
        return 2

    if args.medical_only:
        items = eval_data.load_medical()
    elif args.fleurs_only:
        items = eval_data.load_fleurs(args.n)
    else:
        items = eval_data.load_all(args.n)

    if not items:
        print("No eval items found. If --medical-only, add .wav + .reference.txt pairs")
        print(f"to {config.MEDICAL_DIR}. Otherwise the FLEURS download may have failed.")
        return 1

    print(f"Evaluating {len(items)} clips with model: {config.STT_MODEL_NAME}")

    rows, all_pairs = [], []
    per_source: dict[str, list[tuple[str, str]]] = {}
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item.audio.name}", end="", flush=True)
        hyp, secs, wer, cer = _run_one(item)
        print(f"   {secs:.1f}s   WER {_fmt_pct(wer)}  CER {_fmt_pct(cer)}")
        rows.append(dict(file=item.audio.name, source=item.source,
                         sec=secs, wer=wer, cer=cer))
        all_pairs.append((item.reference, hyp))
        per_source.setdefault(item.source, []).append((item.reference, hyp))

    _print_table(rows)

    print("\nCorpus scores (pooled edits / pooled reference tokens):")
    for source in sorted(per_source):
        _print_corpus(source, per_source[source])
    print("-" * 60)
    _print_corpus("OVERALL", all_pairs)
    print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
