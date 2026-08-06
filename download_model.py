"""Download a Hugging Face model into models/ and print the config.py edit needed.

  python download_model.py                      # default: the tugstugi regional model
  python download_model.py <hf_repo_id>         # any other HF repo
  python download_model.py --list               # show suggested repos

The download is a full snapshot (safetensors + tokenizer + configs). Size is
~3 GB for a Whisper-medium checkpoint, one-time.

Why this is a script and not just `git lfs clone`
-------------------------------------------------
`huggingface_hub.snapshot_download` handles the resume-on-failure, checksum,
and shared cache dedup that plain git lfs doesn't. It also caches under
~/.cache/huggingface/, so re-downloading the same model from another script
is instant. We symlink from models/ into that cache — the model directory
looks local to the project but the bytes live in one place on disk.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import config


SUGGESTED = {
    "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium":
        "Regional dialects (Chatgaya, Sylheti, Barishali, Rangpuri, Noakhali). "
        "981 downloads — the canonical regional model. Same author as your current base.",
    "bengaliAI/tugstugi_bengaliai-asr_whisper-medium":
        "General Bangladeshi Bangla — this is your CURRENT model. Included for reference.",
    "IamSanjid/tugstugi_bengaliai-regional-asr_whisper-medium-ct2":
        "CTranslate2 conversion of the regional model — 3-5x faster inference "
        "with faster-whisper. Different runtime, would need engine.py changes.",
}

DEFAULT_REPO = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"


def _print_suggestions() -> None:
    print("Suggested repos:\n")
    for repo, why in SUGGESTED.items():
        print(f"  {repo}")
        print(f"    {why}\n")


def _local_dir_for(repo_id: str) -> Path:
    """Turn `owner/name` into a short local folder name under models/."""
    slug = repo_id.split("/")[-1]
    return config.MODELS_DIR / slug


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default=DEFAULT_REPO,
                    help=f"HF repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--list", action="store_true",
                    help="Show suggested repos and exit.")
    args = ap.parse_args()

    if args.list:
        _print_suggestions()
        return 0

    # Import inside main so `--list` and `--help` don't pull huggingface_hub.
    from huggingface_hub import snapshot_download

    dest = _local_dir_for(args.repo)
    if dest.exists():
        print(f"Already present: {dest}")
        print("Delete it first if you want to re-download.")
    else:
        print(f"Downloading {args.repo} ...")
        cached_path = snapshot_download(repo_id=args.repo)
        # Symlink so the model directory is uniformly under models/, but the
        # bytes stay in the HF cache and are shared with any other tool that
        # downloads the same repo.
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(cached_path)
        print(f"Symlinked: {dest} -> {cached_path}")

    slug = dest.name
    print()
    print("=" * 68)
    print("To make the project use this model, edit config.py:")
    print()
    print(f"    STT_MODEL_DIR  = MODELS_DIR / '{slug}'")
    print(f"    STT_MODEL_NAME = '{slug}'")
    print()
    print("Then verify:")
    print("    python evaluate.py --fleurs-only --n 5")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
