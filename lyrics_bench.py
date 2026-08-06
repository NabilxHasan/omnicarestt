"""Turn a lyrics YouTube video into (audio_chunk, caption) training pairs.

Design
------
1. Fetch audio + creator-uploaded captions with yt-dlp. We deliberately DON'T
   fall back to auto-captions — Bangla auto-captions have 30-50% WER themselves
   and would poison our "ground truth" for training.
2. Parse the caption file (VTT or SRT) into (start_s, end_s, text) triples.
3. Slice the audio WAV into one clip per caption line using ffmpeg.
   Skip clips shorter than 1 s or longer than 25 s (Whisper's practical limit).
4. Run our current STT on each clip so the reviewer can see model vs truth
   side by side. Reviewer accepts/rejects each pair; accepted ones become
   training data.

Note on "ground truth" quality
------------------------------
Even manual captions can be:
  - loose paraphrases of the actual sung lyrics
  - romanized instead of Bangla script
  - timed to the karaoke bounce, not the actual vocal onset
The accept/reject step is what filters these out — never trust a caption
just because it exists.
"""
from __future__ import annotations

import re
import subprocess
import json
from pathlib import Path
from typing import Optional

import config
from yt_bench import extract_video_id, get_yt_dlp_cmd, get_ffmpeg_path


# --- Caption parsing ----------------------------------------------------

# VTT timestamp: "00:00:12.340 --> 00:00:15.680"
# SRT timestamp: "00:00:12,340 --> 00:00:15,680"
_TS_RE = re.compile(
    r"(\d+):(\d+):([\d.,]+)\s*-->\s*(\d+):(\d+):([\d.,]+)"
)


def _parse_ts(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def parse_captions(caption_file: Path) -> list[tuple[float, float, str]]:
    """Parse VTT or SRT into (start_s, end_s, text) triples.

    Consecutive blocks with the same text are merged — YouTube's rolling
    captions repeat each line across many overlapping windows, which would
    otherwise give us N duplicate chunks per line.
    """
    text = caption_file.read_text(encoding="utf-8", errors="replace")
    out: list[tuple[float, float, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _TS_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start = _parse_ts(m.group(1), m.group(2), m.group(3))
        end   = _parse_ts(m.group(4), m.group(5), m.group(6))
        i += 1
        # Collect text lines until the next blank line or timestamp.
        body = []
        while i < len(lines) and lines[i].strip() and not _TS_RE.search(lines[i]):
            body.append(lines[i].strip())
            i += 1
        clean = " ".join(body).strip()
        if clean:
            out.append((start, end, clean))

    # Merge consecutive identical lines
    merged: list[tuple[float, float, str]] = []
    for start, end, txt in out:
        if merged and merged[-1][2] == txt:
            merged[-1] = (merged[-1][0], end, txt)  # extend end
        else:
            merged.append((start, end, txt))
    return merged


# --- yt-dlp: audio + captions -----------------------------------------

def fetch_audio_and_captions(url: str, out_dir: Path) -> tuple[Path, Optional[Path], dict]:
    """Download audio (16 kHz mono WAV) + creator captions (VTT).

    Returns (wav_path, captions_path_or_None, metadata).
    metadata includes:
      - caption_source: 'manual' | 'auto' | None  (None if no captions of either kind)
      - description: full video description text (used as lyric-bank context for LLM cleanup)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = extract_video_id(url) or "unknown"
    stem = out_dir / f"yt_{video_id}"
    wav = Path(f"{stem}.wav")
    base_cmd = get_yt_dlp_cmd()

    subprocess.run(
        base_cmd + [
            "-x", "--audio-format", "wav",
            "--postprocessor-args", "-ar 16000 -ac 1",
            "--write-subs", "--write-auto-subs",
            "--sub-langs", "bn,bn-BD,bn-IN",
            "--convert-subs", "vtt",
            "--extractor-args", "youtube:player_client=default,web,mweb,ios",
            "-o", f"{stem}.%(ext)s",
            "--no-playlist",
            "--quiet", "--no-warnings",
            url,
        ],
        check=True,
    )

    manual_path: Optional[Path] = None
    auto_path: Optional[Path] = None
    for candidate in sorted(out_dir.glob(f"yt_{video_id}.*.vtt")):
        head = candidate.read_text(encoding="utf-8", errors="replace")[:200]
        if "Kind: captions" in head:
            auto_path = candidate
        else:
            manual_path = candidate

    captions_path = manual_path or auto_path
    caption_source = "manual" if manual_path else ("auto" if auto_path else None)

    meta_json = subprocess.run(
        base_cmd + [
            "-j", "--skip-download", "--no-playlist",
            "--extractor-args", "youtube:player_client=default,web,mweb,ios",
            "--quiet", "--no-warnings", url
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    meta = json.loads(meta_json)
    return wav, captions_path, {
        "video_id": video_id,
        "title": meta.get("title", ""),
        "duration": meta.get("duration", 0),
        "uploader": meta.get("uploader", ""),
        "description": meta.get("description", "") or "",
        "caption_source": caption_source,
    }


# --- Chunk WAV by caption timestamps -----------------------------------

MIN_CHUNK_S = 1.0
MAX_CHUNK_S = 25.0  # Whisper hits its 30 s window; leave headroom


def chunk_audio(wav: Path, captions: list[tuple[float, float, str]],
                out_dir: Path, video_id: str) -> list[dict]:
    """Slice `wav` into per-caption clips. Returns [{path, start, end, text}...]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    ffmpeg_exe = get_ffmpeg_path()

    for idx, (start, end, text) in enumerate(captions):
        dur = end - start
        if dur < MIN_CHUNK_S or dur > MAX_CHUNK_S:
            continue
        chunk_path = out_dir / f"{video_id}_c{idx:04d}.wav"
        if not chunk_path.exists():
            subprocess.run(
                [ffmpeg_exe, "-y", "-loglevel", "error",
                 "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                 "-i", str(wav),
                 "-ac", "1", "-ar", "16000",
                 str(chunk_path)],
                check=True,
            )
        chunks.append({
            "chunk_id": f"{video_id}_c{idx:04d}",
            "path": str(chunk_path),
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "duration_s": round(dur, 3),
            "caption": text,
        })
    return chunks
