"""Batch-transcribe a list of YouTube videos and produce an HTML report.

  python yt_bench.py <URL_1> <URL_2> ...       # inline
  python yt_bench.py -f urls.txt               # from file (one URL per line, # comments ok)
  python yt_bench.py -f urls.txt -o report.html

The report is a single HTML file with one card per video:
  [ embedded YouTube player ]   [ transcript ]
                                [ URL + timing + model name ]

Open the report in a browser, click play on any video, read the transcript
next to it, and eyeball how accurate the STT is. That's the whole workflow.

The script talks to the running API server (default http://localhost:8000)
so the model stays loaded in one process. Start the server first:
    ./run_server.sh &
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

import shutil
import sys
import config

# --- Video ID extraction -------------------------------------------------

_YT_ID_RE = re.compile(
    r"(?:(?:[a-zA-Z0-9-]+\.)?youtube\.com/(?:shorts/|watch\?v=|embed/|v/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    m = _YT_ID_RE.search(url.strip())
    return m.group(1) if m else None


def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def get_yt_dlp_cmd() -> list[str]:
    yt_bin = shutil.which("yt-dlp")
    if yt_bin:
        cmd = [yt_bin]
    else:
        cmd = [sys.executable, "-m", "yt_dlp"]
    
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path:
        cmd.extend(["--ffmpeg-location", str(ffmpeg_path)])
    return cmd


# --- yt-dlp wrapper ------------------------------------------------------

def download_audio(url: str, out_dir: Path) -> tuple[Path, dict]:
    """Download audio + metadata for a single URL. Returns (wav_path, metadata)."""
    video_id = extract_video_id(url) or "unknown"
    wav = out_dir / f"yt_{video_id}.wav"
    base_cmd = get_yt_dlp_cmd()

    if not wav.exists():
        subprocess.run(
            base_cmd + [
                "-x",
                "--audio-format", "wav",
                "--postprocessor-args", "-ar 16000 -ac 1",
                "--extractor-args", "youtube:player_client=default,web,mweb,ios",
                "-o", str(out_dir / f"yt_{video_id}.%(ext)s"),
                "--no-playlist",
                "--quiet", "--no-warnings",
                url,
            ],
            check=True,
        )

    # Grab title + duration for the report. --skip-download makes this fast.
    meta_json = subprocess.run(
        base_cmd + [
            "-j", "--skip-download", "--no-playlist",
            "--quiet", "--no-warnings", url,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    meta = json.loads(meta_json)
    return wav, {
        "video_id": video_id,
        "title": meta.get("title", "(no title)"),
        "duration": meta.get("duration", 0),
        "uploader": meta.get("uploader", ""),
    }


# --- STT via running API ------------------------------------------------

def transcribe_via_api(wav: Path, base_url: str) -> tuple[str, float, str]:
    """POST audio to the running server. Returns (text, server_seconds, model_name)."""
    with wav.open("rb") as f:
        resp = requests.post(
            f"{base_url}/transcribe",
            files={"audio": (wav.name, f, "audio/wav")},
            timeout=600,
        )
    resp.raise_for_status()
    d = resp.json()
    return d["text"], float(d["duration_s"]), d["model"]


# --- HTML report --------------------------------------------------------

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>STT bench — __N__ videos — __TS__</title>
<style>
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body {
        font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 1.5rem 1rem;
        line-height: 1.6;
    }
    h1 { margin: 0 0 0.25rem; font-size: 1.4rem; }
    .summary {
        margin: 0 0 1.5rem;
        color: #888;
        font-size: 0.9rem;
    }
    .card {
        display: grid;
        grid-template-columns: 340px 1fr;
        gap: 1.25rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
    }
    @media (max-width: 800px) {
        .card { grid-template-columns: 1fr; }
    }
    .video iframe {
        width: 100%;
        aspect-ratio: 9/16;
        border: 0;
        border-radius: 8px;
        background: #000;
    }
    .video .title {
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #aaa;
        line-height: 1.3;
    }
    .video .title a { color: inherit; }
    .transcript {
        display: flex;
        flex-direction: column;
        min-width: 0;
    }
    .text {
        flex: 1;
        padding: 1rem;
        background: rgba(128,128,128,0.08);
        border-radius: 8px;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 1.05rem;
        min-height: 8rem;
    }
    .text.error { background: rgba(220,50,50,0.1); color: #d33; }
    .meta {
        margin-top: 0.75rem;
        font-size: 0.8rem;
        color: #888;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1rem;
    }
    .meta code {
        background: rgba(128,128,128,0.15);
        padding: 1px 6px;
        border-radius: 4px;
    }
    .card-num {
        display: inline-block;
        min-width: 1.5rem;
        color: #888;
        font-variant-numeric: tabular-nums;
    }
</style>
</head>
<body>
<h1>STT benchmark</h1>
<div class="summary">
    __N__ videos · model: <code>__MODEL__</code> · generated __TS__
</div>
"""

HTML_FOOT = "</body></html>\n"


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m, s = divmod(int(s), 60)
    return f"{m}:{s:02d}"


def render_card(idx: int, url: str, meta: dict, text: str, server_s: float, error: str | None) -> str:
    vid = meta.get("video_id", "")
    embed = f"https://www.youtube.com/embed/{vid}" if vid else ""
    text_html = (text or "(empty transcript)").replace("<", "&lt;").replace(">", "&gt;")
    title_html = meta.get("title", "").replace("<", "&lt;").replace(">", "&gt;")
    text_cls = "text error" if error else "text"
    text_content = f"[Error] {error}" if error else text_html
    duration_s = meta.get("duration", 0) or 0
    return f"""
<div class="card">
    <div class="video">
        <iframe src="{embed}" allowfullscreen loading="lazy"></iframe>
        <div class="title"><a href="{url}" target="_blank" rel="noopener">{title_html}</a></div>
    </div>
    <div class="transcript">
        <div class="{text_cls}">{text_content}</div>
        <div class="meta">
            <span><span class="card-num">#{idx}</span> · <code>{vid}</code></span>
            <span>video: {_fmt_secs(duration_s)}</span>
            <span>server: {server_s:.1f}s</span>
            <span>uploader: {meta.get("uploader", "?")}</span>
            <span><a href="{url}" target="_blank" rel="noopener">open on YouTube ↗</a></span>
        </div>
    </div>
</div>
"""


# --- URL list parsing ---------------------------------------------------

def read_urls(args_urls: list[str], file_path: Path | None) -> list[str]:
    urls: list[str] = list(args_urls)
    if file_path:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    # De-dup while preserving order.
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --- Main --------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="*", help="YouTube URLs (inline)")
    ap.add_argument("-f", "--file", type=Path, help="File with one URL per line (# comments ok)")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path(f"bench/report_{datetime.now():%Y%m%d_%H%M%S}.html"),
                    help="Output HTML path")
    ap.add_argument("--server", default="http://localhost:8000",
                    help="Base URL of running STT API")
    args = ap.parse_args()

    urls = read_urls(args.urls, args.file)
    if not urls:
        print("No URLs provided. Pass URLs as args or with -f urls.txt", file=sys.stderr)
        return 2

    # Where downloaded audio lands. Reuses existing files, so re-runs are cheap.
    audio_dir = config.AUDIO_DIR / "yt_bench"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Sanity-check the server is up.
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        r.raise_for_status()
        health = r.json()
        model = health.get("model", "?")
    except Exception as e:
        print(f"Server at {args.server} not reachable: {e}", file=sys.stderr)
        print("Start it first:  ./run_server.sh", file=sys.stderr)
        return 1

    print(f"Server OK. Model: {model}")
    print(f"Processing {len(urls)} URLs. Downloads → {audio_dir}")

    cards: list[str] = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            wav, meta = download_audio(url, audio_dir)
        except subprocess.CalledProcessError as e:
            cards.append(render_card(i, url, {"video_id": extract_video_id(url) or "", "title": "(download failed)"},
                                     "", 0.0, f"yt-dlp exit {e.returncode}"))
            continue
        except Exception as e:
            cards.append(render_card(i, url, {"video_id": extract_video_id(url) or "", "title": "(metadata failed)"},
                                     "", 0.0, str(e)))
            continue

        try:
            text, secs, _model = transcribe_via_api(wav, args.server)
            print(f"    ✓ {secs:.1f}s  {text[:80]}{'...' if len(text) > 80 else ''}")
        except Exception as e:
            text, secs = "", 0.0
            print(f"    ✗ transcribe failed: {e}")
            cards.append(render_card(i, url, meta, "", 0.0, str(e)))
            continue

        cards.append(render_card(i, url, meta, text, secs, None))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = (
        HTML_HEAD.replace("__N__", str(len(urls)))
                 .replace("__MODEL__", model)
                 .replace("__TS__", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        + "".join(cards)
        + HTML_FOOT
    )
    args.out.write_text(html, encoding="utf-8")
    print(f"\nReport: {args.out.resolve()}")
    print(f"Open with:  open '{args.out.resolve()}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
