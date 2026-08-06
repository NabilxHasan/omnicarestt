"""FastAPI service — the one surface the Omnicare Railway app calls,
plus a local management UI for testing.

Endpoints
---------
  GET  /                — YouTube bench page (paste URLs → transcripts + player)
  GET  /mic             — mic-recording page (talk → transcript)
  GET  /training        — lyrics-video training-data collection page
  GET  /health          — liveness probe
  POST /transcribe      — multipart audio upload → {"text", "duration_s", "model"}
  POST /bench/transcribe  — {"url": "..."} → transcript entry, appends to history
  GET  /bench/history   — list of past transcriptions
  DELETE /bench/history/{video_id}  — remove one entry

  POST /training/from_youtube  — {"url": "..."} → chunk video by captions, return pairs
  GET  /training/manifest      — list of accepted training pairs so far
  POST /training/accept        — {"chunk_id": "...", "final_text": "..."} → save pair
  DELETE /training/manifest/{chunk_id}  — remove a saved pair
  GET  /training/audio/{chunk_id}  — serve one chunk's audio (for the review UI)
  GET  /training/export        — download manifest.csv (ready for Kaggle)

Design
------
- Model warmed at startup via lifespan hook.
- History persists at `bench/history.json`. Newest first. De-duped by video_id.
- Audio cache lives at `audio/yt_bench/yt_<video_id>.wav`. Re-transcribing a
  URL you've already fetched is nearly instant (skips the download).
- Training data: audio chunks in `training/chunks/`, accepted pairs in
  `training/manifest.jsonl` (JSON Lines — easy append, easy filter).
- No auth (behind private network or Modal/RunPod's own auth).
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import engine2
from engine import transcribe
from yt_bench import download_audio, extract_video_id
from lyrics_bench import fetch_audio_and_captions, parse_captions, chunk_audio
from llm_helper import (
    agreement_score,
    align_reference_to_captions,
    clean_captions_batch,
    extract_reference_lines,
    ollama_available,
    pick_model,
)


# --- Dual-engine transcription -----------------------------------------

def transcribe_both(audio: Path | str) -> dict:
    """Run both STT engines on the same audio and report their agreement.

    They run concurrently on purpose: engine 1 is transformers-on-CPU and
    engine 2 is MLX-on-GPU, so the wall clock is max(a, b) rather than a + b.
    Both calls release the GIL inside their native inference loops.

    Either engine failing is recorded as an error string for that engine
    rather than failing the whole request — one working transcript is still
    worth returning.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _run(fn, label):
        t0 = time.time()
        try:
            return {"text": fn(audio), "seconds": round(time.time() - t0, 2), "error": None}
        except Exception as e:
            return {"text": "", "seconds": round(time.time() - t0, 2), "error": f"{label}: {e}"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_run, transcribe, "engine1")
        f2 = pool.submit(_run, engine2.transcribe, "engine2")
        r1, r2 = f1.result(), f2.result()

    return {
        "text": r1["text"],
        "model": config.STT_MODEL_NAME,
        "duration_s": r1["seconds"],
        "error": r1["error"],
        "text2": r2["text"],
        "model2": engine2.MODEL_NAME,
        "duration2_s": r2["seconds"],
        "error2": r2["error"],
        # 1.0 = identical. Low agreement means at least one engine is in its
        # known weak spot — that is the chunk a human should look at.
        "agreement": round(agreement_score(r1["text"], r2["text"]), 3),
    }

HISTORY_PATH: Path = config.PROJECT_ROOT / "bench" / "history.json"
YT_BENCH_AUDIO_DIR: Path = config.AUDIO_DIR / "yt_bench"

TRAINING_DIR: Path = config.PROJECT_ROOT / "training"
TRAINING_CHUNKS_DIR: Path = TRAINING_DIR / "chunks"
TRAINING_MANIFEST: Path = TRAINING_DIR / "manifest.jsonl"
LYRICS_CACHE_DIR: Path = config.AUDIO_DIR / "lyrics_cache"


# --- History persistence -----------------------------------------------

def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_history(items: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --- Startup / shutdown -----------------------------------------------

import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model asynchronously so port 8000 opens instantly."""
    async def _warm():
        for candidate_dir in (config.AUDIO_DIR / "mine", config.EVAL_FLEURS_DIR):
            if not candidate_dir.exists():
                continue
            seed = next(candidate_dir.glob("*.wav"), None)
            if seed is not None:
                break
        else:
            seed = None

        if seed is not None:
            print(f"[api] warming model with {seed.name} ...")
            t0 = time.time()
            try:
                await asyncio.to_thread(transcribe, seed)
                print(f"[api] warm-up done in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"[api] warm-up background notice ({e}); first request will complete model load")

    asyncio.create_task(_warm())
    yield


app = FastAPI(
    title="Omnicare STT",
    version="0.3.0",
    description=f"Bangla speech-to-text service. Backed by {config.STT_MODEL_NAME}.",
    lifespan=lifespan,
)

STATIC_DIR: Path = config.PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- HTML pages (kept in-file so no template dir to manage) ------------

MIC_PAGE = """<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Omnicare STT — Mic test</title>
    <style>
        :root { color-scheme: light dark; }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
            max-width: 720px; margin: 0 auto; padding: 2rem 1.25rem; line-height: 1.5;
        }
        h1 { margin: 0 0 0.25rem; font-size: 1.5rem; }
        .subtitle { color: #888; font-size: 0.85rem; margin: 0 0 1.5rem; }
        .nav { margin-bottom: 1.5rem; }
        .nav a { color: #888; text-decoration: none; margin-right: 1rem; }
        .nav a.active { color: inherit; font-weight: 600; }
        .rec-btn {
            display: block; width: 100%; padding: 1.25rem;
            font-size: 1.15rem; font-weight: 600; border-radius: 12px;
            border: 2px solid #666; background: transparent; color: inherit;
            cursor: pointer; transition: background 0.15s, border-color 0.15s;
        }
        .rec-btn:hover:not(:disabled) { background: rgba(128,128,128,0.1); }
        .rec-btn.recording {
            background: #d33; border-color: #d33; color: white;
            animation: pulse 1.2s ease-in-out infinite;
        }
        .rec-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(211,51,51,0.5); }
            50%      { box-shadow: 0 0 0 12px rgba(211,51,51,0); }
        }
        .status { margin: 1rem 0; padding: 0.75rem 1rem; border-radius: 8px;
            background: rgba(128,128,128,0.1); font-size: 0.9rem; min-height: 2.5rem; }
        .result { margin-top: 1rem; padding: 1rem; border-radius: 8px;
            background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.2);
            min-height: 6rem; font-size: 1.1rem; word-break: break-word; }
        .result:empty::before { content: "Your transcription will appear here..."; color: #888; font-size: 0.9rem; }
        .result .eng { font-size: 0.7rem; color: #888; text-transform: uppercase;
            letter-spacing: 0.05em; margin: 0.75rem 0 0.25rem; }
        .result .eng:first-child { margin-top: 0; }
        .result .out { white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">▶ YouTube bench</a>
        <a href="/mic" class="active">🎙 Mic test</a>
        <a href="/training">🎓 Training data</a>
    </div>
    <h1>Mic test</h1>
    <p class="subtitle">Model: __MODEL_NAME__</p>

    <button id="rec" class="rec-btn">🎙 Start Recording</button>
    <div id="status" class="status">Click the button and grant microphone permission to begin.</div>
    <div id="result" class="result"></div>

<script>
const btn = document.getElementById('rec');
const status = document.getElementById('status');
const result = document.getElementById('result');
let mediaRecorder = null, chunks = [];
function setStatus(m) { status.textContent = m; }

async function start() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        chunks = [];
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            await send(new Blob(chunks, { type: 'audio/webm' }));
        };
        mediaRecorder.start();
        btn.textContent = '⏹ Stop Recording';
        btn.classList.add('recording');
        setStatus('🔴 Recording... speak, then click Stop.');
    } catch (err) { setStatus('Mic denied: ' + err.message); }
}
function stop() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    btn.textContent = '🎙 Start Recording';
    btn.classList.remove('recording');
    setStatus('⏳ Transcribing...');
}
async function send(blob) {
    btn.disabled = true;
    const form = new FormData();
    form.append('audio', blob, 'mic.webm');
    const t0 = performance.now();
    try {
        const resp = await fetch('/transcribe', { method: 'POST', body: form });
        const secs = ((performance.now() - t0) / 1000).toFixed(1);
        if (!resp.ok) { setStatus(`❌ ${resp.status} after ${secs}s: ${await resp.text()}`); return; }
        const data = await resp.json();
        const esc = s => { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; };
        const agree = data.agreement;
        result.innerHTML =
            `<div class="eng">① ${esc(data.model)} · ${data.duration_s}s</div>` +
            `<div class="out">${esc(data.text) || '(empty)'}</div>` +
            (data.text2 !== undefined
                ? `<div class="eng">② ${esc(data.model2)} · ${data.duration2_s}s</div>` +
                  `<div class="out">${esc(data.text2) || '(empty)'}</div>`
                : '');
        const pct = agree === undefined ? '?' : (agree * 100).toFixed(0);
        setStatus(`✅ Done in ${secs}s · engines agree ${pct}%`);
    } catch (err) { setStatus('❌ ' + err.message); }
    finally { btn.disabled = false; }
}
btn.addEventListener('click', () => (mediaRecorder?.state === 'recording') ? stop() : start());
</script>
</body>
</html>
"""


BENCH_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Omnicare STT — YouTube bench</title>
    <style>
        :root { color-scheme: light dark; }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px; margin: 0 auto; padding: 1.5rem 1rem; line-height: 1.5;
        }
        h1 { margin: 0 0 0.25rem; font-size: 1.4rem; }
        .subtitle { color: #888; font-size: 0.9rem; margin: 0 0 1rem; }
        .nav { margin-bottom: 1.5rem; }
        .nav a { color: #888; text-decoration: none; margin-right: 1rem; }
        .nav a.active { color: inherit; font-weight: 600; }
        .submit-panel {
            padding: 1rem; margin-bottom: 1.5rem;
            border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
        }
        textarea {
            width: 100%; min-height: 5rem; padding: 0.75rem;
            font-family: ui-monospace, "SF Mono", monospace; font-size: 0.9rem;
            background: rgba(128,128,128,0.08); color: inherit;
            border: 1px solid rgba(128,128,128,0.25); border-radius: 8px;
            resize: vertical;
        }
        .submit-row { margin-top: 0.75rem; display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
        button.primary {
            padding: 0.6rem 1.25rem; font-size: 0.95rem; font-weight: 600;
            border-radius: 8px; border: 0; background: #2b7cff; color: white;
            cursor: pointer;
        }
        button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
        button.primary:hover:not(:disabled) { background: #1f66d9; }
        button.ghost {
            padding: 0.4rem 0.75rem; font-size: 0.85rem;
            border-radius: 6px; border: 1px solid rgba(128,128,128,0.35);
            background: transparent; color: inherit; cursor: pointer;
        }
        button.ghost:hover { background: rgba(128,128,128,0.1); }
        .progress { color: #888; font-size: 0.9rem; }
        .card {
            display: grid; grid-template-columns: 340px 1fr; gap: 1.25rem;
            padding: 1.25rem; margin-bottom: 1rem;
            border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
        }
        @media (max-width: 800px) { .card { grid-template-columns: 1fr; } }
        .video iframe {
            width: 100%; aspect-ratio: 9/16; border: 0;
            border-radius: 8px; background: #000;
        }
        .video .title { margin-top: 0.5rem; font-size: 0.85rem; color: #aaa; line-height: 1.3; }
        .video .title a { color: inherit; }
        .transcript { display: flex; flex-direction: column; min-width: 0; }
        .text {
            flex: 1; padding: 1rem; background: rgba(128,128,128,0.08);
            border-radius: 8px; white-space: pre-wrap; word-break: break-word;
            font-size: 1.05rem; min-height: 8rem;
        }
        .text.error { background: rgba(220,50,50,0.1); color: #d33; }
        .engine-label {
            font-size: 0.7rem; color: #888; text-transform: uppercase;
            letter-spacing: 0.05em; margin: 0.5rem 0 0.2rem;
        }
        .engine-label:first-child { margin-top: 0; }
        .badge {
            display: inline-block; padding: 1px 7px; border-radius: 999px;
            font-size: 0.7rem; font-weight: 600; margin-right: 0.3rem;
        }
        .badge.high { background: rgba(60,170,90,0.25); color: #4caf50; }
        .badge.med  { background: rgba(220,160,40,0.25); color: #d9a028; }
        .badge.low  { background: rgba(220,70,70,0.22);  color: #e05555; }
        .meta {
            margin-top: 0.75rem; font-size: 0.8rem; color: #888;
            display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center;
        }
        .meta code { background: rgba(128,128,128,0.15); padding: 1px 6px; border-radius: 4px; }
        .empty { color: #888; text-align: center; padding: 3rem 1rem; font-style: italic; }
        .toast {
            position: fixed; top: 1rem; left: 50%; transform: translateX(-50%);
            padding: 0.75rem 1.25rem; border-radius: 8px;
            background: rgba(30,30,30,0.95); color: white; font-size: 0.9rem;
            z-index: 999; opacity: 0; transition: opacity 0.2s;
        }
        .toast.show { opacity: 1; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/" class="active">▶ YouTube bench</a>
        <a href="/mic">🎙 Mic test</a>
        <a href="/training">🎓 Training data</a>
    </div>
    <h1>YouTube bench</h1>
    <p class="subtitle">Paste YouTube URLs → transcribed with <code>__MODEL_NAME__</code> → verify by clicking play.</p>

    <div class="submit-panel">
        <textarea id="urls" placeholder="Paste one or more YouTube URLs, one per line:
https://www.youtube.com/shorts/xxxxx
https://youtu.be/yyyyy
https://www.youtube.com/watch?v=zzzzz"></textarea>
        <div class="submit-row">
            <button id="submit" class="primary">Transcribe</button>
            <span id="progress" class="progress"></span>
        </div>
    </div>

    <div id="results"></div>
    <div id="toast" class="toast"></div>

<script>
const submitBtn = document.getElementById('submit');
const urlsInput = document.getElementById('urls');
const results   = document.getElementById('results');
const progress  = document.getElementById('progress');
const toast     = document.getElementById('toast');

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s ?? '';
    return d.innerHTML;
}
function showToast(msg, ms=2000) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), ms);
}
function fmtDur(s) {
    s = s || 0;
    if (s < 60) return `${Math.round(s)}s`;
    const m = Math.floor(s/60), r = Math.round(s % 60);
    return `${m}:${String(r).padStart(2, '0')}`;
}

function cardHTML(entry) {
    const embed = entry.video_id ? `https://www.youtube.com/embed/${entry.video_id}` : '';
    const errored = !!entry.error;
    const has2 = !!entry.transcript2;
    const agree = entry.agreement;
    const agreeBadge = (agree === null || agree === undefined) ? '' :
        `<span class="badge ${agree >= 0.7 ? 'high' : (agree >= 0.4 ? 'med' : 'low')}">engines agree ${(agree*100).toFixed(0)}%</span>`;
    const body = errored
        ? `<div class="text error">[Error] ${esc(entry.error)}</div>`
        : `<div class="engine-label">① ${esc(entry.model || 'engine 1')}</div>
           <div class="text">${esc(entry.transcript) || '(empty transcript)'}</div>` +
          (has2 ? `<div class="engine-label">② ${esc(entry.model2 || 'engine 2')}</div>
                   <div class="text">${esc(entry.transcript2)}</div>` : '');
    return `
    <div class="card" data-video-id="${esc(entry.video_id)}">
        <div class="video">
            <iframe src="${embed}" allowfullscreen loading="lazy"></iframe>
            <div class="title"><a href="${esc(entry.url)}" target="_blank" rel="noopener">${esc(entry.title || '(no title)')}</a></div>
            <div style="margin-top:0.4rem">${agreeBadge}</div>
        </div>
        <div class="transcript">
            ${body}
            <div class="meta">
                <span><code>${esc(entry.video_id)}</code></span>
                <span>video: ${fmtDur(entry.duration)}</span>
                <span>①: ${entry.server_s?.toFixed?.(1) ?? '?'}s</span>
                ${has2 ? `<span>②: ${entry.server2_s?.toFixed?.(1) ?? '?'}s</span>` : ''}
                <span><a href="${esc(entry.url)}" target="_blank" rel="noopener">open on YouTube ↗</a></span>
                <span style="flex:1"></span>
                <button class="ghost" onclick="retranscribe('${esc(entry.url)}', '${esc(entry.video_id)}')">Re-transcribe</button>
                <button class="ghost" onclick="deleteEntry('${esc(entry.video_id)}')">Delete</button>
            </div>
        </div>
    </div>`;
}

async function loadHistory() {
    try {
        const resp = await fetch('/bench/history');
        const items = await resp.json();
        if (!items.length) {
            results.innerHTML = '<div class="empty">No transcriptions yet. Paste URLs above and click Transcribe.</div>';
            return;
        }
        results.innerHTML = items.map(cardHTML).join('');
    } catch (err) {
        results.innerHTML = `<div class="empty">Failed to load history: ${esc(err.message)}</div>`;
    }
}

async function transcribeOne(url) {
    const resp = await fetch('/bench/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    });
    if (!resp.ok) {
        const detail = await resp.text();
        return { url, error: `${resp.status}: ${detail}`, video_id: url };
    }
    return await resp.json();
}

async function transcribeMany(urls) {
    submitBtn.disabled = true;
    let done = 0, failed = 0;
    for (const url of urls) {
        progress.textContent = `Processing ${done + 1}/${urls.length}... ${url.slice(-25)}`;
        try {
            const entry = await transcribeOne(url);
            if (entry.error) failed++;
        } catch (err) {
            failed++;
            showToast(`Failed: ${err.message}`);
        }
        done++;
        await loadHistory();  // refresh after each so user sees progress
    }
    progress.textContent = `Done: ${done - failed} ok, ${failed} failed.`;
    submitBtn.disabled = false;
    urlsInput.value = '';
}

async function deleteEntry(videoId) {
    if (!confirm('Delete this entry?')) return;
    await fetch(`/bench/history/${encodeURIComponent(videoId)}`, { method: 'DELETE' });
    await loadHistory();
    showToast('Deleted.');
}

async function retranscribe(url, videoId) {
    submitBtn.disabled = true;
    progress.textContent = `Re-transcribing ${videoId}...`;
    try {
        await transcribeOne(url);
        await loadHistory();
        showToast('Re-transcribed.');
    } finally {
        submitBtn.disabled = false;
        progress.textContent = '';
    }
}

submitBtn.addEventListener('click', () => {
    const raw = urlsInput.value.split(/\\r?\\n/).map(l => l.trim())
                .filter(l => l && !l.startsWith('#'));
    if (!raw.length) {
        showToast('Paste at least one URL.');
        return;
    }
    transcribeMany(raw);
});

loadHistory();
</script>
</body>
</html>
"""


# --- Request models -----------------------------------------------------

class BenchRequest(BaseModel):
    url: str


# --- Endpoints ----------------------------------------------------------

@app.get("/")
def bench_page():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse(BENCH_PAGE.replace("__MODEL_NAME__", config.STT_MODEL_NAME))


@app.get("/mic", response_class=HTMLResponse)
def mic_page() -> str:
    return MIC_PAGE.replace("__MODEL_NAME__", config.STT_MODEL_NAME)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.STT_MODEL_NAME}


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
) -> dict:
    """Multipart audio upload (used by /mic and by external callers).

    Returns BOTH engines' transcripts plus their agreement score. Older
    callers that only read `text` and `model` keep working unchanged.
    """
    audio_file = audio or file
    if not audio_file or not audio_file.filename:
        raise HTTPException(400, "missing audio or file parameter in upload")
    suffix = Path(audio_file.filename).suffix or ".wav"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(audio_file.file, tmp)
        tmp.close()
        return transcribe_both(tmp.name)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"transcription failed: {e}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.get("/bench/history")
def bench_history() -> list[dict]:
    """Return past YouTube transcriptions, newest first."""
    return load_history()


@app.delete("/bench/history/{video_id}")
def bench_delete(video_id: str) -> dict:
    """Remove one entry from history. Does not delete cached audio."""
    history = load_history()
    new_history = [h for h in history if h.get("video_id") != video_id]
    save_history(new_history)
    return {"ok": True, "removed": len(history) - len(new_history)}


@app.post("/bench/transcribe")
def bench_transcribe(body: BenchRequest) -> dict:
    """Download audio for a YouTube URL, transcribe, save to history.

    De-dupes on video_id — resubmitting the same URL updates the existing entry.
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(400, "url required")

    YT_BENCH_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    try:
        wav, meta = download_audio(url, YT_BENCH_AUDIO_DIR)
    except Exception as e:
        raise HTTPException(400, f"download failed: {e}")

    try:
        both = transcribe_both(wav)
    except Exception as e:
        raise HTTPException(500, f"transcription failed: {e}")

    entry = {
        "url": url,
        "video_id": meta.get("video_id", extract_video_id(url) or ""),
        "title": meta.get("title", ""),
        "duration": meta.get("duration", 0),
        "uploader": meta.get("uploader", ""),
        "transcript": both["text"],
        "server_s": both["duration_s"],
        "model": both["model"],
        "transcript2": both["text2"],
        "server2_s": both["duration2_s"],
        "model2": both["model2"],
        "agreement": both["agreement"],
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    # De-dupe by video_id: newest wins, moved to top.
    history = load_history()
    history = [h for h in history if h.get("video_id") != entry["video_id"]]
    history.insert(0, entry)
    save_history(history)
    return entry


# ========================================================================
# TRAINING DATA COLLECTION
# ========================================================================

def load_manifest() -> list[dict]:
    """Read accepted training pairs from JSON Lines file."""
    if not TRAINING_MANIFEST.exists():
        return []
    entries = []
    for line in TRAINING_MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def append_manifest(entry: dict) -> None:
    """Append one accepted pair. De-dupes by chunk_id (last write wins)."""
    entries = [e for e in load_manifest() if e.get("chunk_id") != entry.get("chunk_id")]
    entries.append(entry)
    TRAINING_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_MANIFEST.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


def remove_from_manifest(chunk_id: str) -> int:
    entries = load_manifest()
    kept = [e for e in entries if e.get("chunk_id") != chunk_id]
    TRAINING_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_MANIFEST.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )
    return len(entries) - len(kept)


class TrainingSourceRequest(BaseModel):
    url: str


class TrainingAcceptRequest(BaseModel):
    chunk_id: str
    caption: str            # original caption
    final_text: str         # what the user wants as the training label (may equal caption)
    source_url: str
    source_video_id: str
    start_s: float
    end_s: float
    duration_s: float


@app.post("/training/from_youtube")
def training_from_youtube(body: TrainingSourceRequest) -> dict:
    """Fetch a lyrics video, chunk by captions, run STT on each chunk.

    Returns a list of candidate pairs (chunk_id, audio URL, caption, stt_text).
    User then reviews each and calls /training/accept for the good ones.

    NOTE: this is a long-running request (30 s to several minutes depending
    on video length + chunk count). The frontend should show a spinner.
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(400, "url required")

    LYRICS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        wav, captions_path, meta = fetch_audio_and_captions(url, LYRICS_CACHE_DIR)
    except Exception as e:
        raise HTTPException(400, f"download failed: {e}")

    if captions_path is None or not captions_path.exists():
        raise HTTPException(
            400,
            "no Bangla captions (manual or auto) available for this video. "
            "yt-dlp couldn't find bn/bn-BD/bn-IN subtitles. Try a video that "
            "either has creator-uploaded lyrics or at least YouTube's auto-captions.",
        )

    captions = parse_captions(captions_path)
    if not captions:
        raise HTTPException(400, "caption file parsed to 0 lines")

    # If we fell back to auto-captions, run each line through Ollama, using the
    # video description as the ground-truth lyric bank. Manual captions are
    # trusted as-is.
    caption_source = meta.get("caption_source", "manual")
    llm_model_used: Optional[str] = None
    if caption_source == "auto" and ollama_available():
        llm_model_used = pick_model()
        if llm_model_used:
            cleaned = clean_captions_batch(captions, meta.get("description", ""))
            captions = [
                (start, end, cleaned_text if cleaned_text else "")
                for (start, end, _orig), cleaned_text in zip(captions, cleaned)
            ]
            # Drop lines the LLM marked as background music (empty string).
            captions = [(s, e, t) for (s, e, t) in captions if t]

    chunks = chunk_audio(wav, captions, TRAINING_CHUNKS_DIR, meta["video_id"])
    if not chunks:
        raise HTTPException(400, "no chunks in valid duration range (1-25 s)")

    # Second reference, independent of the caption track: the uploader's own
    # written text from the description, aligned onto the caption timeline.
    # Where this agrees with the caption, the label is trustworthy without a
    # human ever listening. Where it disagrees, the chunk needs review.
    reference_lines: list[str] = []
    if ollama_available():
        reference_lines = extract_reference_lines(meta.get("description", ""))
    aligned_reference = align_reference_to_captions(
        reference_lines,
        [(c["start_s"], c["end_s"], c["caption"]) for c in chunks],
    ) if reference_lines else ["" for _ in chunks]

    # Run BOTH STT engines on each chunk. This is the slow part.
    pairs = []
    for c, ref2 in zip(chunks, aligned_reference):
        both = transcribe_both(Path(c["path"]))
        caption = c["caption"]
        pairs.append({
            "chunk_id": c["chunk_id"],
            "audio_url": f"/training/audio/{c['chunk_id']}",
            "caption": caption,
            "reference2": ref2,                       # from description via LLM
            "stt_text": both["text"],                 # engine 1 (regional)
            "stt_text2": both["text2"],               # engine 2 (base turbo)
            "stt_agreement": both["agreement"],       # engine1 vs engine2
            "ref_agreement": round(agreement_score(caption, ref2), 3) if ref2 else None,
            "start_s": c["start_s"],
            "end_s": c["end_s"],
            "duration_s": c["duration_s"],
        })

    return {
        "source_url": url,
        "video_id": meta["video_id"],
        "title": meta["title"],
        "duration": meta["duration"],
        "n_chunks": len(pairs),
        "caption_source": caption_source,
        "llm_used": llm_model_used,
        "reference2_lines": len(reference_lines),
        "engines": [config.STT_MODEL_NAME, engine2.MODEL_NAME],
        "pairs": pairs,
    }


@app.post("/training/from_mic")
async def training_from_mic(audio: UploadFile = File(...)) -> dict:
    """Record-your-own-voice training source.

    Why this exists: YouTube gives us lots of audio but only the dialects and
    speakers that happen to be on YouTube. For Omnicare the audio that matters
    is *your* microphone, *your* speakers, *your* clinical vocabulary — and
    none of that is on YouTube. Recording it directly is the only way to get it.

    Flow: browser posts a WebM blob -> ffmpeg normalises it to the same
    16 kHz mono WAV shape as the YouTube chunks -> both engines transcribe ->
    the pair comes back for review exactly like a YouTube chunk, so the review
    UI needs no special case. You then type what you actually said and Accept.
    """
    TRAINING_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Stable, sortable id. Seconds resolution is enough — one recording per
    # second per user is not a real collision risk here.
    chunk_id = f"mic_{datetime.now():%Y%m%d_%H%M%S}"
    dest = TRAINING_CHUNKS_DIR / f"{chunk_id}.wav"

    suffix = Path(audio.filename or "mic.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(audio.file, tmp)
        tmp.close()

        # Normalise to 16 kHz mono WAV so mic chunks and YouTube chunks are
        # byte-compatible as training data. Whisper resamples anyway, but a
        # uniform corpus avoids surprises at fine-tune time.
        import subprocess
        from yt_bench import get_ffmpeg_path
        ffmpeg_exe = get_ffmpeg_path()
        subprocess.run(
            [ffmpeg_exe, "-y", "-loglevel", "error", "-i", tmp.name,
             "-ac", "1", "-ar", "16000", str(dest)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(400, f"could not decode uploaded audio: {e}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    try:
        import soundfile as sf
        info = sf.info(str(dest))
        duration_s = round(info.frames / info.samplerate, 3)
    except Exception:
        duration_s = 0.0

    both = transcribe_both(dest)

    # Same shape as a YouTube pair so the review UI renders it unchanged.
    # caption/reference2 are empty: for a mic recording YOU are the reference,
    # which is the whole point — you type the truth into the label box.
    return {
        "source_url": "",
        "video_id": chunk_id,
        "title": f"Mic recording {chunk_id}",
        "n_chunks": 1,
        "caption_source": "mic",
        "llm_used": None,
        "engines": [config.STT_MODEL_NAME, engine2.MODEL_NAME],
        "pairs": [{
            "chunk_id": chunk_id,
            "audio_url": f"/training/audio/{chunk_id}",
            "caption": "",
            "reference2": "",
            "stt_text": both["text"],
            "stt_text2": both["text2"],
            "stt_agreement": both["agreement"],
            "ref_agreement": None,
            "start_s": 0.0,
            "end_s": duration_s,
            "duration_s": duration_s,
        }],
    }


@app.get("/training/manifest")
def training_manifest() -> list[dict]:
    """All accepted training pairs so far."""
    return load_manifest()


@app.post("/training/accept")
def training_accept(body: TrainingAcceptRequest) -> dict:
    """Save one accepted (audio, text) training pair to the manifest."""
    if not body.chunk_id or not body.final_text.strip():
        raise HTTPException(400, "chunk_id and non-empty final_text required")
    entry = body.model_dump()
    entry["accepted_at"] = datetime.now().isoformat(timespec="seconds")
    entry["model_when_accepted"] = config.STT_MODEL_NAME
    append_manifest(entry)
    return {"ok": True, "total": len(load_manifest())}


@app.delete("/training/manifest/{chunk_id}")
def training_manifest_delete(chunk_id: str) -> dict:
    removed = remove_from_manifest(chunk_id)
    return {"ok": True, "removed": removed}


@app.get("/training/audio/{chunk_id}")
def training_audio(chunk_id: str):
    """Serve one chunk's WAV so the review UI can play it."""
    # Sanitize: only allow the pattern we generate.
    if not chunk_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "bad chunk_id")
    path = TRAINING_CHUNKS_DIR / f"{chunk_id}.wav"
    if not path.exists():
        raise HTTPException(404, "chunk not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/training/export")
def training_export() -> StreamingResponse:
    """Download manifest as CSV — the format the Kaggle fine-tune notebook consumes."""
    entries = load_manifest()
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["chunk_id", "final_text", "caption", "duration_s",
                    "source_url", "source_video_id", "accepted_at"],
        extrasaction="ignore",
    )
    writer.writeheader()
    for e in entries:
        writer.writerow(e)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="training_manifest.csv"'},
    )


TRAINING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Omnicare STT — Training data</title>
    <style>
        :root { color-scheme: light dark; }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px; margin: 0 auto; padding: 1.5rem 1rem; line-height: 1.5;
        }
        h1 { margin: 0 0 0.25rem; font-size: 1.4rem; }
        .subtitle { color: #888; font-size: 0.9rem; margin: 0 0 1rem; }
        .nav { margin-bottom: 1.5rem; }
        .nav a { color: #888; text-decoration: none; margin-right: 1rem; }
        .nav a.active { color: inherit; font-weight: 600; }
        .panel {
            padding: 1rem; margin-bottom: 1.5rem;
            border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
        }
        input[type=text] {
            width: 100%; padding: 0.6rem 0.8rem; font-size: 0.95rem;
            background: rgba(128,128,128,0.08); color: inherit;
            border: 1px solid rgba(128,128,128,0.25); border-radius: 8px;
        }
        .row { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-top: 0.75rem; }
        button.primary {
            padding: 0.6rem 1.25rem; font-size: 0.95rem; font-weight: 600;
            border-radius: 8px; border: 0; background: #2b7cff; color: white; cursor: pointer;
        }
        button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
        button.ghost {
            padding: 0.4rem 0.75rem; font-size: 0.85rem;
            border-radius: 6px; border: 1px solid rgba(128,128,128,0.35);
            background: transparent; color: inherit; cursor: pointer;
        }
        button.ghost:hover { background: rgba(128,128,128,0.1); }
        .status { color: #888; font-size: 0.9rem; }
        .counts {
            padding: 0.6rem 1rem; margin-bottom: 1rem;
            background: rgba(50,150,80,0.15); border-radius: 8px;
            font-size: 0.9rem;
        }
        .pair {
            padding: 0.85rem; margin-bottom: 0.6rem;
            border: 1px solid rgba(128,128,128,0.25); border-radius: 10px;
            display: grid; grid-template-columns: 220px 1fr auto; gap: 0.75rem;
            align-items: start;
        }
        @media (max-width: 800px) {
            .pair { grid-template-columns: 1fr; }
        }
        .pair audio { width: 100%; }
        .pair .times { font-size: 0.75rem; color: #888; margin-top: 0.35rem; }
        .pair .labels { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
        .pair .caption-row, .pair .stt-row { margin-bottom: 0.5rem; }
        .pair .caption-row .text {
            background: rgba(50,150,80,0.1); padding: 0.5rem 0.75rem; border-radius: 6px;
            font-size: 1rem; word-break: break-word;
        }
        .pair .stt-row .text {
            background: rgba(128,128,128,0.1); padding: 0.5rem 0.75rem; border-radius: 6px;
            font-size: 1rem; word-break: break-word;
        }
        .pair .final-input {
            width: 100%; padding: 0.5rem 0.75rem; font-size: 1rem;
            background: rgba(30,130,220,0.1); border: 1px solid rgba(30,130,220,0.4);
            color: inherit; border-radius: 6px; margin-top: 0.25rem; resize: vertical;
            min-height: 3rem; font-family: inherit;
        }
        .pair .actions {
            display: flex; flex-direction: column; gap: 0.4rem;
        }
        .pair.accepted { background: rgba(50,150,80,0.08); border-color: rgba(50,150,80,0.4); }
        .pair.accepted .actions button.primary { background: #4caf50; }
        .pair .text.muted { color: #888; font-style: italic; font-size: 0.9rem; }
        .badge {
            display: inline-block; padding: 1px 7px; border-radius: 999px;
            font-size: 0.7rem; font-weight: 600; margin-right: 0.3rem;
        }
        .badge.high { background: rgba(60,170,90,0.25); color: #4caf50; }
        .badge.med  { background: rgba(220,160,40,0.25); color: #d9a028; }
        .badge.low  { background: rgba(220,70,70,0.22);  color: #e05555; }
        .empty { color: #888; text-align: center; padding: 2rem 1rem; font-style: italic; }
        .src-title {
            font-size: 0.75rem; font-weight: 600; color: #888;
            text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.6rem;
        }
        .hint { font-size: 0.85rem; color: #888; margin: 0 0 0.75rem; }
        .rec-btn.recording {
            background: #d33;
            animation: pulse 1.2s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(211,51,51,0.5); }
            50%      { box-shadow: 0 0 0 10px rgba(211,51,51,0); }
        }
        .toast {
            position: fixed; top: 1rem; left: 50%; transform: translateX(-50%);
            padding: 0.75rem 1.25rem; border-radius: 8px;
            background: rgba(30,30,30,0.95); color: white; font-size: 0.9rem;
            z-index: 999; opacity: 0; transition: opacity 0.2s;
        }
        .toast.show { opacity: 1; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">▶ YouTube bench</a>
        <a href="/mic">🎙 Mic test</a>
        <a href="/training" class="active">🎓 Training data</a>
    </div>
    <h1>Training data collection</h1>
    <p class="subtitle">Paste a lyrics YouTube URL (creator-uploaded Bangla captions required). Each caption becomes a training candidate.</p>

    <div class="panel">
        <div class="src-title">Source A — YouTube video</div>
        <input id="url" type="text" placeholder="https://www.youtube.com/watch?v=...">
        <div class="row">
            <button id="fetch" class="primary">Fetch + chunk + STT</button>
            <span id="status" class="status"></span>
            <span style="flex:1"></span>
            <button class="ghost" onclick="window.open('/training/export')">Export manifest (CSV)</button>
        </div>
    </div>

    <div class="panel">
        <div class="src-title">Source B — record your own voice</div>
        <p class="hint">The audio that matters most for Omnicare isn't on YouTube — it's your mic, your accent, your clinical vocabulary. Record a phrase, then type exactly what you said.</p>
        <div class="row">
            <button id="rec" class="primary rec-btn">🎙 Start recording</button>
            <span id="recStatus" class="status"></span>
        </div>
    </div>

    <div id="counts" class="counts">Loading saved count...</div>

    <h2 style="font-size: 1.1rem; margin: 1rem 0 0.5rem;">Candidates from current source</h2>
    <div id="pairs"></div>

    <h2 style="font-size: 1.1rem; margin: 2rem 0 0.5rem;">Already accepted (saved)</h2>
    <div id="saved"></div>

    <div id="toast" class="toast"></div>

<script>
const urlInput = document.getElementById('url');
const fetchBtn = document.getElementById('fetch');
const status   = document.getElementById('status');
const recBtn   = document.getElementById('rec');
const recStatus= document.getElementById('recStatus');
const pairsDiv = document.getElementById('pairs');
const savedDiv = document.getElementById('saved');
const countsDiv= document.getElementById('counts');
const toast    = document.getElementById('toast');

function esc(s) { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; }
function showToast(msg, ms=2000) {
    toast.textContent = msg; toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), ms);
}

async function refreshCounts() {
    const items = await fetch('/training/manifest').then(r => r.json());
    countsDiv.textContent = `${items.length} accepted training pairs saved to manifest.jsonl.`;
    renderSaved(items);
}

function renderSaved(items) {
    if (!items.length) {
        savedDiv.innerHTML = '<div class="empty">Nothing saved yet.</div>';
        return;
    }
    savedDiv.innerHTML = items.slice(0, 30).map(e => `
        <div class="pair accepted">
            <div>
                <audio controls preload="none" src="/training/audio/${esc(e.chunk_id)}"></audio>
                <div class="times">${esc(e.chunk_id)} · ${(e.duration_s || 0).toFixed(1)}s</div>
            </div>
            <div>
                <div class="labels">saved label</div>
                <div class="caption-row"><div class="text">${esc(e.final_text)}</div></div>
                <div class="times">from <a href="${esc(e.source_url)}" target="_blank">${esc(e.source_video_id)}</a> @ ${e.start_s?.toFixed?.(1) ?? '?'}s</div>
            </div>
            <div class="actions">
                <button class="ghost" onclick="unsave('${esc(e.chunk_id)}')">Remove</button>
            </div>
        </div>
    `).join('') + (items.length > 30 ? `<div class="empty">…and ${items.length - 30} more (download the CSV to see all).</div>` : '');
}

function confBadge(p) {
    // Two independent signals:
    //   ref_agreement  = caption vs description-derived reference (is the LABEL right?)
    //   stt_agreement  = engine1 vs engine2          (did the MODELS hear the same thing?)
    const ref = p.ref_agreement;
    const stt = p.stt_agreement;
    let cls = 'low', label = 'review';
    if (ref !== null && ref !== undefined && ref >= 0.7) { cls = 'high'; label = 'label confirmed'; }
    else if (ref !== null && ref !== undefined && ref >= 0.4) { cls = 'med'; label = 'label partial'; }
    else if (ref === null || ref === undefined) { cls = 'med'; label = 'no 2nd reference'; }
    const sttPct = (stt * 100).toFixed(0);
    return `<span class="badge ${cls}">${label}</span>
            <span class="badge ${stt >= 0.7 ? 'high' : (stt >= 0.4 ? 'med' : 'low')}">engines ${sttPct}%</span>`;
}

function pairHTML(p, sourceUrl, sourceVideoId, savedIds) {
    const already = savedIds.has(p.chunk_id);
    const ref2 = p.reference2 || '';
    return `
    <div class="pair${already ? ' accepted' : ''}" data-chunk="${esc(p.chunk_id)}">
        <div>
            <audio controls preload="none" src="/training/audio/${esc(p.chunk_id)}"></audio>
            <div class="times">${esc(p.chunk_id)} · ${p.duration_s.toFixed(1)}s · ${p.start_s.toFixed(1)}s → ${p.end_s.toFixed(1)}s</div>
            <div style="margin-top:0.4rem">${confBadge(p)}</div>
        </div>
        <div>
            <div class="labels">① caption track</div>
            <div class="caption-row"><div class="text">${esc(p.caption)}</div></div>

            <div class="labels">② 2nd reference — from video description via LLM</div>
            <div class="caption-row"><div class="text${ref2 ? '' : ' muted'}">${ref2 ? esc(ref2) : '(no match found in description)'}</div></div>

            <div class="labels">③ STT engine 1 — regional (Bangla-strong)</div>
            <div class="stt-row"><div class="text">${esc(p.stt_text)}</div></div>

            <div class="labels">④ STT engine 2 — base turbo (English/noise-strong)</div>
            <div class="stt-row"><div class="text">${esc(p.stt_text2 || '')}</div></div>

            <div class="labels">final training label — edit if needed, then Accept</div>
            <textarea class="final-input">${esc(p.caption)}</textarea>
            <div class="row" style="margin-top:0.35rem">
                <button class="ghost" onclick="useText(this, 1)">use ①</button>
                <button class="ghost" onclick="useText(this, 2)" ${ref2 ? '' : 'disabled'}>use ②</button>
                <button class="ghost" onclick="useText(this, 3)">use ③</button>
                <button class="ghost" onclick="useText(this, 4)">use ④</button>
            </div>
        </div>
        <div class="actions">
            <button class="primary" onclick="acceptPair(this, '${esc(p.chunk_id)}', '${esc(p.caption).replace(/'/g,"&apos;")}', '${esc(sourceUrl)}', '${esc(sourceVideoId)}', ${p.start_s}, ${p.end_s}, ${p.duration_s})">${already ? 'Update' : 'Accept'}</button>
            <button class="ghost" onclick="this.closest('.pair').remove()">Skip</button>
        </div>
    </div>`;
}

// Copy one of the four candidate texts into the editable label box.
function useText(btn, which) {
    const pair = btn.closest('.pair');
    const texts = pair.querySelectorAll('.caption-row .text, .stt-row .text');
    const chosen = texts[which - 1];
    if (chosen && !chosen.classList.contains('muted')) {
        pair.querySelector('.final-input').value = chosen.textContent.trim();
    }
}

async function acceptPair(btn, chunkId, originalCaption, sourceUrl, sourceVideoId, start_s, end_s, duration_s) {
    const pair = btn.closest('.pair');
    const finalText = pair.querySelector('.final-input').value.trim();
    if (!finalText) { showToast('Empty label — not saving.'); return; }
    btn.disabled = true;
    try {
        const resp = await fetch('/training/accept', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({
                chunk_id: chunkId, caption: originalCaption, final_text: finalText,
                source_url: sourceUrl, source_video_id: sourceVideoId,
                start_s, end_s, duration_s
            }),
        });
        const data = await resp.json();
        pair.classList.add('accepted');
        btn.textContent = 'Update';
        showToast(`Saved. Total: ${data.total}`);
        refreshCounts();
    } catch (err) { showToast('Save failed: ' + err.message); }
    finally { btn.disabled = false; }
}

async function unsave(chunkId) {
    if (!confirm('Remove this saved pair?')) return;
    await fetch(`/training/manifest/${encodeURIComponent(chunkId)}`, {method:'DELETE'});
    showToast('Removed.');
    refreshCounts();
}

// --- Source B: record your own voice -----------------------------------
let mediaRecorder = null, recChunks = [];

async function startRec() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        recChunks = [];
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recChunks.push(e.data); };
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());   // release the mic light
            await sendRecording(new Blob(recChunks, { type: 'audio/webm' }));
        };
        mediaRecorder.start();
        recBtn.textContent = '⏹ Stop recording';
        recBtn.classList.add('recording');
        recStatus.textContent = '🔴 Recording — speak, then press Stop.';
    } catch (err) {
        recStatus.textContent = 'Mic unavailable: ' + err.message;
    }
}

function stopRec() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    recBtn.textContent = '🎙 Start recording';
    recBtn.classList.remove('recording');
    recStatus.textContent = '⏳ Transcribing with both engines...';
}

async function sendRecording(blob) {
    recBtn.disabled = true;
    const form = new FormData();
    form.append('audio', blob, 'mic.webm');
    try {
        const resp = await fetch('/training/from_mic', { method: 'POST', body: form });
        if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
        const data = await resp.json();
        recStatus.textContent = 'Recorded — type what you actually said, then Accept.';
        const saved = new Set((await fetch('/training/manifest').then(r => r.json())).map(e => e.chunk_id));
        // Prepend so the newest recording is at the top, above any YouTube chunks.
        pairsDiv.insertAdjacentHTML('afterbegin',
            data.pairs.map(p => pairHTML(p, data.source_url, data.video_id, saved)).join(''));
        // Nothing to pre-fill for a mic clip — seed the label box with engine 1's
        // guess so you can correct it instead of typing from scratch.
        const box = pairsDiv.querySelector('.pair .final-input');
        if (box && !box.value) box.value = data.pairs[0].stt_text || '';
        box?.focus();
    } catch (err) {
        recStatus.textContent = 'Failed: ' + err.message;
    } finally {
        recBtn.disabled = false;
    }
}

recBtn.addEventListener('click', () =>
    (mediaRecorder && mediaRecorder.state === 'recording') ? stopRec() : startRec());

fetchBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) { showToast('Paste a URL first.'); return; }
    fetchBtn.disabled = true;
    status.textContent = 'Fetching, chunking, transcribing — this can take a minute...';
    pairsDiv.innerHTML = '';
    try {
        const resp = await fetch('/training/from_youtube', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({url}),
        });
        if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
        const data = await resp.json();
        let sourceLabel = data.caption_source === 'manual'
            ? '✅ manual captions (trusted)'
            : (data.llm_used
                ? `⚠️ auto-captions cleaned by ${data.llm_used} (review carefully)`
                : '❌ raw auto-captions (LLM offline — review carefully)');
        status.textContent = `Got ${data.n_chunks} chunks from "${data.title}" — ${sourceLabel}`;
        const saved = new Set((await fetch('/training/manifest').then(r=>r.json())).map(e=>e.chunk_id));
        pairsDiv.innerHTML = data.pairs.map(p => pairHTML(p, data.source_url, data.video_id, saved)).join('');
    } catch (err) {
        status.textContent = '';
        pairsDiv.innerHTML = `<div class="empty">Failed: ${esc(err.message)}</div>`;
    } finally {
        fetchBtn.disabled = false;
    }
});

refreshCounts();
</script>
</body>
</html>
"""


@app.get("/training", response_class=HTMLResponse)
def training_page() -> str:
    return TRAINING_PAGE.replace("__MODEL_NAME__", config.STT_MODEL_NAME)
