"""Reference client — how the Omnicare Railway app should call the STT service.

This file is NOT part of the service. It is documentation-as-code for the
app-side developer (you), so the wire contract is unambiguous. Copy the
`transcribe_via_api` function into the Railway app and adjust the URL.

Run against a local server for verification:
    ./run_server.sh          # in one terminal
    python client_example.py audio/eval_fleurs/fleurs_bn_00.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

# Point at wherever the STT server is reachable from the caller.
# Local dev:  http://localhost:8000
# Modal:      https://<your-modal-app>.modal.run
# RunPod:     https://<pod-id>-8000.proxy.runpod.net
STT_API_URL = "http://localhost:8000"


def transcribe_via_api(audio_path: str | Path, base_url: str = STT_API_URL) -> dict:
    """POST an audio file, return the parsed JSON response.

    Raises `requests.HTTPError` on non-2xx so callers know something failed
    rather than silently receiving an empty transcript.
    """
    audio_path = Path(audio_path)
    with audio_path.open("rb") as f:
        # `files=` is the multipart form; the tuple carries filename +
        # content so FastAPI's UploadFile sees a real filename (and can
        # infer the extension for ffmpeg).
        resp = requests.post(
            f"{base_url}/transcribe",
            files={"audio": (audio_path.name, f, "application/octet-stream")},
            timeout=120,  # generous — first-request warmup + long audio
        )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python client_example.py <audio_file>")
        return 2
    result = transcribe_via_api(sys.argv[1])
    print(f"model:     {result['model']}")
    print(f"took:      {result['duration_s']} s")
    print(f"text:      {result['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
