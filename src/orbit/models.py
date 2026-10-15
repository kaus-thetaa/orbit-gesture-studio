"""MediaPipe model files: located in the ORBIT config dir, downloaded on first use.

Model files are NOT stored in the repository. They are Google-hosted MediaPipe Tasks
bundles; check their licence terms before redistributing them.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import paths
from .errors import ModelError

log = logging.getLogger(__name__)

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand_landmarker": {
        "file": "hand_landmarker.task",
        "url": f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    },
    "face_landmarker": {
        "file": "face_landmarker.task",
        "url": f"{_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    },
}
_MIN_BYTES = 500_000  # anything smaller is an error page, not a model


def model_path(name: str, directory: Optional[Path] = None) -> Path:
    if name not in MODELS:
        raise ModelError(f"Unknown model {name!r}")
    return (Path(directory) if directory else paths.models_dir()) / MODELS[name]["file"]


def ensure_model(
    name: str,
    directory: Optional[Path] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Return the local path to a model, downloading it if necessary."""
    dest = model_path(name, directory)
    if dest.exists() and dest.stat().st_size >= _MIN_BYTES:
        return dest

    url = MODELS[name]["url"]
    log.info("Downloading %s model from %s", name, url)
    tmp = dest.with_suffix(".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "orbit-gesture-studio"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if tmp.stat().st_size < _MIN_BYTES:
            raise ModelError("Downloaded file is suspiciously small")
        os.replace(tmp, dest)
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ModelError(
            f"Could not download the {name} model: {exc}\n"
            f"Download it manually from:\n  {url}\n"
            f"and save it as:\n  {dest}"
        ) from exc
    log.info("Saved %s (%d bytes)", dest, dest.stat().st_size)
    return dest
