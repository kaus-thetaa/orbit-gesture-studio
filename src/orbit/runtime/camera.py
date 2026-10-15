"""Threaded webcam capture with automatic reconnection.

The capture thread always keeps only the *latest* frame, so inference never works on a
stale backlog. If reads fail repeatedly (unplugged, privacy shutter, another app took the
device) the camera is released, status becomes "reconnecting", and reopening is retried
every `reconnect_interval_s`. latest() returns None when the newest frame is older than
`stale_after_s`, so a frozen feed is treated like a lost one.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from ..settings import CameraSettings

log = logging.getLogger(__name__)


def default_opener(cfg: CameraSettings) -> Optional[Any]:
    import cv2

    backends: List[int]
    choice = cfg.backend.lower()
    if choice == "dshow":
        backends = [cv2.CAP_DSHOW]
    elif choice == "msmf":
        backends = [cv2.CAP_MSMF]
    elif choice == "any":
        backends = [cv2.CAP_ANY]
    else:  # auto: DirectShow opens much faster than MSMF on most Windows laptops
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]

    for be in backends:
        cap = cv2.VideoCapture(cfg.index, be)
        if not cap.isOpened():
            cap.release()
            continue
        if cfg.fourcc and len(cfg.fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    return None


class CameraSource:
    def __init__(self, cfg: CameraSettings, opener: Callable[[CameraSettings], Optional[Any]] = default_opener,
                 on_status: Optional[Callable[[str], None]] = None) -> None:
        self.cfg, self._opener, self.on_status = cfg, opener, on_status
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._ts = 0.0
        self._status = "stopped"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._restart = threading.Event()

    @property
    def status(self) -> str:
        return self._status

    def _set_status(self, status: str) -> None:
        if status != self._status:
            self._status = status
            log.info("camera: %s", status)
            if self.on_status:
                try:
                    self.on_status(status)
                except Exception:
                    log.exception("camera status callback failed")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._set_status("connecting")
        self._thread = threading.Thread(target=self._run, name="orbit-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._set_status("stopped")

    def restart(self) -> None:
        """Re-open the device (e.g. after changing the camera index)."""
        self._restart.set()

    def latest(self) -> Optional[Tuple[int, np.ndarray]]:
        with self._lock:
            if self._frame is None or time.monotonic() - self._ts > self.cfg.stale_after_s:
                return None
            return self._seq, self._frame

    def _run(self) -> None:
        cap = None
        failures = 0
        first = True
        while not self._stop.is_set():
            if self._restart.is_set():
                self._restart.clear()
                if cap is not None:
                    cap.release()
                    cap = None
            if cap is None:
                self._set_status("connecting" if first else "reconnecting")
                try:
                    cap = self._opener(self.cfg)
                except Exception:
                    log.exception("camera open failed")
                    cap = None
                if cap is None:
                    self._stop.wait(self.cfg.reconnect_interval_s)
                    continue
                failures = 0
                first = False
            try:
                ok, frame = cap.read()
            except Exception:
                log.exception("camera read raised")
                ok, frame = False, None
            if ok and frame is not None:
                failures = 0
                with self._lock:
                    self._frame, self._ts = frame, time.monotonic()
                    self._seq += 1
                self._set_status("streaming")
            else:
                failures += 1
                if failures >= self.cfg.read_failure_limit:
                    log.warning("camera lost after %d failed reads", failures)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    with self._lock:
                        self._frame = None
                    self._set_status("reconnecting")
                    self._stop.wait(self.cfg.reconnect_interval_s)
                else:
                    self._stop.wait(0.01)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def list_cameras(max_index: int = 5, cfg: Optional[CameraSettings] = None) -> List[dict]:
    """Probe camera indices (used by `orbit cameras`)."""
    out = []
    base = cfg or CameraSettings()
    for i in range(max_index + 1):
        c = CameraSettings(**{**base.__dict__, "index": i})
        cap = None
        try:
            cap = default_opener(c)
            if cap is None:
                out.append({"index": i, "ok": False})
                continue
            ok, frame = cap.read()
            out.append({"index": i, "ok": bool(ok), "shape": None if frame is None else tuple(frame.shape)})
        except Exception as exc:
            out.append({"index": i, "ok": False, "error": str(exc)})
        finally:
            if cap is not None:
                cap.release()
    return out
