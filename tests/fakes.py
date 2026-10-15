"""Test doubles: no camera, no MediaPipe, no real keyboard."""
from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np


class FakeKeyboard:
    def __init__(self):
        self.combos: List[Tuple[str, ...]] = []
        self.fail = False
        self._lock = threading.Lock()

    def press_combo(self, combo):
        if self.fail:
            raise RuntimeError("boom")
        with self._lock:
            self.combos.append(tuple(combo))


class FakeTracker:
    """Returns whatever HandObservation the test sets next."""

    def __init__(self):
        self.next = None
        self.calls = 0

    def process(self, frame, ts):
        self.calls += 1
        return self.next

    def close(self):
        pass


class FakeCamera:
    status = "streaming"

    def __init__(self):
        self.frame: Optional[np.ndarray] = np.zeros((480, 640, 3), np.uint8)

    def restart(self):
        pass

    def latest(self):
        return None if self.frame is None else (1, self.frame)
