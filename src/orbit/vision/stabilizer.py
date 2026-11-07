"""Turns noisy per-frame predictions into clean pressed/released events.

Activation needs `confirm_frames` consecutive identical predictions; release needs
`release_frames` consecutive frames that do not match the active gesture (so a dropped
detection or a one-frame flicker does not end a hold). Frame counts are read from the
settings object on every update, so GUI changes apply live.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from ..settings import RecognitionSettings
from .types import GestureEvent


class GestureStabilizer:
    def __init__(self, cfg: RecognitionSettings, source: str = "hand", clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.source = source
        self.clock = clock
        self.active: Optional[str] = None
        self._cand: Optional[str] = None
        self._cand_n = 0
        self._miss = 0

    def update(self, name: Optional[str]) -> List[GestureEvent]:
        events: List[GestureEvent] = []
        now = self.clock()

        # 1. candidate counter
        if name is None:
            self._cand, self._cand_n = None, 0
        elif name == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = name, 1

        # 2. release hysteresis
        if self.active is not None:
            self._miss = 0 if name == self.active else self._miss + 1
            if self._miss >= self.cfg.release_frames:
                events.append(GestureEvent("released", self.active, self.source, now))
                self.active, self._miss = None, 0

        # 3. activation
        if self.active is None and self._cand is not None and self._cand_n >= self.cfg.confirm_frames:
            self.active, self._miss = self._cand, 0
            events.append(GestureEvent("pressed", self.active, self.source, now))
        return events

    def reset(self) -> List[GestureEvent]:
        """Force-release whatever is active (e.g. camera lost)."""
        events: List[GestureEvent] = []
        if self.active is not None:
            events.append(GestureEvent("released", self.active, self.source, self.clock()))
        self.active, self._cand, self._cand_n, self._miss = None, None, 0, 0
        return events
