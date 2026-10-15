"""Turns noisy per-frame predictions into clean pressed/released events.

Activation needs `confirm_frames` consecutive identical predictions; release needs
`release_frames` consecutive frames that do not match the active gesture (so a dropped
detection or a one-frame flicker does not end a hold). Frame counts are read from the
settings object on every update, so GUI changes apply live.

DwellDetector is the time-based equivalent used for gaze / face signals.
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


class DwellDetector:
    """Time-based pressed/released for a boolean signal, with dwell and cooldown."""

    def __init__(self, name: str, source: str, dwell_ms: int, release_ms: int, cooldown_ms: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.name, self.source = name, source
        self.dwell, self.release, self.cooldown = dwell_ms / 1000.0, release_ms / 1000.0, cooldown_ms / 1000.0
        self.clock = clock
        self.pressed = False
        self._since: Optional[float] = None
        self._last_active = 0.0
        self._cooldown_until = 0.0

    def update(self, active: bool) -> List[GestureEvent]:
        now = self.clock()
        events: List[GestureEvent] = []
        if active:
            self._last_active = now
            if self._since is None:
                self._since = now
            if not self.pressed and now - self._since >= self.dwell and now >= self._cooldown_until:
                self.pressed = True
                events.append(GestureEvent("pressed", self.name, self.source, now))
        else:
            if self.pressed:
                if now - self._last_active >= self.release:
                    self.pressed = False
                    self._since = None
                    self._cooldown_until = now + self.cooldown
                    events.append(GestureEvent("released", self.name, self.source, now))
            else:
                self._since = None
        return events

    def reset(self) -> List[GestureEvent]:
        ev = []
        if self.pressed:
            ev.append(GestureEvent("released", self.name, self.source, self.clock()))
        self.pressed, self._since = False, None
        return ev
