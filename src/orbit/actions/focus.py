"""Focus mode state.

Focus mode itself is a small, safe state machine: ON/OFF plus listeners. In the GUI a
listener shows a click-through screen-edge vignette (visual "focus" cue) and the status
badge. Keeping the *state* separate from the *visuals* means it works headless too and is
trivially testable. It does not close, minimise or move any of your windows.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List

log = logging.getLogger(__name__)

Listener = Callable[[bool, Dict], None]


class FocusController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._listeners: List[Listener] = []

    @property
    def active(self) -> bool:
        return self._active

    def add_listener(self, cb: Listener) -> None:
        self._listeners.append(cb)

    def _notify(self, active: bool, options: Dict) -> None:
        for cb in list(self._listeners):
            try:
                cb(active, options)
            except Exception:
                log.exception("focus listener failed")

    def activate(self, dim: int = 35) -> bool:
        """Idempotent. Returns True if the state changed."""
        with self._lock:
            if self._active:
                return False
            self._active = True
        log.info("Focus mode ON (dim=%s%%)", dim)
        self._notify(True, {"dim": dim})
        return True

    def deactivate(self) -> bool:
        with self._lock:
            if not self._active:
                return False
            self._active = False
        log.info("Focus mode OFF")
        self._notify(False, {})
        return True
