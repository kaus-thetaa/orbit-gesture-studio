"""Safety controller (arm / disarm / emergency stop) and global hotkeys.

ORBIT always starts DISARMED: the camera, recognition and the live preview run, and the
UI shows what *would* happen ("dry run"), but no desktop action executes until you arm
it. Disarming or the emergency stop releases everything that is currently held/toggled.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

Listener = Callable[[bool, str], None]  # (armed, reason)


class SafetyController:
    def __init__(self, armed: bool = False) -> None:
        self._armed = armed
        self._lock = threading.Lock()
        self._listeners: List[Listener] = []
        self.last_reason = "startup"

    @property
    def armed(self) -> bool:
        return self._armed

    def add_listener(self, cb: Listener) -> None:
        self._listeners.append(cb)

    def _set(self, armed: bool, reason: str) -> bool:
        with self._lock:
            changed = armed != self._armed
            self._armed = armed
            self.last_reason = reason
        if changed or not armed:  # always re-run release logic on disarm/e-stop
            log.warning("ORBIT %s (%s)", "ARMED" if armed else "DISARMED", reason)
            for cb in list(self._listeners):
                try:
                    cb(armed, reason)
                except Exception:
                    log.exception("safety listener failed")
        return changed

    def arm(self, reason: str = "user") -> None:
        self._set(True, reason)

    def disarm(self, reason: str = "user") -> None:
        self._set(False, reason)

    def toggle(self, reason: str = "user") -> bool:
        if self._armed:
            self.disarm(reason)
        else:
            self.arm(reason)
        return self._armed

    def emergency_stop(self, reason: str = "emergency stop") -> None:
        self._set(False, reason)


class HotkeyManager:
    """Global hotkeys via pynput. Fails soft: if unavailable, the GUI buttons still work."""

    def __init__(self, bindings: Dict[str, Callable[[], None]]) -> None:
        self.bindings = bindings
        self._listener = None
        self.error: Optional[str] = None

    def start(self) -> bool:
        try:
            from pynput import keyboard

            wrapped = {}
            for combo, fn in self.bindings.items():
                keyboard.HotKey.parse(combo)  # raises ValueError on a bad combo
                wrapped[combo] = fn
            self._listener = keyboard.GlobalHotKeys(wrapped)
            self._listener.daemon = True
            self._listener.start()
            return True
        except Exception as exc:
            self.error = str(exc)
            log.warning("Global hotkeys unavailable: %s", exc)
            return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # pragma: no cover
                pass
            self._listener = None
