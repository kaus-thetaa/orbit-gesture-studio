"""Applies activation modes (hold / toggle / trigger) to stabilised gesture events."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

from ..dispatcher import ActionDispatcher, ActionRequest
from ..mapping import Mapping, MappingStore
from ..safety import SafetyController
from ..vision.types import GestureEvent

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    mapping_id: str
    gesture: str
    action: str
    phase: str
    executed: bool
    reason: str = ""


class MappingEngine:
    def __init__(self, store: MappingStore, dispatcher: ActionDispatcher, safety: SafetyController,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store, self.dispatcher, self.safety, self.clock = store, dispatcher, safety, clock
        self._lock = threading.RLock()
        self._active: Dict[str, Mapping] = {}      # mapping id -> mapping (currently "on")
        self._last_fire: Dict[str, float] = {}
        self.suspended = False                      # e.g. while recording a gesture

    # ------------------------------------------------------------------ queries
    def active_ids(self) -> List[str]:
        with self._lock:
            return list(self._active)

    # ------------------------------------------------------------------ events
    def handle(self, ev: GestureEvent) -> List[Decision]:
        with self._lock:
            return self._pressed(ev) if ev.kind == "pressed" else self._released(ev)

    def _req(self, m: Mapping, phase: str) -> ActionRequest:
        return ActionRequest(m.id, m.action, phase, dict(m.params), m.mode)

    def _pressed(self, ev: GestureEvent) -> List[Decision]:
        m = self.store.for_gesture(ev.name)
        if m is None:
            return []
        if self.suspended:
            return [Decision(m.id, m.gesture, m.action, "activate", False, "suspended")]
        if not self.safety.armed:
            return [Decision(m.id, m.gesture, m.action, "activate", False, "dry run (disarmed)")]
        if m.mode == "hold" and m.id in self._active:
            return []  # already on; idempotent
        now = self.clock()
        last = self._last_fire.get(m.id)
        if last is not None and (now - last) * 1000 < m.cooldown_ms:
            return [Decision(m.id, m.gesture, m.action, "activate", False, "cooldown")]

        if m.mode == "toggle" and m.id in self._active:
            ok = self.dispatcher.submit(self._req(m, "deactivate"))
            self._active.pop(m.id, None)
            self._last_fire[m.id] = now
            return [Decision(m.id, m.gesture, m.action, "deactivate", ok)]

        ok = self.dispatcher.submit(self._req(m, "activate"))
        if ok and m.mode in ("hold", "toggle"):
            self._active[m.id] = m
        self._last_fire[m.id] = now
        return [Decision(m.id, m.gesture, m.action, "activate", ok, "" if ok else "queue rejected")]

    def _released(self, ev: GestureEvent) -> List[Decision]:
        out: List[Decision] = []
        for mid, m in list(self._active.items()):
            if m.gesture == ev.name and m.mode == "hold":
                ok = self.dispatcher.submit(self._req(m, "deactivate"))
                self._active.pop(mid, None)
                self._last_fire[mid] = self.clock()
                out.append(Decision(m.id, m.gesture, m.action, "deactivate", ok))
        return out

    # ------------------------------------------------------------------ safety
    def release_all(self, reason: str = "") -> List[Decision]:
        """Undo everything currently on (disarm, e-stop, camera lost, shutdown)."""
        out: List[Decision] = []
        with self._lock:
            for mid, m in list(self._active.items()):
                ok = self.dispatcher.submit(self._req(m, "deactivate"))
                out.append(Decision(m.id, m.gesture, m.action, "deactivate", ok, reason))
            self._active.clear()
        if out:
            log.warning("released %d active mapping(s): %s", len(out), reason)
        return out

    def set_suspended(self, value: bool) -> None:
        self.suspended = value
        if value:
            self.release_all("suspended")
