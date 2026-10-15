"""Executes actions on a worker thread, away from the camera/vision loop.

* bounded queue - a slow action can never block frame capture; if it backs up, new
  requests are dropped (and logged) instead of piling up
* every request is re-validated against the allowlist right before it runs
* exceptions are caught and reported, never propagated into the vision thread
* when disarmed, only *deactivate* requests (which undo things) and explicit manual
  tests (force=True) are accepted
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .actions.base import ActionContext, ActionRegistry
from .errors import ConfigError
from .safety import SafetyController

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionRequest:
    mapping_id: str
    action: str
    phase: str  # "activate" | "deactivate"
    params: Dict[str, Any]
    mode: str = "trigger"


@dataclass(frozen=True)
class ActionResult:
    request: ActionRequest
    ok: bool
    error: str = ""
    duration_ms: float = 0.0


class ActionDispatcher:
    def __init__(self, registry: ActionRegistry, ctx: ActionContext, safety: SafetyController,
                 on_result: Optional[Callable[[ActionResult], None]] = None, queue_size: int = 32) -> None:
        self.registry, self.ctx, self.safety, self.on_result = registry, ctx, safety, on_result
        self._q: "queue.Queue[Optional[ActionRequest]]" = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---- lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="orbit-dispatcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.purge()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    # ---- API
    def submit(self, req: ActionRequest, force: bool = False) -> bool:
        if req.phase == "activate" and not force and not self.safety.armed:
            log.debug("dropped %s/%s: disarmed", req.action, req.phase)
            return False
        try:
            self._q.put_nowait(req)
            return True
        except queue.Full:
            log.error("action queue full - dropped %s/%s", req.action, req.phase)
            return False

    def purge(self) -> int:
        """Drop everything not yet started (used by the emergency stop)."""
        n = 0
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return n
            self._q.task_done()
            if item is not None:
                n += 1

    def join_idle(self, timeout: float = 2.0) -> bool:
        """Block until the queue is drained (tests / shutdown)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._q.unfinished_tasks == 0:
                return True
            time.sleep(0.005)
        return False

    # ---- worker
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if req is None:
                    return
                self._execute(req)
            finally:
                self._q.task_done()

    def _execute(self, req: ActionRequest) -> None:
        t0 = time.perf_counter()
        ok, err = True, ""
        try:
            action = self.registry.get(req.action)
            params = action.validate(req.params, self.ctx.denylist)
            if req.phase == "activate":
                action.activate(self.ctx, params)
            elif req.phase == "deactivate":
                action.deactivate(self.ctx, params)
            else:
                raise ConfigError(f"unknown phase {req.phase!r}")
            log.info("action %s/%s ok (mapping %s)", req.action, req.phase, req.mapping_id)
        except Exception as exc:
            ok, err = False, f"{type(exc).__name__}: {exc}"
            log.error("action %s/%s FAILED (mapping %s): %s", req.action, req.phase, req.mapping_id, err)
        result = ActionResult(req, ok, err, (time.perf_counter() - t0) * 1000)
        if self.on_result:
            try:
                self.on_result(result)
            except Exception:
                log.exception("on_result callback failed")
