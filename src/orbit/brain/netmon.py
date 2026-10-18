"""Is the internet reachable? Cheap TCP probe, cached, with change notifications."""
from __future__ import annotations

import logging
import socket
import threading
from typing import Callable, List, Optional

log = logging.getLogger(__name__)


def tcp_probe(host: str, port: int = 443, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class NetMonitor:
    def __init__(self, host: str = "generativelanguage.googleapis.com", port: int = 443, interval_s: float = 15.0,
                 offline_interval_s: float = 4.0, probe: Optional[Callable[[], bool]] = None) -> None:
        self.probe = probe or (lambda: tcp_probe(host, port))
        self.interval, self.offline_interval = interval_s, offline_interval_s
        self.online: Optional[bool] = None          # None until the first check
        self._listeners: List[Callable[[bool], None]] = []
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def add_listener(self, cb: Callable[[bool], None]) -> None:
        self._listeners.append(cb)

    def check(self) -> bool:
        now_online = bool(self.probe())
        changed = now_online != self.online
        first = self.online is None
        self.online = now_online
        if changed:
            log.info("network: %s", "online" if now_online else "offline")
            for cb in list(self._listeners):
                try:
                    cb(now_online)
                except Exception:
                    log.exception("network listener failed")
        return now_online

    def check_soon(self) -> None:
        self._wake.set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="orbit-net", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.check()
            self._wake.wait(self.interval if self.online else self.offline_interval)
            self._wake.clear()
