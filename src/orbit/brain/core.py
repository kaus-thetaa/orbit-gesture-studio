"""Brain: the facade the rest of ORBIT talks to."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, List, Optional

from .. import paths
from ..settings import Settings
from .assistant import Answer, Assistant
from .context import ContextStore
from .netmon import NetMonitor
from .providers import GeminiProvider, OllamaProvider, ProviderError
from .router import Router
from .routines import RoutineRunner, RoutineScheduler, parse_routines
from .secrets import get_secret

log = logging.getLogger(__name__)


class Brain:
    def __init__(self, settings: Settings, store: Optional[ContextStore] = None, gemini=None, ollama=None,
                 net: Optional[NetMonitor] = None, now: Callable[[], datetime] = datetime.now,
                 on_event: Optional[Callable[[str], None]] = None) -> None:
        self.settings, self.cfg = settings, settings.brain
        c = self.cfg
        self.store = store or ContextStore()
        host, port = _host_port(c.gemini_base_url)
        self.net = net or NetMonitor(host=host, port=port)
        self.gemini = gemini if gemini is not None else GeminiProvider(
            c.gemini_model, lambda: get_secret("GEMINI_API_KEY"), c.gemini_base_url, c.cloud_timeout_s,
            state_path=paths.config_dir() / "gemini_state.json")
        self.ollama = ollama if ollama is not None else OllamaProvider(
            c.ollama_model, c.ollama_host, c.ollama_keep_alive, c.local_timeout_s)
        self.router = Router(c, self.store, self.gemini, self.ollama, self.net)
        self.assistant = Assistant(c, self.store, self.router, now=now)
        routines, self.warnings = parse_routines(c.routines)
        self.on_event = on_event
        self.runner = RoutineRunner(self.store, self.router, now=now)
        self.scheduler = RoutineScheduler(routines, self.store, self.runner, self.net, now=now,
                                          on_event=lambda m: self.on_event and self.on_event(m))
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="orbit-brain")

    def start(self) -> None:
        self.net.start()
        self.scheduler.start()
        if self.cfg.warm_local_model and self.ollama is not None:
            threading.Thread(target=self._warm, daemon=True, name="orbit-warm").start()

    def _warm(self) -> None:
        try:
            self.ollama.warm()
            log.info("local model warmed")
        except ProviderError as exc:
            log.info("local model not warmed: %s", exc)

    def stop(self) -> None:
        self.scheduler.stop()
        self.net.stop()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def ask(self, text: str) -> Answer:
        return self.assistant.ask(text)

    def ask_async(self, text: str, callback: Callable[[Answer], None]) -> None:
        def work() -> None:
            try:
                callback(self.assistant.ask(text))
            except Exception:  # noqa: BLE001 - never lose the callback
                log.exception("ask failed")
                callback(Answer("Error", "error", 0.0))
        self._pool.submit(work)

    def status(self) -> dict:
        st = self.router.status()
        st["routines"] = [r.name for r in self.scheduler.routines]
        return st


def _host_port(url: str):
    from urllib.parse import urlparse
    u = urlparse(url)
    return u.hostname or "generativelanguage.googleapis.com", u.port or (80 if u.scheme == "http" else 443)
