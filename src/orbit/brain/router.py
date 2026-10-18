"""Chooses Gemini or Ollama per question - both get the SAME context.

Policy
  * private memory  -> local model only (never leaves the machine)
  * online, key set, budget left, not cooling down -> Gemini first, Ollama as fallback
  * offline / no key / budget nearly used / rate-limited / repeatedly failing -> Ollama
  * hedging: if the first provider hasn't answered after `hedge_after_ms`, ALSO ask the
    other one and use whichever answers first (latency is the priority)
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

from ..settings import BrainSettings
from .context import ContextStore
from .providers import (AuthError, LLMReply, LLMRequest, ProviderError, ProviderUnavailable, RateLimited)

log = logging.getLogger(__name__)


class AllProvidersFailed(ProviderError):
    def __init__(self, msg: str, reasons: List[str]) -> None:
        super().__init__(msg)
        self.reasons = reasons


@dataclass
class Breaker:
    fails: int = 0
    open_until: float = 0.0
    reason: str = ""

    def is_open(self, now: float) -> bool:
        return now < self.open_until


class Router:
    def __init__(self, settings: BrainSettings, store: ContextStore, gemini=None, ollama=None, net=None,
                 clock: Callable[[], float] = time.monotonic, today: Callable[[], str] = lambda: date.today().isoformat()) -> None:
        self.s, self.store, self.gemini, self.ollama, self.net = settings, store, gemini, ollama, net
        self.clock, self.today = clock, today
        self.breakers: Dict[str, Breaker] = {"gemini": Breaker(), "ollama": Breaker()}
        self.last_provider: Optional[str] = None
        self.last_error: str = ""

    # ------------------------------------------------------------------ availability
    def cloud_state(self) -> str:
        g = self.gemini
        if g is None or not g.configured():
            return "no-key"
        if self.net is not None and self.net.online is False:
            return "offline"
        if self.breakers["gemini"].is_open(self.clock()):
            return "cooldown:" + (self.breakers["gemini"].reason or "errors")
        budget = self.s.gemini_daily_requests
        if budget and self.store.usage(self.today(), "gemini")[0] >= budget * self.s.switch_at_fraction:
            return "budget"
        return "ok"

    def local_state(self) -> str:
        if self.ollama is None:
            return "off"
        return "cooldown" if self.breakers["ollama"].is_open(self.clock()) else "ok"

    def plan(self, local_only: bool = False) -> list:
        cloud = self.gemini if (not local_only and self.cloud_state() == "ok") else None
        local = self.ollama if self.local_state() == "ok" else None
        if local_only:
            return [local] if local else []
        pref = self.s.prefer
        if pref == "ollama":
            order = [local, cloud]
        elif pref == "gemini":
            order = [cloud, local]
        else:
            order = [cloud, local]            # auto: cloud when usable, otherwise local is all that's left
        return [p for p in order if p is not None]

    # ------------------------------------------------------------------ bookkeeping
    def _ok(self, p, reply: LLMReply) -> None:
        self.breakers[p.name] = Breaker()
        self.last_provider, self.last_error = p.name, ""
        self.store.add_usage(self.today(), p.name, reply.tokens_in, reply.tokens_out)

    def _fail(self, p, exc: Exception) -> None:
        b = self.breakers[p.name]
        now = self.clock()
        self.last_error = f"{p.name}: {exc}"
        if isinstance(exc, RateLimited):
            b.open_until, b.reason = now + max(exc.retry_after, 5.0), "rate-limited"
        elif isinstance(exc, AuthError):
            b.open_until, b.reason = now + 600.0, "auth"
        else:
            b.fails += 1
            if b.fails >= 2 or isinstance(exc, ProviderUnavailable):
                b.open_until, b.reason = now + (20.0 if p.name == "gemini" else 10.0), "errors"
        if p.name == "gemini" and self.net is not None and isinstance(exc, ProviderUnavailable):
            self.net.check_soon()
        log.warning("%s failed: %s", p.name, exc)

    # ------------------------------------------------------------------ ask
    def ask(self, req: LLMRequest, local_only: bool = False) -> LLMReply:
        plan = self.plan(local_only)
        if not plan:
            raise AllProvidersFailed(
                "no model available", [f"cloud: {self.cloud_state()}", f"local: {self.local_state()}"] + ([self.last_error] if self.last_error else []))
        q: "queue.Queue[Tuple[object, Optional[LLMReply], Optional[Exception]]]" = queue.Queue()

        def work(p) -> None:
            try:
                q.put((p, p.generate(req), None))
            except Exception as exc:  # noqa: BLE001 - reported through the queue
                q.put((p, None, exc))

        state = {"idx": 0, "pending": 0}

        def launch() -> None:
            p = plan[state["idx"]]
            state["idx"] += 1
            state["pending"] += 1
            threading.Thread(target=work, args=(p,), daemon=True, name=f"orbit-ask-{p.name}").start()

        launch()
        hedge = self.s.hedge_after_ms / 1000.0
        overall = (req.timeout_s or max(self.s.cloud_timeout_s, self.s.local_timeout_s)) + 2.0
        end, errors = time.monotonic() + overall, []
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise AllProvidersFailed("timed out", errors or ["timeout"])
            can_hedge = hedge > 0 and state["idx"] < len(plan)
            try:
                p, reply, exc = q.get(timeout=max(0.001, min(hedge, remaining) if can_hedge else remaining))
            except queue.Empty:
                if can_hedge:
                    log.info("slow answer from %s; also asking %s", plan[0].name, plan[state["idx"]].name)
                    launch()
                continue
            state["pending"] -= 1
            if exc is None and reply is not None:
                self._ok(p, reply)
                return reply
            self._fail(p, exc)  # type: ignore[arg-type]
            errors.append(f"{p.name}: {exc}")
            if state["idx"] < len(plan):
                launch()
            elif state["pending"] == 0:
                raise AllProvidersFailed("all models failed", errors)

    def status(self) -> dict:
        req, _, _ = self.store.usage(self.today(), "gemini")
        return {"cloud": self.cloud_state(), "local": self.local_state(), "last": self.last_provider,
                "gemini_requests_today": req, "budget": self.s.gemini_daily_requests, "online": getattr(self.net, "online", None)}
