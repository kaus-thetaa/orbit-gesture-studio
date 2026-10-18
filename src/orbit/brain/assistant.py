"""The assistant: instant skills first, then ONE short model call with the shared context."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from ..errors import ConfigError
from ..settings import BrainSettings
from .context import ContextStore
from .providers import LLMRequest
from .router import AllProvidersFailed, Router
from .skills import MenuSkill, TimeSkill

log = logging.getLogger(__name__)

_FILLER = re.compile(r"^(?:sure|certainly|of course|okay|ok|absolutely|alright)[!,.]*\s+", re.I)
_MD = re.compile(r"[*_`#>]+|^\s*[-•]\s+", re.M)
_REMEMBER = re.compile(r"^(?:please\s+)?(?:remember|note|save)(?:\s+that|\s+this)?\s*[:,]?\s*(.+)$", re.I | re.S)
_PRIVATE = re.compile(r"^(?:privately|private|locally|local only|in private)\s*[:,]?\s*(.+)$", re.I | re.S)
_FORGET = re.compile(r"^(?:please\s+)?forget(?:\s+about|\s+that)?\s*[:,]?\s*(.+)$", re.I | re.S)
_LIST = re.compile(r"^(?:what do you know(?: about me)?|show (?:my )?memory|list (?:my )?(?:memory|facts)|memory)\s*\??$", re.I)


@dataclass
class Answer:
    text: str
    source: str            # time | menu | memory | gemini | ollama | error
    latency_ms: float
    note: str = ""


def shorten(text: str, max_words: int) -> str:
    """One line, plain text, at most max_words."""
    text = _MD.sub(" ", text or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    line = _FILLER.sub("", lines[0])
    words = line.split()
    if len(words) > max_words:
        first = re.split(r"(?<=[.!?])\s+", line, maxsplit=1)[0]
        words = first.split() if 0 < len(first.split()) <= max_words else words[:max_words]
    return " ".join(words).rstrip(",;:-").strip()


class Assistant:
    def __init__(self, settings: BrainSettings, store: ContextStore, router: Router,
                 now: Callable[[], datetime] = datetime.now) -> None:
        self.s, self.store, self.router, self.now = settings, store, router, now
        self.menu = MenuSkill(store, lambda: self.now().date().isoformat(), now)
        self.time = TimeSkill(now)

    # ------------------------------------------------------------------ public
    def ask(self, text: str) -> Answer:
        t0 = time.perf_counter()
        text = " ".join((text or "").split())
        if not text:
            return Answer("", "error", 0.0)
        done = lambda a, src, note="": Answer(a, src, (time.perf_counter() - t0) * 1000, note)

        cmd = self._command(text)
        if cmd is not None:
            return done(cmd, "memory")
        for skill, src in ((self.time, "time"), (self.menu, "menu")):
            ans = skill.try_answer(text)
            if ans is not None:
                self.store.add_turn("user", text)
                self.store.add_turn("assistant", ans)
                return done(ans, src)

        try:
            req, local_only, used = self._build_request(text)
            reply = self.router.ask(req, local_only=local_only)
        except AllProvidersFailed as exc:
            log.warning("no model answered: %s", exc.reasons)
            if local_only:
                msg = "Private: needs Ollama"
            elif self.router.cloud_state() != "ok":
                msg = "Offline: start Ollama"
            else:
                msg = "No answer"
            return done(msg, "error", "; ".join(exc.reasons))
        out = shorten(reply.text, self.s.max_words) or "Don't know"
        self.store.mark_used(used)
        self.store.add_turn("user", text)
        self.store.add_turn("assistant", out)
        return done(out, reply.provider, reply.model)

    # ------------------------------------------------------------------ memory commands
    def _command(self, text: str) -> Optional[str]:
        m = _FORGET.match(text)
        if m:
            n = self.store.forget(m.group(1))
            return "Forgotten" if n else "Nothing matched"
        if _LIST.match(text):
            facts = self.store.facts()
            if not facts:
                return "Nothing yet"
            return f"{len(facts)} facts: " + "; ".join(f.text for f in facts[:3])[:120]
        m = _REMEMBER.match(text)
        if m:
            body, scope = m.group(1).strip(), "shared"
            pm = _PRIVATE.match(body)
            if pm:
                body, scope = pm.group(1).strip(), "private"
            try:
                _, created = self.store.remember(body, scope)
            except ConfigError as exc:
                return str(exc)[:80]
            return ("Noted (private)" if scope == "private" else "Noted") if created else "Already known"
        return None

    # ------------------------------------------------------------------ model path
    def _system(self, facts: list, doc_line: str) -> str:
        now = self.now()
        lines = [
            "You are ORBIT, a minimal assistant living on the user's laptop screen.",
            f"Reply with ONE short line, at most {self.s.max_words} words. Plain text: no markdown, no preamble, no apology.",
            "Use the facts and documents below; if you don't know, reply exactly: Don't know",
            f"Now: {now.strftime('%A %d %B %Y, %H:%M')}",
        ]
        if facts:
            lines.append("Facts about the user:")
            lines += [f"- {f.text}" for f in facts]
        if doc_line:
            lines.append(doc_line)
        return "\n".join(lines)[: self.s.context_chars]

    def _build_request(self, text: str):
        facts = self.store.relevant_facts(text)
        local_only = any(f.scope == "private" for f in facts)
        if facts and not self.s.share_memory_with_cloud:
            local_only = True                 # cloud sharing disabled: facts may only go to the local model
        turns = self.store.recent_turns(4)
        msgs: List = [(r, t) for r, t in turns] + [("user", text)]
        req = LLMRequest(self._system(facts, self.menu.context_line()), msgs, max_tokens=self.s.max_output_tokens)
        return req, local_only, [f.id for f in facts]
