"""The ONE shared context both models see: facts, recent turns, documents, usage counters.

SQLite, local only. Facts are tagged `shared` (may be sent to the cloud model) or `private`
(only ever sent to the local model). Obvious secrets are refused outright.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .. import paths
from ..errors import ConfigError

MAX_FACT_CHARS = 300
_SENSITIVE = [
    (re.compile(r"\b(password|passcode|passwd|pin|otp|cvv|cvc|secret|api[ _-]?key|token|bearer|private key)\b", re.I), "looks like a secret"),
    (re.compile(r"(?:\d[ -]?){9,}"), "contains a long number (card/ID/phone?)"),
    (re.compile(r"\b(AIza[0-9A-Za-z_\-]{10,}|sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|github_pat_\w+)"), "looks like an API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
]
_WORD = re.compile(r"[a-z0-9']+")
_STOP = {"the", "a", "an", "is", "are", "my", "me", "i", "to", "of", "and", "what", "whats", "for", "at", "in", "on", "it", "do", "does", "when", "who", "how"}


def looks_sensitive(text: str) -> Optional[str]:
    for rx, why in _SENSITIVE:
        if rx.search(text):
            return why
    return None


def keywords(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


@dataclass(frozen=True)
class Fact:
    id: int
    text: str
    scope: str   # "shared" | "private"
    uses: int


class ContextStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else paths.config_dir() / "brain.sqlite3"
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL,
                scope TEXT NOT NULL CHECK(scope IN ('shared','private')), created REAL, last_used REAL, uses INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, ts REAL, role TEXT, text TEXT);
            CREATE TABLE IF NOT EXISTS docs(name TEXT PRIMARY KEY, date TEXT, payload TEXT, updated REAL);
            CREATE TABLE IF NOT EXISTS usage(day TEXT, provider TEXT, requests INTEGER DEFAULT 0,
                tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, PRIMARY KEY(day, provider));
        """)
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------ facts
    def remember(self, text: str, scope: str = "shared", now: Optional[float] = None) -> Tuple[int, bool]:
        text = " ".join(text.split())
        if not text:
            raise ConfigError("Nothing to remember")
        if len(text) > MAX_FACT_CHARS:
            raise ConfigError(f"Too long (max {MAX_FACT_CHARS} characters)")
        why = looks_sensitive(text)
        if why:
            raise ConfigError(f"Not stored: {why}")
        if scope not in ("shared", "private"):
            raise ConfigError("scope must be shared or private")
        now = time.time() if now is None else now
        with self._lock:
            row = self._db.execute("SELECT id FROM facts WHERE lower(text)=lower(?)", (text,)).fetchone()
            if row:
                self._db.execute("UPDATE facts SET scope=?, last_used=? WHERE id=?", (scope, now, row[0]))
                self._db.commit()
                return row[0], False
            cur = self._db.execute("INSERT INTO facts(text, scope, created, last_used) VALUES(?,?,?,?)", (text, scope, now, now))
            self._db.commit()
            return int(cur.lastrowid), True

    def forget(self, query: str) -> int:
        q = query.strip().lower()
        if not q:
            return 0
        with self._lock:
            cur = self._db.execute("DELETE FROM facts WHERE lower(text) LIKE ?", (f"%{q}%",))
            self._db.commit()
            return cur.rowcount

    def facts(self) -> List[Fact]:
        with self._lock:
            rows = self._db.execute("SELECT id, text, scope, uses FROM facts ORDER BY uses DESC, last_used DESC").fetchall()
        return [Fact(*r) for r in rows]

    def relevant_facts(self, query: str, limit: int = 8) -> List[Fact]:
        """Facts sharing words with the query (plus the most-used few), best first."""
        q = keywords(query)
        scored = []
        for f in self.facts():
            overlap = len(q & keywords(f.text))
            if overlap or f.uses >= 3:
                scored.append((overlap * 10 + min(f.uses, 9), f))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:limit]]

    def mark_used(self, ids: List[int]) -> None:
        if not ids:
            return
        with self._lock:
            self._db.executemany("UPDATE facts SET uses = uses + 1, last_used=? WHERE id=?", [(time.time(), i) for i in ids])
            self._db.commit()

    # ------------------------------------------------------------ conversation
    def add_turn(self, role: str, text: str, now: Optional[float] = None) -> None:
        with self._lock:
            self._db.execute("INSERT INTO turns(ts, role, text) VALUES(?,?,?)", (time.time() if now is None else now, role, text[:500]))
            self._db.execute("DELETE FROM turns WHERE id NOT IN (SELECT id FROM turns ORDER BY id DESC LIMIT 200)")
            self._db.commit()

    def recent_turns(self, n: int = 4, max_age_s: float = 600.0, now: Optional[float] = None) -> List[Tuple[str, str]]:
        cutoff = (time.time() if now is None else now) - max_age_s
        with self._lock:
            rows = self._db.execute("SELECT role, text FROM turns WHERE ts >= ? ORDER BY id DESC LIMIT ?", (cutoff, n)).fetchall()
        return list(reversed(rows))

    # ------------------------------------------------------------ documents (e.g. today's mess menu)
    def put_doc(self, name: str, date: str, payload: str, now: Optional[float] = None) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO docs(name, date, payload, updated) VALUES(?,?,?,?)",
                             (name, date, payload, time.time() if now is None else now))
            self._db.commit()

    def get_doc(self, name: str) -> Optional[Tuple[str, str, float]]:
        with self._lock:
            return self._db.execute("SELECT date, payload, updated FROM docs WHERE name=?", (name,)).fetchone()

    # ------------------------------------------------------------ usage (for the cloud budget)
    def add_usage(self, day: str, provider: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO usage(day, provider) VALUES(?,?)", (day, provider))
            self._db.execute("UPDATE usage SET requests=requests+1, tokens_in=tokens_in+?, tokens_out=tokens_out+? WHERE day=? AND provider=?",
                             (tokens_in, tokens_out, day, provider))
            self._db.commit()

    def usage(self, day: str, provider: str) -> Tuple[int, int, int]:
        with self._lock:
            row = self._db.execute("SELECT requests, tokens_in, tokens_out FROM usage WHERE day=? AND provider=?", (day, provider)).fetchone()
        return row or (0, 0, 0)
