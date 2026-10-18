"""Gemini (cloud) and Ollama (local) behind one interface.

Speed notes: each provider keeps ONE persistent HTTP(S) connection (no TLS handshake per
question) and requests tiny outputs. Model names and thinking parameters change often, so
both are discovered/adapted at runtime instead of being hard-coded.

Only the standard library is used (urllib/http.client) - no extra dependencies.
"""
from __future__ import annotations

import http.client
import json
import logging
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..errors import OrbitError

log = logging.getLogger(__name__)


class ProviderError(OrbitError):
    """Generic failure (bad response, unsupported parameter...)."""


class ProviderUnavailable(ProviderError):
    """Offline, refused, timed out, or a 5xx - try the other provider."""


class RateLimited(ProviderError):
    def __init__(self, msg: str, retry_after: float = 30.0) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class AuthError(ProviderError):
    """Missing/invalid API key."""


@dataclass
class LLMRequest:
    system: str
    messages: List[Tuple[str, str]]          # (role, text), role in {"user", "assistant"}
    max_tokens: int = 60
    temperature: float = 0.2
    json_mode: bool = False
    timeout_s: Optional[float] = None        # override the provider default


@dataclass
class LLMReply:
    text: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


# ====================================================================== persistent HTTP
class HttpClient:
    """One reusable connection; transparently reconnects if the server closed it."""

    def __init__(self, base_url: str, default_timeout: float = 10.0) -> None:
        u = urlparse(base_url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"bad base url {base_url!r}")
        self.scheme, self.host, self.port = u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.prefix = u.path.rstrip("/")
        self.timeout = default_timeout
        self._conn: Optional[http.client.HTTPConnection] = None
        self._lock = threading.Lock()

    def _connect(self, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        self._conn = cls(self.host, self.port, timeout=timeout)
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def request(self, method: str, path: str, body: Optional[dict] = None, headers: Optional[Dict[str, str]] = None,
                timeout: Optional[float] = None) -> Tuple[int, Dict[str, str], bytes]:
        timeout = timeout or self.timeout
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        hdrs = {"Content-Type": "application/json", "Accept": "application/json", **(headers or {})}
        with self._lock:
            for attempt in (0, 1):
                reused = self._conn is not None
                conn = self._conn or self._connect(timeout)
                try:
                    conn.timeout = timeout
                    if conn.sock is not None:
                        conn.sock.settimeout(timeout)
                    conn.request(method, self.prefix + path, body=payload, headers=hdrs)
                    resp = conn.getresponse()
                    data = resp.read()
                    if resp.getheader("Connection", "").lower() == "close":
                        self._conn = None
                        conn.close()
                    return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
                except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError, http.client.CannotSendRequest) as exc:
                    self._conn = None            # stale keep-alive connection: reconnect once
                    try:
                        conn.close()
                    except Exception:
                        pass
                    if attempt == 1:
                        raise ProviderUnavailable(f"connection lost: {exc}") from exc
                except (socket.timeout, TimeoutError) as exc:
                    self._conn = None
                    raise ProviderUnavailable(f"timed out after {timeout:.0f}s") from exc
                except (OSError, http.client.HTTPException) as exc:   # refused, DNS failure, TLS, dead socket...
                    self._conn = None
                    if reused and attempt == 0:      # a previously-good connection died: reconnect once
                        continue
                    raise ProviderUnavailable(f"{type(exc).__name__}: {exc}") from exc
        raise ProviderUnavailable("unreachable")  # pragma: no cover


def _json(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8", "replace"))
    except ValueError as exc:
        raise ProviderError(f"invalid JSON from server: {data[:120]!r}") from exc


def _err_text(data: bytes) -> str:
    try:
        j = json.loads(data.decode("utf-8", "replace"))
        e = j.get("error", j)
        return (e.get("message") if isinstance(e, dict) else str(e)) or ""
    except Exception:
        return data[:200].decode("utf-8", "replace")


# ====================================================================== Gemini
class GeminiProvider:
    name = "gemini"
    # Tried in order until one is accepted (Gemini 3 uses thinkingLevel; 2.5 uses thinkingBudget).
    # Thinking tokens add latency and eat the output budget, so we ask for as little as possible.
    THINKING_VARIANTS: List[Optional[dict]] = [{"thinkingLevel": "minimal"}, {"thinkingLevel": "low"}, {"thinkingBudget": 0}, None]

    def __init__(self, model: str, key_fn: Callable[[], Optional[str]], base_url: str, timeout: float = 6.0,
                 state_path: Optional[Path] = None) -> None:
        self.model, self._key_fn, self.timeout = model, key_fn, timeout
        self.http = HttpClient(base_url, timeout)
        self._variant = 0
        self._resolved = False
        self.last_attempts = 0
        self._state_path = Path(state_path) if state_path else None
        self._load_state()

    def _load_state(self) -> None:
        """Remember which thinking setting this model accepted, so later runs don't re-probe (each probe is a round trip)."""
        try:
            if self._state_path and self._state_path.exists():
                saved = json.loads(self._state_path.read_text()).get(self.model)
                if isinstance(saved, int) and 0 <= saved < len(self.THINKING_VARIANTS):
                    self._variant = saved
        except (OSError, ValueError):
            pass

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            data = json.loads(self._state_path.read_text()) if self._state_path.exists() else {}
            data[self.model] = self._variant
            self._state_path.write_text(json.dumps(data))
        except (OSError, ValueError):
            pass

    def configured(self) -> bool:
        return bool(self._key_fn())

    def _headers(self) -> Dict[str, str]:
        key = self._key_fn()
        if not key:
            raise AuthError("No Gemini API key (set GEMINI_API_KEY or run: python -m orbit brain set-key)")
        return {"x-goog-api-key": key}

    def list_models(self) -> List[str]:
        status, _, data = self.http.request("GET", "/v1beta/models?pageSize=200", headers=self._headers())
        self._raise_for(status, data)
        out = []
        for m in _json(data).get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                out.append(m["name"].split("/", 1)[-1])
        return out

    def _auto_pick(self) -> bool:
        """The configured model id doesn't exist (names change): pick the fastest-looking available one."""
        try:
            models = self.list_models()
        except ProviderError:
            return False
        for needle in ("flash-lite", "flash"):
            cands = sorted(m for m in models if needle in m and "preview" not in m and "image" not in m and "tts" not in m)
            cands = cands or sorted(m for m in models if needle in m and "image" not in m and "tts" not in m)
            if cands:
                log.warning("Gemini model %r unavailable; using %r (set brain.gemini_model to pin one)", self.model, cands[-1])
                self.model = cands[-1]
                return True
        return False

    def _raise_for(self, status: int, data: bytes, headers: Optional[Dict[str, str]] = None) -> None:
        if status < 400:
            return
        msg = _err_text(data)
        if status == 429:
            try:
                ra = float((headers or {}).get("retry-after", "30"))
            except ValueError:
                ra = 30.0
            raise RateLimited(f"Gemini rate limit/quota: {msg}", ra)
        if status in (401, 403):
            raise AuthError(f"Gemini rejected the key ({status}): {msg}")
        if status >= 500:
            raise ProviderUnavailable(f"Gemini server error {status}: {msg}")
        raise ProviderError(f"Gemini error {status}: {msg}")

    def generate(self, req: LLMRequest) -> LLMReply:
        t0 = time.perf_counter()
        contents = [{"role": "user" if r == "user" else "model", "parts": [{"text": t}]} for r, t in req.messages]
        self.last_attempts = 0
        for _ in range(len(self.THINKING_VARIANTS) + 1):
            self.last_attempts += 1
            gen = {"maxOutputTokens": max(req.max_tokens, 48), "temperature": req.temperature}
            if req.json_mode:
                gen["responseMimeType"] = "application/json"
            variant = self.THINKING_VARIANTS[self._variant]
            if variant:
                gen["thinkingConfig"] = variant
            body = {"system_instruction": {"parts": [{"text": req.system}]}, "contents": contents, "generationConfig": gen}
            status, hdrs, data = self.http.request(
                "POST", f"/v1beta/models/{self.model}:generateContent", body, self._headers(), req.timeout_s or self.timeout)
            if status == 404 and not self._resolved:
                self._resolved = True
                if self._auto_pick():
                    continue
            if status == 400 and variant is not None and "think" in _err_text(data).lower():
                self._variant += 1            # this model doesn't accept that thinking setting
                log.info("Gemini %s rejected thinking setting %s; trying %s", self.model, variant, self.THINKING_VARIANTS[min(self._variant, len(self.THINKING_VARIANTS) - 1)])
                self._save_state()
                continue
            self._raise_for(status, data, hdrs)
            j = _json(data)
            cands = j.get("candidates") or []
            text = "".join(p.get("text", "") for p in (cands[0].get("content", {}).get("parts", []) if cands else [])).strip()
            if not text:
                reason = (cands[0].get("finishReason") if cands else None) or j.get("promptFeedback", {}).get("blockReason") or "empty"
                raise ProviderError(f"Gemini returned no text ({reason})")
            u = j.get("usageMetadata", {})
            return LLMReply(text, self.name, self.model, int(u.get("promptTokenCount", 0)),
                            int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0)), (time.perf_counter() - t0) * 1000)
        raise ProviderError("Gemini: no accepted request variant")


# ====================================================================== Ollama
_NOT_GENERAL = re.compile(r"coder|code|embed|nomic|rerank|llava|vision|bge|minilm", re.I)


def pick_ollama_model(requested: str, models: List[Tuple[str, int]]) -> Optional[str]:
    """Which installed model to use: the requested one, else the same family, else the smallest GENERAL model."""
    names = [n for n, _ in models]
    if not names:
        return None
    if requested in names:
        return requested
    base = requested.split(":")[0]
    same = [n for n in names if n.split(":")[0] == base]
    if same:
        return next((n for n in same if n.endswith(":latest")), same[0])
    general = [m for m in models if not _NOT_GENERAL.search(m[0])]
    return min(general or models, key=lambda m: m[1])[0]


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, host: str, keep_alive: str = "30m", timeout: float = 25.0) -> None:
        self.model, self.keep_alive, self.timeout = model, keep_alive, timeout
        self.http = HttpClient(host, timeout)
        self._think_ok = True
        self._checked = False

    def configured(self) -> bool:
        return True

    def installed(self) -> List[Tuple[str, int]]:
        status, _, data = self.http.request("GET", "/api/tags", timeout=3.0)
        if status >= 400:
            raise ProviderUnavailable(f"Ollama /api/tags -> {status}")
        return [(m.get("name", ""), int(m.get("size", 0))) for m in _json(data).get("models", [])]

    def _ensure_model(self) -> None:
        """Use the configured model if installed, otherwise the smallest installed one."""
        if self._checked:
            return
        models = self.installed()
        pick = pick_ollama_model(self.model, models)
        if pick is None:
            raise ProviderUnavailable("Ollama has no models; run: ollama pull " + self.model)
        if pick != self.model:
            same_family = pick.split(":")[0] == self.model.split(":")[0]
            (log.info if same_family else log.warning)("Ollama model %r not installed; using %r (set brain.ollama_model to pin one)", self.model, pick)
            self.model = pick
        self._checked = True

    def warm(self) -> None:
        """Load the model into memory so the first real answer is fast."""
        self._ensure_model()
        self.http.request("POST", "/api/chat", {"model": self.model, "messages": [], "keep_alive": self.keep_alive}, timeout=self.timeout)

    def generate(self, req: LLMRequest) -> LLMReply:
        t0 = time.perf_counter()
        self._ensure_model()
        msgs = [{"role": "system", "content": req.system}] + [{"role": r, "content": t} for r, t in req.messages]
        for _ in range(2):
            body: Dict[str, Any] = {
                "model": self.model, "messages": msgs, "stream": False, "keep_alive": self.keep_alive,
                "options": {"num_predict": req.max_tokens, "temperature": req.temperature},
            }
            if self._think_ok:
                body["think"] = False          # no hidden reasoning: speed
            if req.json_mode:
                body["format"] = "json"
            status, _, data = self.http.request("POST", "/api/chat", body, timeout=req.timeout_s or self.timeout)
            if status == 400 and self._think_ok and "think" in _err_text(data).lower():
                self._think_ok = False         # model doesn't support the flag
                continue
            if status == 404:
                self._checked = False
                raise ProviderUnavailable(f"Ollama model {self.model!r} not found; run: ollama pull {self.model}")
            if status >= 400:
                raise ProviderError(f"Ollama error {status}: {_err_text(data)}")
            j = _json(data)
            text = (j.get("message", {}).get("content") or "").strip()
            if not text:
                raise ProviderError("Ollama returned no text")
            return LLMReply(text, self.name, self.model, int(j.get("prompt_eval_count", 0)), int(j.get("eval_count", 0)),
                            (time.perf_counter() - t0) * 1000)
        raise ProviderError("Ollama: request rejected")
