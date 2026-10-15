"""Gesture -> action mappings: strict validation + JSON persistence.

Mapping file format (version 1):

{
  "version": 1,
  "mappings": [
    {"id": "focus-two-fingers", "gesture": "two_fingers", "action": "focus",
     "mode": "hold", "params": {"dim": 35}, "enabled": true, "cooldown_ms": 500}
  ]
}

Mappings can only reference registered actions and validated parameters. There is no
field that accepts a command line, a script or an expression.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import paths
from .actions.base import ActionRegistry
from .errors import ConfigError
from .fsutil import atomic_write_json, quarantine_corrupt

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
GESTURE_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
SCHEMA_VERSION = 1


@dataclass
class Mapping:
    id: str
    gesture: str
    action: str
    mode: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    cooldown_ms: int = 500

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "Mapping":
        if not isinstance(raw, dict):
            raise ConfigError("mapping must be an object")
        allowed = {"id", "gesture", "action", "mode", "params", "enabled", "cooldown_ms"}
        unknown = set(raw) - allowed
        if unknown:
            raise ConfigError(f"unknown mapping field(s): {sorted(unknown)}")
        for req in ("id", "gesture", "action", "mode"):
            if req not in raw:
                raise ConfigError(f"mapping is missing {req!r}")
        return cls(
            id=raw["id"], gesture=raw["gesture"], action=raw["action"], mode=raw["mode"],
            params=raw.get("params") or {}, enabled=raw.get("enabled", True),
            cooldown_ms=raw.get("cooldown_ms", 500),
        )


DEFAULT_MAPPINGS: List[dict] = [
    {"id": "focus-two-fingers", "gesture": "two_fingers", "action": "focus", "mode": "hold",
     "params": {"dim": 35}, "enabled": True, "cooldown_ms": 500},
    # Examples that ship DISABLED - enable them in Gesture Studio once you trust the pipeline.
    {"id": "desktop-open-palm", "gesture": "open_palm", "action": "show_desktop", "mode": "toggle",
     "params": {}, "enabled": False, "cooldown_ms": 1500},
    {"id": "log-fist", "gesture": "fist", "action": "log_message", "mode": "hold",
     "params": {"message": "fist held"}, "enabled": False, "cooldown_ms": 300},
]


class MappingStore:
    def __init__(self, registry: ActionRegistry, path: Optional[Path] = None,
                 denylist: Callable[[], Iterable[str]] = lambda: ()) -> None:
        self.registry = registry
        self.path = Path(path) if path else paths.mappings_path()
        self._denylist = denylist
        self._lock = threading.RLock()
        self._mappings: List[Mapping] = []
        self.load_errors: List[str] = []

    # ------------------------------------------------------------ validation
    def validate(self, m: Mapping, existing: Optional[List[Mapping]] = None, replacing: Optional[str] = None) -> Mapping:
        if not isinstance(m.id, str) or not ID_RE.match(m.id):
            raise ConfigError("id must be lowercase letters/digits/-/_ (max 48 chars)")
        if not isinstance(m.gesture, str) or not GESTURE_RE.match(m.gesture):
            raise ConfigError("gesture must be a name like 'two_fingers'")
        if not isinstance(m.enabled, bool):
            raise ConfigError("enabled must be true/false")
        if isinstance(m.cooldown_ms, bool) or not isinstance(m.cooldown_ms, int) or not (0 <= m.cooldown_ms <= 10_000):
            raise ConfigError("cooldown_ms must be a whole number between 0 and 10000")
        m.params = self.registry.validate(m.action, m.params, m.mode, self._denylist())
        pool = self._mappings if existing is None else existing
        for other in pool:
            if other.id == replacing:
                continue
            if other.id == m.id:
                raise ConfigError(f"A mapping with id {m.id!r} already exists")
            if m.enabled and other.enabled and other.gesture == m.gesture:
                raise ConfigError(f"Gesture {m.gesture!r} is already used by enabled mapping {other.id!r}")
        return m

    # ------------------------------------------------------------ persistence
    def load(self) -> List[Mapping]:
        with self._lock:
            self._mappings, self.load_errors = [], []
            raw_list: List[Any]
            if not self.path.exists():
                raw_list = DEFAULT_MAPPINGS
                created = True
            else:
                created = False
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict) or not isinstance(data.get("mappings"), list):
                        raise ValueError("expected {'version': 1, 'mappings': [...]}")
                    if data.get("version", SCHEMA_VERSION) != SCHEMA_VERSION:
                        raise ValueError(f"unsupported mapping file version {data.get('version')!r}")
                    raw_list = data["mappings"]
                except (OSError, ValueError) as exc:
                    log.error("mappings file unreadable: %s", exc)
                    self.load_errors.append(f"mappings file unreadable ({exc}); defaults restored")
                    try:
                        quarantine_corrupt(self.path)
                    except OSError:
                        pass
                    raw_list, created = DEFAULT_MAPPINGS, True
            for raw in raw_list:
                try:
                    m = self.validate(Mapping.from_dict(raw), existing=self._mappings)
                except ConfigError as exc:
                    msg = f"skipped invalid mapping {raw.get('id') if isinstance(raw, dict) else raw!r}: {exc}"
                    log.warning(msg)
                    self.load_errors.append(msg)
                    continue
                self._mappings.append(m)
            if created:
                try:
                    self.save()
                except OSError as exc:
                    log.warning("could not write default mappings: %s", exc)
            return list(self._mappings)

    def save(self) -> None:
        with self._lock:
            atomic_write_json(self.path, {"version": SCHEMA_VERSION, "mappings": [m.to_dict() for m in self._mappings]})

    # ------------------------------------------------------------ queries
    def all(self) -> List[Mapping]:
        with self._lock:
            return [Mapping(**m.to_dict()) for m in self._mappings]

    def get(self, mapping_id: str) -> Optional[Mapping]:
        with self._lock:
            return next((Mapping(**m.to_dict()) for m in self._mappings if m.id == mapping_id), None)

    def for_gesture(self, gesture: str) -> Optional[Mapping]:
        with self._lock:
            return next((Mapping(**m.to_dict()) for m in self._mappings if m.enabled and m.gesture == gesture), None)

    # ------------------------------------------------------------ edits
    def add(self, m: Mapping) -> Mapping:
        with self._lock:
            self.validate(m)
            self._mappings.append(m)
            self.save()
            return m

    def update(self, mapping_id: str, new: Mapping) -> Mapping:
        with self._lock:
            idx = next((i for i, m in enumerate(self._mappings) if m.id == mapping_id), None)
            if idx is None:
                raise ConfigError(f"No mapping with id {mapping_id!r}")
            self.validate(new, replacing=mapping_id)
            self._mappings[idx] = new
            self.save()
            return new

    def remove(self, mapping_id: str) -> bool:
        with self._lock:
            before = len(self._mappings)
            self._mappings = [m for m in self._mappings if m.id != mapping_id]
            if len(self._mappings) != before:
                self.save()
                return True
            return False

    def set_enabled(self, mapping_id: str, enabled: bool) -> Mapping:
        with self._lock:
            cur = next((m for m in self._mappings if m.id == mapping_id), None)
            if cur is None:
                raise ConfigError(f"No mapping with id {mapping_id!r}")
            new = Mapping(**{**cur.to_dict(), "enabled": enabled})
            return self.update(mapping_id, new)


def make_id(gesture: str, action: str, taken: Iterable[str]) -> str:
    base = f"{action.replace('_', '-')}-{gesture.replace('_', '-')}"[:44]
    taken = set(taken)
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"
