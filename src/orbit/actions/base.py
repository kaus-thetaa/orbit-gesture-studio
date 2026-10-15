"""Action framework: explicit registration, declared parameters, strict validation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from ..errors import ConfigError

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
MODES = ("hold", "toggle", "trigger")
_SAFE_TEXT = re.compile(r"^[\w .,:;!?()\-+/'#@]{0,80}$")


@dataclass(frozen=True)
class ParamSpec:
    type: str  # "int" | "float" | "bool" | "str"
    default: Any = None
    required: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    doc: str = ""


@dataclass
class ActionContext:
    keyboard: Any            # KeyboardBackend
    focus: Any               # FocusController
    platform: Any            # PlatformAdapter
    denylist: Tuple[str, ...] = ()


class Action:
    name: str = ""
    description: str = ""
    modes: Tuple[str, ...] = ("trigger",)
    stateful: bool = False   # has a meaningful deactivate()
    params: Dict[str, ParamSpec] = {}

    # ---- validation (runs when a mapping is created AND before every execution)
    def validate(self, params: Any, denylist: Iterable[str] = ()) -> Dict[str, Any]:
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ConfigError(f"{self.name}: params must be an object")
        unknown = set(params) - set(self.params)
        if unknown:
            raise ConfigError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(self.params)}")
        out: Dict[str, Any] = {}
        for key, spec in self.params.items():
            if key not in params:
                if spec.required:
                    raise ConfigError(f"{self.name}: missing required parameter {key!r}")
                out[key] = spec.default
                continue
            out[key] = self._coerce(key, spec, params[key])
        return out

    def _coerce(self, key: str, spec: ParamSpec, value: Any) -> Any:
        where = f"{self.name}.{key}"
        if spec.type == "bool":
            if not isinstance(value, bool):
                raise ConfigError(f"{where}: expected true/false")
            return value
        if spec.type in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{where}: expected a number")
            if spec.type == "int":
                if isinstance(value, float) and not value.is_integer():
                    raise ConfigError(f"{where}: expected a whole number")
                value = int(value)
            else:
                value = float(value)
            if spec.min is not None and value < spec.min or spec.max is not None and value > spec.max:
                raise ConfigError(f"{where}: must be between {spec.min} and {spec.max}")
            return value
        if spec.type == "str":
            if not isinstance(value, str) or not _SAFE_TEXT.match(value):
                raise ConfigError(f"{where}: expected short plain text (letters, digits, basic punctuation)")
            return value
        raise ConfigError(f"{where}: unsupported parameter type {spec.type!r}")  # pragma: no cover

    # ---- execution (called from the dispatcher thread only)
    def activate(self, ctx: ActionContext, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def deactivate(self, ctx: ActionContext, params: Dict[str, Any]) -> None:
        return None


class ActionRegistry:
    """The allowlist. If an action is not registered here it cannot run - period."""

    def __init__(self) -> None:
        self._actions: Dict[str, Action] = {}

    def register(self, action: Action) -> None:
        if not NAME_RE.match(action.name):
            raise ValueError(f"bad action name {action.name!r}")
        if action.name in self._actions:
            raise ValueError(f"action {action.name!r} already registered")
        for m in action.modes:
            if m not in MODES:
                raise ValueError(f"{action.name}: unknown mode {m!r}")
        self._actions[action.name] = action

    def get(self, name: str) -> Action:
        try:
            return self._actions[name]
        except KeyError:
            raise ConfigError(f"Action {name!r} is not registered (allowlist: {sorted(self._actions)})") from None

    def names(self) -> list:
        return sorted(self._actions)

    def validate(self, name: str, params: Any, mode: str, denylist: Iterable[str] = ()) -> Dict[str, Any]:
        action = self.get(name)
        if mode not in action.modes:
            raise ConfigError(f"Action {name!r} supports modes {list(action.modes)}, not {mode!r}")
        return action.validate(params, denylist)
