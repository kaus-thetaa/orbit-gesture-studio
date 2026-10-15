"""Keyboard shortcut parsing/validation and the (swappable) keyboard backend.

Shortcuts are plain strings like "ctrl+shift+t". They are parsed against a closed
vocabulary (modifiers, a-z, 0-9, F1-F12, a few navigation keys) - nothing is ever passed
to a shell or evaluated. A configurable denylist blocks destructive combinations such as
alt+f4.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional, Protocol, Tuple

from ..errors import ConfigError

log = logging.getLogger(__name__)

MODIFIERS = ("ctrl", "alt", "shift", "win")
_ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "option": "alt", "opt": "alt",
    "cmd": "win", "command": "win", "super": "win", "meta": "win", "windows": "win",
    "return": "enter", "escape": "esc", "pgup": "pageup", "pgdn": "pagedown",
}
NAMED_KEYS = {"tab", "enter", "space", "esc", "left", "right", "up", "down", "home", "end", "pageup", "pagedown"} | {
    f"f{i}" for i in range(1, 13)
}
_SINGLE = set("abcdefghijklmnopqrstuvwxyz0123456789")
MAX_TOKENS = 4


class ShortcutError(ConfigError):
    pass


def canonicalize(text: str) -> Tuple[str, ...]:
    """Parse and normalise a shortcut (no denylist check). Raises ShortcutError."""
    if not isinstance(text, str) or not text.strip():
        raise ShortcutError("Shortcut must be a non-empty string like 'ctrl+shift+t'")
    raw = [t.strip().lower() for t in text.split("+")]
    if any(not t for t in raw):
        raise ShortcutError(f"Malformed shortcut {text!r}")
    tokens = [_ALIASES.get(t, t) for t in raw]
    if len(tokens) > MAX_TOKENS:
        raise ShortcutError(f"Shortcut {text!r} has too many keys (max {MAX_TOKENS})")
    if len(set(tokens)) != len(tokens):
        raise ShortcutError(f"Shortcut {text!r} repeats a key")
    mods = [t for t in tokens if t in MODIFIERS]
    keys = [t for t in tokens if t not in MODIFIERS]
    if len(keys) != 1:
        raise ShortcutError(f"Shortcut {text!r} needs exactly one non-modifier key")
    key = keys[0]
    if key not in _SINGLE and key not in NAMED_KEYS:
        raise ShortcutError(f"Key {key!r} is not allowed. Allowed: letters, digits, F1-F12, {sorted(NAMED_KEYS - {f'f{i}' for i in range(1, 13)})}")
    if not mods and key in _SINGLE:
        raise ShortcutError("Plain letters/digits need a modifier (ctrl, alt, shift, win)")
    ordered = [m for m in MODIFIERS if m in mods]
    return tuple(ordered + [key])


def parse_shortcut(text: str, denylist: Iterable[str] = ()) -> Tuple[str, ...]:
    combo = canonicalize(text)
    denied = set()
    for entry in denylist:
        try:
            denied.add(canonicalize(entry))
        except ShortcutError:
            continue
    if combo in denied:
        raise ShortcutError(f"Shortcut {'+'.join(combo)!r} is on the safety denylist")
    return combo


class KeyboardBackend(Protocol):
    def press_combo(self, combo: Tuple[str, ...]) -> None: ...


class PynputKeyboard:
    """Sends real key events via pynput. Imported lazily so tests/headless use never need it."""

    def __init__(self) -> None:
        self._ctl = None
        self._Key = None

    def _ensure(self):
        if self._ctl is None:
            try:
                from pynput.keyboard import Controller, Key
            except Exception as exc:  # ImportError, or no display server on Linux
                raise RuntimeError(f"keyboard control unavailable: {exc}") from exc
            self._ctl, self._Key = Controller(), Key
        return self._ctl, self._Key

    def _resolve(self, token: str):
        ctl, Key = self._ensure()
        mods = {"ctrl": Key.ctrl, "alt": Key.alt, "shift": Key.shift, "win": Key.cmd}
        if token in mods:
            return mods[token]
        if token in _SINGLE:
            return token
        special = {"pageup": "page_up", "pagedown": "page_down"}
        return getattr(Key, special.get(token, token))

    def press_combo(self, combo: Tuple[str, ...]) -> None:
        ctl, _ = self._ensure()
        *mods, key = combo
        pressed = []
        try:
            for m in mods:
                k = self._resolve(m)
                ctl.press(k)
                pressed.append(k)
                time.sleep(0.01)
            k = self._resolve(key)
            ctl.press(k)
            time.sleep(0.01)
            ctl.release(k)
        finally:  # never leave a modifier stuck down
            for k in reversed(pressed):
                try:
                    ctl.release(k)
                except Exception:  # pragma: no cover
                    log.exception("failed to release %s", k)
