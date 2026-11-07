"""The built-in, explicitly registered actions."""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

from ..errors import ActionError, ActionUnsupported
from .base import Action, ActionContext, ActionRegistry, ParamSpec
from .keyboard import parse_shortcut

log = logging.getLogger(__name__)


def _press(ctx: ActionContext, shortcut: str) -> None:
    combo = parse_shortcut(shortcut, ctx.denylist)  # re-validated at execution time
    try:
        ctx.keyboard.press_combo(combo)
    except Exception as exc:
        raise ActionError(f"could not send {'+'.join(combo)}: {exc}") from exc


class FocusAction(Action):
    name = "focus"
    description = "Focus mode: screen-edge vignette + status indicator while active."
    modes = ("hold", "toggle")
    stateful = True
    params = {"dim": ParamSpec("int", default=35, min=0, max=80, doc="Vignette darkness, percent")}

    def activate(self, ctx, params):
        ctx.focus.activate(dim=params["dim"])

    def deactivate(self, ctx, params):
        ctx.focus.deactivate()


class ShortcutAction(Action):
    name = "keyboard_shortcut"
    description = "Send an approved keyboard shortcut (e.g. ctrl+shift+t)."
    modes = ("trigger",)
    params = {"keys": ParamSpec("str", required=True, doc="e.g. ctrl+shift+t")}

    def validate(self, params, denylist=()):
        out = super().validate(params, denylist)
        parse_shortcut(out["keys"], denylist)
        return out

    def activate(self, ctx, params):
        _press(ctx, params["keys"])


class _PlatformShortcut(Action):
    def __init__(self, name: str, key: str, description: str, modes: Tuple[str, ...], stateful: bool = False) -> None:
        self.name, self.key, self.description, self.modes, self.stateful = name, key, description, modes, stateful
        self.params = {}

    def _send(self, ctx: ActionContext) -> None:
        shortcut = ctx.platform.shortcut_for(self.key)
        if not shortcut:
            raise ActionUnsupported(f"{self.name} is not supported on {ctx.platform.name}")
        _press(ctx, shortcut)


class SnapAction(_PlatformShortcut):
    def activate(self, ctx, params):
        self._send(ctx)


class ShowDesktopAction(_PlatformShortcut):
    """Show/restore the desktop. The OS shortcut is itself a toggle, so we track our own
    state to stay idempotent (activate twice = one press). If you also toggle manually,
    the tracked state can drift - documented limitation."""

    def __init__(self) -> None:
        super().__init__("show_desktop", "show_desktop",
                         "Show the desktop while held / toggled, then restore windows.",
                         ("hold", "toggle"), stateful=True)
        self._shown = False
        self._lock = threading.Lock()

    def activate(self, ctx, params):
        with self._lock:
            if self._shown:
                return
            self._send(ctx)
            self._shown = True

    def deactivate(self, ctx, params):
        with self._lock:
            if not self._shown:
                return
            self._send(ctx)
            self._shown = False


class LogMessageAction(Action):
    """Harmless action for testing mappings without touching the desktop."""

    name = "log_message"
    description = "Write a message to the log (safe way to test a gesture)."
    modes = ("hold", "toggle", "trigger")
    stateful = True
    params = {"message": ParamSpec("str", default="gesture fired")}

    def activate(self, ctx, params):
        log.info("log_message ON: %s", params["message"])

    def deactivate(self, ctx, params):
        log.info("log_message OFF: %s", params["message"])


def build_default_registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg.register(FocusAction())
    reg.register(ShortcutAction())
    reg.register(SnapAction("snap_window_left", "snap_left", "Snap the active window to the left half.", ("trigger",)))
    reg.register(SnapAction("snap_window_right", "snap_right", "Snap the active window to the right half.", ("trigger",)))
    reg.register(ShowDesktopAction())
    reg.register(SnapAction("window_overview", "window_overview", "Open the OS window overview / task view.", ("trigger",)))
    reg.register(LogMessageAction())
    return reg
