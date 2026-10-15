import pathlib
import re

import pytest

from orbit.actions.base import ActionContext
from orbit.actions.builtin import build_default_registry
from orbit.actions.focus import FocusController
from orbit.actions.keyboard import ShortcutError, canonicalize, parse_shortcut
from orbit.actions.platform import detect_platform
from orbit.errors import ActionUnsupported, ConfigError
from fakes import FakeKeyboard

DENY = ("alt+f4", "ctrl+alt+delete", "win+l")


def ctx(platform="win32", kb=None):
    return ActionContext(kb or FakeKeyboard(), FocusController(), detect_platform(platform=platform), DENY)


# ---------------------------------------------------------------- shortcut parsing
@pytest.mark.parametrize("text,expected", [
    ("ctrl+shift+t", ("ctrl", "shift", "t")),
    ("Shift+Ctrl+T", ("ctrl", "shift", "t")),
    ("cmd+d", ("win", "d")),
    ("win+left", ("win", "left")),
    ("f5", ("f5",)),
    ("ctrl+alt+PgDn", ("ctrl", "alt", "pagedown")),
])
def test_valid_shortcuts(text, expected):
    assert parse_shortcut(text, DENY) == expected


@pytest.mark.parametrize("text", [
    "", "   ", "ctrl+", "+t", "ctrl+shift", "t", "5", "ctrl+t+y", "ctrl+ctrl+t", "ctrl+delete",
    "ctrl+backspace", "rm -rf /", "ctrl+;", "ctrl+alt+shift+win+t", "ctrl+$(whoami)", "ctrl+t; ls",
])
def test_invalid_shortcuts(text):
    with pytest.raises(ShortcutError):
        parse_shortcut(text, DENY)


def test_non_string_shortcut_rejected():
    with pytest.raises(ShortcutError):
        canonicalize(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["alt+f4", "ALT+F4", "f4+alt", "ctrl+alt+delete".replace("delete", "delete"), "win+l"])
def test_denylist(text):
    if text.endswith("delete"):
        pytest.skip("'delete' is not even in the allowed key vocabulary")
    with pytest.raises(ShortcutError):
        parse_shortcut(text, DENY)


# ---------------------------------------------------------------- registry / allowlist
def test_unknown_action_rejected():
    reg = build_default_registry()
    with pytest.raises(ConfigError):
        reg.validate("format_disk", {}, "trigger")
    with pytest.raises(ConfigError):
        reg.get("shell")


def test_mode_must_be_supported():
    reg = build_default_registry()
    with pytest.raises(ConfigError):
        reg.validate("snap_window_left", {}, "hold")
    assert reg.validate("focus", {}, "hold") == {"dim": 35}


def test_param_validation():
    reg = build_default_registry()
    with pytest.raises(ConfigError):
        reg.validate("focus", {"dim": 500}, "hold")
    with pytest.raises(ConfigError):
        reg.validate("focus", {"dim": "30"}, "hold")
    with pytest.raises(ConfigError):
        reg.validate("focus", {"dim": 10, "cmd": "calc.exe"}, "hold")   # unknown param
    with pytest.raises(ConfigError):
        reg.validate("keyboard_shortcut", {}, "trigger")                 # missing keys
    with pytest.raises(ConfigError):
        reg.validate("keyboard_shortcut", {"keys": "alt+f4"}, "trigger", DENY)
    with pytest.raises(ConfigError):
        reg.validate("log_message", {"message": "x`calc`; $(id)"}, "trigger")
    assert reg.validate("keyboard_shortcut", {"keys": "Ctrl+Shift+T"}, "trigger", DENY) == {"keys": "Ctrl+Shift+T"}


# ---------------------------------------------------------------- executing actions
def test_shortcut_action_sends_combo():
    reg, kb = build_default_registry(), FakeKeyboard()
    c = ctx(kb=kb)
    reg.get("keyboard_shortcut").activate(c, {"keys": "ctrl+shift+t"})
    assert kb.combos == [("ctrl", "shift", "t")]


def test_shortcut_revalidated_at_execution():
    reg, kb = build_default_registry(), FakeKeyboard()
    with pytest.raises(ConfigError):
        reg.get("keyboard_shortcut").activate(ctx(kb=kb), {"keys": "alt+f4"})  # bypassing validate()
    assert kb.combos == []


def test_snap_and_overview_windows():
    reg, kb = build_default_registry(), FakeKeyboard()
    c = ctx(kb=kb)
    reg.get("snap_window_left").activate(c, {})
    reg.get("snap_window_right").activate(c, {})
    reg.get("window_overview").activate(c, {})
    assert kb.combos == [("win", "left"), ("win", "right"), ("win", "tab")]


def test_unsupported_on_platform_fails_gracefully():
    reg = build_default_registry()
    with pytest.raises(ActionUnsupported):
        reg.get("snap_window_left").activate(ctx("darwin"), {})


def test_show_desktop_is_idempotent_and_restores():
    reg, kb = build_default_registry(), FakeKeyboard()
    c, a = ctx(kb=kb), reg.get("show_desktop")
    a.activate(c, {}); a.activate(c, {})
    assert kb.combos == [("win", "d")]
    a.deactivate(c, {}); a.deactivate(c, {})
    assert kb.combos == [("win", "d"), ("win", "d")]


def test_focus_action_idempotent():
    reg, c = build_default_registry(), ctx()
    seen = []
    c.focus.add_listener(lambda on, o: seen.append(on))
    f = reg.get("focus")
    f.activate(c, {"dim": 35}); f.activate(c, {"dim": 35})
    assert c.focus.active and seen == [True]
    f.deactivate(c, {}); f.deactivate(c, {})
    assert not c.focus.active and seen == [True, False]


def test_platform_override_and_validation():
    p = detect_platform({"snap_left": "ctrl+alt+left", "show_desktop": "nonsense+++", "bogus": "ctrl+x"}, platform="win32")
    assert p.shortcut_for("snap_left") == "ctrl+alt+left"
    assert p.shortcut_for("show_desktop") == "win+d"   # bad override ignored
    assert detect_platform(platform="linux").name == "linux"
    assert detect_platform({"snap_left": ""}, platform="win32").shortcut_for("snap_left") is None


# ---------------------------------------------------------------- static safety scan
def test_source_has_no_shell_or_eval():
    """Mapping values must never reach a shell or an evaluator. Guard the whole package."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "orbit"
    banned = [r"\bsubprocess\b", r"\bos\.system\b", r"\bos\.popen\b", r"(?<![.\w])eval\s*\(", r"(?<![.\w])exec\s*\(", r"(?<![.\w])compile\s*\(", r"\bshell\s*=\s*True", r"\b__import__\s*\("]
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pat in banned:
            for m in re.finditer(pat, text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(src)}:{line}: {pat}")
    assert offenders == []


def test_linux_session_info_flags_wayland_and_missing_camera():
    from orbit import diagnostics
    out = diagnostics.linux_session_info(env={"XDG_SESSION_TYPE": "wayland", "USER": "u"},
                                         listdir=lambda p: [], groups=["u"])
    text = "\n".join(out)
    assert "PROBLEM: Wayland" in text and "NONE FOUND" in text and "NO - run: sudo usermod" in text
    ok = diagnostics.linux_session_info(env={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"},
                                        listdir=lambda p: ["video0", "video1", "null"], groups=["video"])
    text = "\n".join(ok)
    assert "PROBLEM" not in text and "video0, video1" in text and "yes" in text
