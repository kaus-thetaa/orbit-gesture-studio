"""THE ONLY module in ORBIT allowed to start another program.

Hard rules (enforced by tests/test_actions.py::test_source_has_no_shell_or_eval):
  * no shell: argv is always a list and shell=False
  * only `git` (read/commit/push subcommands from a fixed allowlist) and a VS Code launcher
  * destructive/unsafe flags are refused here no matter who asks: force pushes, --no-verify, deletes, -c overrides
  * git never prompts (GIT_TERMINAL_PROMPT=0) and every call has a timeout
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

from ..errors import OrbitError

log = logging.getLogger(__name__)


class RunnerError(OrbitError):
    pass


ALLOWED_GIT = frozenset({"rev-parse", "status", "add", "commit", "push", "fetch", "rev-list", "remote", "log", "diff", "branch", "config", "symbolic-ref"})
FORBIDDEN_FLAGS = frozenset({"-f", "--force", "--force-with-lease", "--force-if-includes", "--no-verify", "--delete", "-d", "-D",
                             "--mirror", "--prune", "--exec", "--upload-pack", "--receive-pack", "-c", "--amend", "--all", "--tags"})
ALLOWED_EDITORS = frozenset({"code", "codium", "code-insiders"})
AI_PROVIDER_HOSTS = frozenset({"claude.ai", "chatgpt.com", "gemini.google.com"})
CHROME_LAUNCHERS = ("google-chrome", "google-chrome-stable", "chrome")


def _exec(argv: Sequence[str], cwd: Optional[str], timeout: float, env: Optional[dict] = None) -> "subprocess.CompletedProcess[str]":
    """The single place subprocess is called. argv must be a list; there is no shell."""
    if isinstance(argv, str) or not all(isinstance(a, str) and "\0" not in a for a in argv):
        raise RunnerError("argv must be a list of strings")
    return subprocess.run(list(argv), cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, shell=False, check=False)


def git(repo: Path, *args: str, timeout: float = 30.0) -> str:
    if not args or args[0] not in ALLOWED_GIT:
        raise RunnerError(f"git subcommand {args[:1]} is not allowed")
    for a in args[1:]:
        if a in FORBIDDEN_FLAGS or a.startswith("--force"):
            raise RunnerError(f"git flag {a!r} is not allowed")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", "GIT_PAGER": "cat"}
    try:
        cp = _exec(["git", "-C", str(repo), *args], None, timeout, env)
    except FileNotFoundError as exc:
        raise RunnerError("git is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"git {args[0]} timed out after {timeout:.0f}s (waiting for credentials or network?)") from exc
    if cp.returncode != 0:
        msg = (cp.stderr or cp.stdout or "").strip().splitlines()
        raise RunnerError(f"git {args[0]} failed: {msg[-1] if msg else cp.returncode}")
    return cp.stdout


def open_in_editor(editor: str, path: Path) -> None:
    if editor not in ALLOWED_EDITORS:
        raise RunnerError(f"editor {editor!r} not allowed (use one of {sorted(ALLOWED_EDITORS)})")
    exe = shutil.which(editor)
    if not exe:
        raise RunnerError(f"'{editor}' not found on PATH (VS Code: Command Palette -> 'Shell Command: Install code in PATH')")
    subprocess.Popen([exe, str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,   # detached
                     start_new_session=True, shell=False)


def open_ai_chat(profile: str, url: str) -> None:
    if not re.fullmatch(r"(?:Default|Profile [0-9]{1,3})", profile):
        raise RunnerError("Chrome profile must be `Default` or a folder such as `Profile 1`.")
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not hostname or not any(
        hostname == domain or hostname.endswith("." + domain) for domain in AI_PROVIDER_HOSTS
    ) or parsed.username or parsed.password:
        raise RunnerError("Chat URL must be an HTTPS link on claude.ai, chatgpt.com, or gemini.google.com.")
    exe = next((path for name in CHROME_LAUNCHERS if (path := shutil.which(name))), None)
    if not exe:
        raise RunnerError("Chrome was not found on PATH.")
    try:
        subprocess.Popen(
            [exe, f"--profile-directory={profile}", url],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            shell=False,
        )
    except OSError as exc:
        raise RunnerError(f"Could not open chat in Chrome: {exc}") from exc
