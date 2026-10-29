"""Plan -> confirm -> execute for "commit and push my project".

The plan is read-only. Nothing is staged, committed or pushed until execute() is called, and the
caller (the assistant) only calls it after an explicit "yes". Refusals ("blocked") are final.

Guards (each one exists because of a real failure mode):
  * the folder must BE the git repository root - an ORBIT folder nested inside another repo would
    otherwise push that other repo's history
  * detached HEAD, no remote, or remote ahead of you -> blocked (never force, never merge for you)
  * secret-looking files/strings (keys, tokens, .env, private keys) -> blocked, listing paths only
  * the plan is fingerprinted; if anything changes before "yes", execute() refuses
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..errors import OrbitError
from . import runner

log = logging.getLogger(__name__)

_NAME_DENY = re.compile(r"(^|/)(\.env(\..*)?|id_rsa.*|id_ed25519.*|.*\.pem|.*\.key|.*\.p12|.*\.pfx|secrets\.json|credentials.*\.json|.*\.kdbx)$", re.I)
_CONTENT_DENY = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[=:]\s*['\"][^'\"\s]{16,}['\"]"),
]
MAX_SCAN_BYTES = 1_000_000


@dataclass
class Change:
    path: str
    code: str            # M modified, A added/untracked, D deleted, R renamed


@dataclass
class CommitGroup:
    message: str
    paths: List[str]


@dataclass
class PushPlan:
    project: Path
    branch: str = ""
    remote: str = ""
    remote_url: str = ""
    set_upstream: bool = False
    ahead: int = 0
    behind: int = 0
    changes: List[Change] = field(default_factory=list)
    groups: List[CommitGroup] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def nothing_to_do(self) -> bool:
        return not self.blocked and not self.changes and self.ahead == 0

    def summary(self) -> str:
        if self.blocked:
            return "Blocked: " + self.blocked[0]
        if self.nothing_to_do:
            return f"{self.project.name}: nothing to push"
        n = len(self.groups)
        parts = [f"{self.project.name}: {len(self.changes)} change{'s' if len(self.changes) != 1 else ''}"]
        if n:
            parts.append(f"{n} commit{'s' if n != 1 else ''}")
        parts.append(f"push to {self.remote}/{self.branch}")
        return ", ".join(parts) + ". Say yes to go"

    def detail(self) -> str:
        lines = [f"{self.project}  ->  {self.remote_url or self.remote} ({self.branch})"]
        lines += [f"  commit: {g.message}   [{len(g.paths)} file(s)]" for g in self.groups]
        if self.ahead:
            lines.append(f"  also pushing {self.ahead} commit(s) you already made")
        lines += [f"  warning: {w}" for w in self.warnings]
        lines += [f"  BLOCKED: {b}" for b in self.blocked]
        return "\n".join(lines)


# ====================================================================== parsing / scanning
def parse_status(raw: str) -> List[Change]:
    """`git status --porcelain=v1 -z` -> changes. Renames carry the old path as an extra token."""
    out: List[Change] = []
    toks = raw.split("\0")
    i = 0
    while i < len(toks):
        t = toks[i]
        i += 1
        if len(t) < 4:
            continue
        xy, path = t[:2], t[3:]
        if "R" in xy or "C" in xy:
            i += 1                                   # skip the original path
        code = "D" if "D" in xy else "A" if ("?" in xy or "A" in xy or "R" in xy or "C" in xy) else "M"
        out.append(Change(path, code))
    return out


def scan_for_secrets(repo: Path, paths: List[str]) -> List[str]:
    """Return human-readable findings (paths + reason, never the secret itself)."""
    findings: List[str] = []
    for rel in paths:
        if _NAME_DENY.search(rel):
            findings.append(f"{rel} (sensitive file name)")
            continue
        fp = repo / rel
        try:
            if not fp.is_file() or fp.stat().st_size > MAX_SCAN_BYTES:
                continue
            data = fp.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        text = data.decode("utf-8", "replace")
        if any(rx.search(text) for rx in _CONTENT_DENY):
            findings.append(f"{rel} (looks like it contains a key/token)")
    return findings


# ====================================================================== grouping / messages
def classify(path: str) -> Tuple[str, str, str]:
    """-> (group key, conventional prefix, scope)"""
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    if p.startswith("tests/") or "/tests/" in p or name.startswith("test_"):
        return "tests", "test", ""
    if p.endswith((".md", ".rst", ".txt")) and not name.startswith("requirements") or p.startswith("docs/"):
        return "docs", "docs", ""
    if p.startswith(("config/", "scripts/", ".github/")) or name in (".gitignore", "pyproject.toml", "makefile", "dockerfile") or name.startswith("requirements"):
        return "build", "chore", ""
    parts = p.split("/")
    if parts[0] == "src" and len(parts) >= 4:
        return "/".join(parts[:3]), "feat", parts[2]
    if parts[0] == "src" and len(parts) == 3:
        return "src/core", "feat", "core"
    return parts[0] if len(parts) > 1 else "root", "chore", ""


def _names(paths: List[str]) -> str:
    stems = []
    for p in paths:
        s = p.rsplit("/", 1)[-1]
        stems.append(s.rsplit(".", 1)[0] if "." in s.strip(".") else s)
    shown = ", ".join(stems[:3])
    return shown + (f" and {len(stems) - 3} more" if len(stems) > 3 else "")


def message_for(prefix: str, scope: str, changes: List[Change], override_prefix: Optional[str] = None) -> str:
    codes = {c.code for c in changes}
    verb = "add" if codes == {"A"} else "remove" if codes == {"D"} else "update"
    prefix = override_prefix or prefix
    if prefix == "feat" and verb != "add":
        prefix = "chore" if verb == "remove" else "feat"
    head = f"{prefix}({scope})" if scope else prefix
    msg = f"{head}: {verb} {_names([c.path for c in changes])}"
    return msg[:72]


def group_changes(changes: List[Change], max_groups: int = 6, single: bool = False, prefix_hint: Optional[str] = None) -> List[CommitGroup]:
    if not changes:
        return []
    if single:
        scopes = {classify(c.path)[0] for c in changes}
        return [CommitGroup(message_for("chore", "", changes, prefix_hint) if len(scopes) > 1 else message_for(*classify(changes[0].path)[1:], changes, prefix_hint),
                            [c.path for c in changes])]
    buckets: Dict[str, List[Change]] = {}
    meta: Dict[str, Tuple[str, str]] = {}
    for c in changes:
        key, prefix, scope = classify(c.path)
        buckets.setdefault(key, []).append(c)
        meta[key] = (prefix, scope)
    order = ["build", "src/core"] + sorted(k for k in buckets if k.startswith("src/") and k != "src/core") + \
            sorted(k for k in buckets if k not in ("build", "src/core", "tests", "docs") and not k.startswith("src/")) + ["tests", "docs"]
    keys = [k for k in order if k in buckets]
    while len(keys) > max_groups:                    # merge the two smallest buckets into "misc"
        keys.sort(key=lambda k: len(buckets[k]))
        a, b = keys[0], keys[1]
        merged = buckets.pop(a) + buckets.pop(b)
        merged = buckets.pop("misc", []) + merged    # (no-op if misc was one of a/b: already popped)
        buckets["misc"], meta["misc"] = merged, ("chore", "")
        keys = [k for k in keys if k not in (a, b, "misc")] + ["misc"]
    return [CommitGroup(message_for(*meta[k], buckets[k], prefix_hint), [c.path for c in buckets[k]]) for k in keys]


def valid_message(msg: str) -> str:
    msg = " ".join(msg.split())
    if not (3 <= len(msg) <= 72) or not msg.isprintable():
        raise OrbitError("Commit message must be 3-72 printable characters on one line")
    if msg.startswith("-"):
        raise OrbitError("Commit message cannot start with '-'")
    return msg


# ====================================================================== plan / execute
def _fingerprint(head: str, status_raw: str) -> str:
    return hashlib.sha1((head + "\0" + status_raw).encode("utf-8", "replace")).hexdigest()


def build_plan(project: Path, max_groups: int = 6, single: bool = False, prefix_hint: Optional[str] = None,
               git: Callable[..., str] = runner.git) -> PushPlan:
    project = Path(project).resolve()
    plan = PushPlan(project)
    try:
        top = Path(git(project, "rev-parse", "--show-toplevel").strip()).resolve()
    except runner.RunnerError:
        plan.blocked.append(f"{project.name} is not a git repository (run git init there first)")
        return plan
    if top != project:
        plan.blocked.append(f"{project.name} sits inside the repository at {top}; pushing would include other projects. "
                            f"Give {project.name} its own repo or move it")
        return plan
    try:                                               # works for a brand-new repo with no commits too
        plan.branch = git(project, "symbolic-ref", "--short", "-q", "HEAD").strip()
    except runner.RunnerError:
        plan.branch = ""
    if not plan.branch:
        plan.blocked.append("detached HEAD: check out a branch first")
        return plan
    try:
        upstream = git(project, "rev-parse", "--abbrev-ref", "@{u}").strip()
        plan.remote = upstream.split("/", 1)[0]
    except runner.RunnerError:
        remotes = git(project, "remote").split()
        if not remotes:
            plan.blocked.append("no remote configured (git remote add origin <url>)")
            return plan
        plan.remote, plan.set_upstream = ("origin" if "origin" in remotes else remotes[0]), True
    try:
        plan.remote_url = git(project, "config", "--get", f"remote.{plan.remote}.url").strip()
    except runner.RunnerError:
        pass
    if not plan.set_upstream:
        try:
            git(project, "fetch", "--quiet", plan.remote, timeout=30)
            behind, ahead = git(project, "rev-list", "--left-right", "--count", "@{u}...HEAD").split()
            plan.behind, plan.ahead = int(behind), int(ahead)
        except (runner.RunnerError, ValueError):
            plan.warnings.append("could not check the remote for new commits")
    if plan.behind:
        plan.blocked.append(f"{plan.remote}/{plan.branch} has {plan.behind} commit(s) you don't have; pull first")
        return plan
    raw = git(project, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    plan.changes = parse_status(raw)
    plan.fingerprint = _fingerprint(git(project, "rev-parse", "HEAD").strip() if _has_commits(project, git) else "none", raw)
    findings = scan_for_secrets(project, [c.path for c in plan.changes if c.code != "D"])
    if findings:
        plan.blocked.append("possible secrets in: " + "; ".join(findings[:3]) + (f" (+{len(findings) - 3} more)" if len(findings) > 3 else ""))
        return plan
    if any((project / c.path).is_file() and (project / c.path).stat().st_size > 20_000_000 for c in plan.changes if c.code != "D"):
        plan.warnings.append("a file over 20 MB is included")
    plan.groups = group_changes(plan.changes, max_groups, single, prefix_hint)
    return plan


def _has_commits(project: Path, git) -> bool:
    try:
        git(project, "rev-parse", "--verify", "HEAD")
        return True
    except runner.RunnerError:
        return False


def execute(plan: PushPlan, message_override: Optional[str] = None, git: Callable[..., str] = runner.git) -> str:
    """Stage + commit each group, then push. Re-checks that nothing changed since the preview."""
    if plan.blocked:
        raise OrbitError(plan.blocked[0])
    project = plan.project
    raw = git(project, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    head = git(project, "rev-parse", "HEAD").strip() if _has_commits(project, git) else "none"
    if _fingerprint(head, raw) != plan.fingerprint:
        raise OrbitError("Files changed since the preview; ask again")
    groups = plan.groups
    if message_override is not None:
        if len(groups) > 1:
            groups = [CommitGroup(valid_message(message_override), [p for g in groups for p in g.paths])]
        elif groups:
            groups = [CommitGroup(valid_message(message_override), groups[0].paths)]
    made = 0
    for g in groups:
        msg = valid_message(g.message)
        git(project, "add", "-A", "--", *g.paths)
        git(project, "commit", "-m", msg)
        made += 1
    args = ["push"] + (["-u"] if plan.set_upstream else []) + [plan.remote, plan.branch]
    git(project, *args, timeout=90)
    return f"Pushed {made} commit{'s' if made != 1 else ''}" + (f" (+{plan.ahead} earlier)" if plan.ahead else "") + f" to {plan.remote}/{plan.branch}"
