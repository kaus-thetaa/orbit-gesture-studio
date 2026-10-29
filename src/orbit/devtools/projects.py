"""Find "my folder called orbit in desktop": configured projects first, then direct children of the allowed roots."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import OrbitError
from ..settings import DevSettings


class ProjectError(OrbitError):
    pass


def norm(name: str) -> str:
    n = re.sub(r"\b(my|the|folder|project|repo|repository|directory|called|named|in|on|desktop|projects)\b", " ", name.lower())
    return re.sub(r"[^a-z0-9]+", "", n)


def list_projects(cfg: DevSettings) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for root in cfg.roots:
        base = Path(root).expanduser()
        try:
            for child in sorted(base.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    found.setdefault(child.name, child.resolve())
        except OSError:
            continue
    for name, path in cfg.projects.items():
        p = Path(str(path)).expanduser()
        if p.is_dir():
            found[name] = p.resolve()
    return found


def resolve_project(name: str, cfg: DevSettings) -> Path:
    projects = list_projects(cfg)
    want = norm(name)
    if not want:
        raise ProjectError("Which project?")
    exact = [n for n in projects if norm(n) == want]
    if len(exact) == 1:
        return projects[exact[0]]
    if len(exact) > 1:
        raise ProjectError(f"Several match {name!r}: {', '.join(exact[:4])}")
    prefix = [n for n in projects if norm(n).startswith(want)]
    if len(prefix) == 1:
        return projects[prefix[0]]
    if len(prefix) > 1:
        raise ProjectError(f"Which one: {', '.join(prefix[:4])}?")
    raise ProjectError(f"No folder called {name!r} in {', '.join(cfg.roots)}")


def find_project_in_text(text: str, cfg: DevSettings) -> Optional[str]:
    """Pick the project name mentioned in a free-text request (longest match wins)."""
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    best = None
    for name in list_projects(cfg):
        words = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if words and f" {words} " in t and (best is None or len(words) > len(best)):
            best = words
    if best:
        return best
    for name in list_projects(cfg):                    # e.g. "orbit" for a folder named orbit-gesture-studio
        first = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()[:1]
        if first and len(first[0]) >= 4 and f" {first[0]} " in t:
            return name
    return None
