"""Natural-language front door for the dev tools, with the confirm step.

"go to my folder called orbit in desktop and push the code"
   -> finds the project, builds a READ-ONLY plan, answers "orbit: 12 changes, 4 commits, push to origin/main. Say yes to go"
   -> only a following "yes" (within `confirm_seconds`) runs it; "no" cancels; anything else keeps it pending
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..errors import OrbitError
from ..settings import DevSettings
from . import gitplan, runner
from .projects import ProjectError, find_project_in_text, list_projects, resolve_project

log = logging.getLogger(__name__)

_YES = re.compile(r"^(?:yes|y|yeah|yep|confirm|do it|go|go ahead|ok|okay|proceed)\b[\s,.:!-]*(?:(?:with )?(?:message|msg)?[\s:]*(.+))?$", re.I)
_NO = re.compile(r"^(?:no|n|nope|cancel|stop|abort|never ?mind)\b", re.I)
_PUSH = re.compile(r"\b(push|commit|ship|sync)\b", re.I)
_OPEN = re.compile(r"\b(open|launch|start)\b", re.I)
_STATUS = re.compile(r"\b(status|what changed|changes in|diff)\b", re.I)


@dataclass
class Pending:
    plan: gitplan.PushPlan
    expires: float


class DevSkill:
    def __init__(self, cfg: DevSettings, clock: Callable[[], float] = time.monotonic,
                 git: Callable[..., str] = runner.git, open_editor: Callable = runner.open_in_editor) -> None:
        self.cfg, self.clock, self.git, self.open_editor = cfg, clock, git, open_editor
        self.pending: Optional[Pending] = None
        self.last_detail = ""

    # ------------------------------------------------------------------
    def handle(self, text: str) -> Optional[str]:
        if not self.cfg.enabled:
            return None
        t = text.strip()
        if self.pending is not None:
            if self.clock() > self.pending.expires:
                self.pending = None
            else:
                r = self._pending_reply(t)
                if r is not None:
                    return r
        low = t.lower()
        wants_push, wants_open, wants_status = bool(_PUSH.search(low)), bool(_OPEN.search(low)), bool(_STATUS.search(low))
        if not (wants_push or wants_open or wants_status):
            return None
        name = find_project_in_text(low, self.cfg)
        if name is None:
            known = ", ".join(list(list_projects(self.cfg))[:5]) or "none found"
            # only claim the request if it clearly talks about code/projects; otherwise let the model have it
            if re.search(r"\b(folder|project|repo|code|vs ?code|git|commit)\b", low):
                return f"Which project? ({known})"
            return None
        try:
            project = resolve_project(name, self.cfg)
        except ProjectError as exc:
            return str(exc)
        try:
            if wants_open and not wants_push:
                self.open_editor(self.cfg.editor, project)
                return f"Opened {project.name} in VS Code"
            if wants_status and not wants_push:
                plan = gitplan.build_plan(project, self.cfg.max_commit_groups, git=self.git)
                self.last_detail = plan.detail()
                return plan.summary().replace(". Say yes to go", "")
            single = bool(re.search(r"\b(one|single|1) commit\b", low))
            hint = "fix" if re.search(r"\bfix(es|ed)?\b", low) else None
            plan = gitplan.build_plan(project, self.cfg.max_commit_groups, single, hint, git=self.git)
        except (runner.RunnerError, OrbitError) as exc:
            return str(exc)[:140]
        self.last_detail = plan.detail()
        if plan.blocked or plan.nothing_to_do:
            return plan.summary()
        self.pending = Pending(plan, self.clock() + self.cfg.confirm_seconds)
        return plan.summary()

    def _pending_reply(self, t: str) -> Optional[str]:
        assert self.pending is not None
        low = t.lower().strip()
        if _NO.match(low):
            self.pending = None
            return "Cancelled"
        if low in ("details", "detail", "show", "show details", "what will you commit"):
            return " | ".join(g.message for g in self.pending.plan.groups)[:220] or "Only pushing existing commits"
        m = _YES.match(low)
        if m:
            plan, self.pending = self.pending.plan, None
            override = None
            if m.group(1):
                override = re.sub(r"^(?:message|msg)[\s:]+", "", t[t.lower().index(m.group(1)):], flags=re.I).strip()
            try:
                self.last_detail = plan.detail()
                return gitplan.execute(plan, override, git=self.git)
            except (runner.RunnerError, OrbitError) as exc:
                return str(exc)[:140]
        return None
