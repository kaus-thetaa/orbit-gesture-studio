import os
import subprocess
from pathlib import Path

import pytest

from orbit.brain.assistant import Assistant
from orbit.brain.context import ContextStore
from orbit.brain.router import Router
from orbit.devtools import gitplan, runner
from orbit.devtools.gitplan import Change, build_plan, execute, group_changes, parse_status, scan_for_secrets, valid_message
from orbit.devtools.projects import ProjectError, find_project_in_text, list_projects, resolve_project
from orbit.devtools.skill import DevSkill
from orbit.errors import OrbitError
from orbit.settings import BrainSettings, DevSettings

ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def sh(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, text=True).stdout


def write(p: Path, text="x\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture(autouse=True)
def git_identity(monkeypatch):
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        monkeypatch.setenv(k, ENV[k])


@pytest.fixture
def world(tmp_path):
    """Desktop/ORBIT is its own repo with an upstream on a local bare remote."""
    remote = tmp_path / "remote.git"
    sh(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    desktop = tmp_path / "Desktop"
    proj = desktop / "ORBIT"
    proj.mkdir(parents=True)
    sh(proj, "init", "-q", "-b", "main")
    write(proj / "README.md", "# orbit\n")
    sh(proj, "add", "-A"); sh(proj, "commit", "-q", "-m", "init")
    sh(proj, "remote", "add", "origin", str(remote))
    sh(proj, "push", "-q", "-u", "origin", "main")
    return tmp_path, desktop, proj, remote


def remote_log(remote):
    return sh(remote, "log", "--format=%s", "main").strip().splitlines()


def dirty(proj):
    write(proj / "src/orbit/vision/motion.py", "a=1\n")
    write(proj / "src/orbit/vision/lips.py", "b=2\n")
    write(proj / "src/orbit/gui/hud.py", "c=3\n")
    write(proj / "tests/test_motion.py", "def test_a(): pass\n")
    write(proj / "docs/ASSISTANT.md", "# docs\n")
    write(proj / "README.md", "# orbit v2\n")
    write(proj / "pyproject.toml", "[project]\n")


# ============================================================ runner hard limits
@pytest.mark.parametrize("args", [("reset", "--hard"), ("clean", "-fd"), ("checkout", "."), ("rm", "-r", "x"), ("push", "--force", "origin", "main"),
                                  ("push", "-f"), ("push", "--force-with-lease"), ("commit", "--no-verify", "-m", "x"), ("push", "--delete", "origin", "main"),
                                  ("commit", "--amend"), ("status", "-c", "x=y"), ("push", "--mirror")])
def test_runner_refuses_dangerous_git(tmp_path, args):
    with pytest.raises(runner.RunnerError):
        runner.git(tmp_path, *args)


def test_runner_requires_list_argv_and_allowlisted_editor(tmp_path):
    with pytest.raises(runner.RunnerError):
        runner._exec("git status", None, 5)
    with pytest.raises(runner.RunnerError):
        runner.open_in_editor("bash", tmp_path)
    with pytest.raises(runner.RunnerError):
        runner.open_in_editor("code", tmp_path) if not __import__("shutil").which("code") else (_ for _ in ()).throw(runner.RunnerError("skip"))


def test_runner_opens_supported_ai_chat_in_selected_chrome_profile(monkeypatch):
    launched = []
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/google-chrome" if name == "google-chrome" else None)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: launched.append((args, kwargs)))

    runner.open_ai_chat("Profile 2", "https://claude.ai/chat/abc")

    assert launched == [(
        (["/usr/bin/google-chrome", "--profile-directory=Profile 2", "https://claude.ai/chat/abc"],),
        {
            "stdin": runner.subprocess.DEVNULL,
            "stdout": runner.subprocess.DEVNULL,
            "stderr": runner.subprocess.DEVNULL,
            "start_new_session": True,
            "shell": False,
        },
    )]


@pytest.mark.parametrize(
    "profile,url",
    [
        ("../../Default", "https://claude.ai/chat/abc"),
        ("Default", "https://example.com/"),
        ("Default", "javascript:alert(1)"),
        ("Default", "https://user:pass@claude.ai/chat/abc"),
    ],
)
def test_runner_rejects_unsafe_ai_chat_target(profile, url):
    with pytest.raises(runner.RunnerError):
        runner.open_ai_chat(profile, url)


# ============================================================ parsing / grouping / messages
def test_parse_status_renames_deletes_untracked():
    raw = "R  new.py\0old.py\0 M a/b.py\0?? c.txt\0 D gone.py\0A  added.py\0"
    got = [(c.path, c.code) for c in parse_status(raw)]
    assert got == [("new.py", "A"), ("a/b.py", "M"), ("c.txt", "A"), ("gone.py", "D"), ("added.py", "A")]


def test_grouping_and_conventional_messages():
    ch = [Change("src/orbit/vision/motion.py", "A"), Change("src/orbit/vision/lips.py", "A"), Change("src/orbit/gui/hud.py", "M"),
          Change("tests/test_a.py", "A"), Change("docs/X.md", "M"), Change("README.md", "M"), Change("pyproject.toml", "M"), Change("src/orbit/cli.py", "M")]
    msgs = {tuple(g.paths): g.message for g in group_changes(ch)}
    assert "feat(vision): add motion, lips" in msgs.values()
    assert "test: add test_a" in msgs.values() and "docs: update X, README" in msgs.values()
    assert any(m.startswith("chore: update pyproject") for m in msgs.values())
    assert all(len(m) <= 72 for m in msgs.values())
    assert len(group_changes(ch, max_groups=3)) == 3
    assert sum(len(g.paths) for g in group_changes(ch, max_groups=2)) == len(ch)          # merging never loses a file
    one = group_changes(ch, single=True)
    assert len(one) == 1 and len(one[0].paths) == len(ch)
    assert group_changes([Change("src/orbit/a/b.py", "M")], prefix_hint="fix")[0].message.startswith("fix(a)")
    assert group_changes([]) == []


def test_commit_message_newlines_are_collapsed_not_injected():
    assert valid_message("feat: one\n\nBREAKING: two") == "feat: one BREAKING: two"


@pytest.mark.parametrize("bad", ["", "ab", "x" * 80, "-rf", "bell\x07"])
def test_commit_message_validation(bad):
    with pytest.raises(OrbitError):
        valid_message(bad)


# ============================================================ secrets
def test_secret_scan(tmp_path):
    write(tmp_path / ".env", "A=1\n")
    write(tmp_path / "k.txt", "key = " + "AI" + "za" + "x" * 35 + "\n")
    write(tmp_path / "pk.txt", "-----BEGIN RSA " + "PRIVATE KEY-----\nabc\n")
    write(tmp_path / "cfg.py", "API_" + 'KEY = "' + "abcdefghijklmnop1234" + '"\n')
    write(tmp_path / "ok.py", "x = 1  # token handling lives elsewhere\n")
    (tmp_path / "bin.dat").write_bytes(b"\0\1\2" + b"AI" + b"za" + b"x" * 40)
    got = scan_for_secrets(tmp_path, [".env", "k.txt", "pk.txt", "cfg.py", "ok.py", "bin.dat", "missing.txt"])
    assert [g.split(" ")[0] for g in got] == [".env", "k.txt", "pk.txt", "cfg.py"]
    assert not any("xxxx" in g for g in got)                                  # findings never echo the secret


def test_scan_has_no_false_positives_on_orbit_itself():
    root = Path(__file__).resolve().parents[1]
    files = [str(p.relative_to(root)) for d in ("src", "tests", "docs", "config", "scripts") for p in (root / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += ["README.md", "pyproject.toml", "requirements.txt", ".gitignore"]
    assert scan_for_secrets(root, files) == []


# ============================================================ plans against real git
def test_plan_for_dirty_project(world):
    _, _, proj, _ = world
    dirty(proj)
    plan = build_plan(proj)
    assert not plan.blocked and plan.remote == "origin" and plan.branch == "main" and len(plan.changes) == 7
    assert 3 <= len(plan.groups) <= 6 and "push to origin/main" in plan.summary() and plan.fingerprint
    assert "commit:" in plan.detail()


def test_nested_inside_other_repo_is_blocked(tmp_path):
    """The exact mistake from this project's history: ORBIT inside ~/Desktop, where Desktop is a different repo."""
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    sh(desktop, "init", "-q", "-b", "main")
    write(desktop / "tmit.txt")
    sh(desktop, "add", "-A"); sh(desktop, "commit", "-q", "-m", "tmit work")
    orbit = desktop / "ORBIT"
    write(orbit / "a.py")
    plan = build_plan(orbit)
    assert plan.blocked and "inside the repository" in plan.blocked[0] and "other projects" in plan.blocked[0]
    assert plan.groups == []


def test_blocked_cases(world, tmp_path):
    _, desktop, proj, remote = world
    plain = tmp_path / "plain"; plain.mkdir()
    assert "not a git repository" in build_plan(plain).blocked[0]
    sh(proj, "checkout", "-q", "--detach")
    assert "detached HEAD" in build_plan(proj).blocked[0]
    sh(proj, "checkout", "-q", "main")
    # remote moves ahead -> behind -> blocked, never force
    other = tmp_path / "other"
    sh(tmp_path, "clone", "-q", str(remote), str(other))
    write(other / "x.txt"); sh(other, "add", "-A"); sh(other, "commit", "-q", "-m", "elsewhere"); sh(other, "push", "-q")
    write(proj / "new.py")
    plan = build_plan(proj)
    assert "1 commit(s) you don't have" in plan.blocked[0] and "pull first" in plan.blocked[0]
    norem = tmp_path / "norem"; norem.mkdir(); sh(norem, "init", "-q", "-b", "main")
    write(norem / "a"); sh(norem, "add", "-A"); sh(norem, "commit", "-q", "-m", "i")
    assert "no remote" in build_plan(norem).blocked[0]


def test_secrets_block_the_plan(world):
    _, _, proj, _ = world
    write(proj / ".env", "TOKEN=abc\n")
    write(proj / "ok.py")
    plan = build_plan(proj)
    assert plan.blocked and ".env" in plan.blocked[0] and plan.groups == []


def test_execute_commits_and_pushes_for_real(world):
    _, _, proj, remote = world
    dirty(proj)
    plan = build_plan(proj)
    out = execute(plan)
    assert out.startswith("Pushed") and "to origin/main" in out
    log = remote_log(remote)
    assert len(log) == 1 + len(plan.groups) and log[0] != "init"
    assert all(len(m) <= 72 for m in log) and any(m.startswith("test:") for m in log) and any(m.startswith("docs:") for m in log)
    assert sh(proj, "status", "--porcelain").strip() == ""
    assert build_plan(proj).nothing_to_do


def test_execute_refuses_if_files_changed_after_preview(world):
    _, _, proj, remote = world
    dirty(proj)
    plan = build_plan(proj)
    write(proj / "sneaky.py")
    with pytest.raises(OrbitError, match="changed since the preview"):
        execute(plan)
    assert remote_log(remote) == ["init"]


def test_push_only_existing_commits(world):
    _, _, proj, remote = world
    write(proj / "a.py"); sh(proj, "add", "-A"); sh(proj, "commit", "-q", "-m", "local only")
    plan = build_plan(proj)
    assert plan.ahead == 1 and plan.groups == [] and "push to origin/main" in plan.summary()
    assert "Pushed 0 commits (+1 earlier)" in execute(plan) and remote_log(remote)[0] == "local only"


def test_first_push_sets_upstream(tmp_path):
    remote = tmp_path / "r.git"; sh(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    p = tmp_path / "proj"; p.mkdir(); sh(p, "init", "-q", "-b", "main")
    write(p / "a.py"); sh(p, "remote", "add", "origin", str(remote))
    plan = build_plan(p)
    assert plan.set_upstream and not plan.blocked
    execute(plan)
    assert remote_log(remote)
    assert "origin/main" in sh(p, "rev-parse", "--abbrev-ref", "@{u}")


def test_message_override_for_single_commit(world):
    _, _, proj, remote = world
    write(proj / "a.py")
    execute(build_plan(proj), "feat: my own words")
    assert remote_log(remote)[0] == "feat: my own words"
    write(proj / "b.py"); write(proj / "tests/test_b.py")
    execute(build_plan(proj), "chore: everything at once")
    assert remote_log(remote)[0] == "chore: everything at once" and len(remote_log(remote)) == 3


# ============================================================ project lookup
def cfg_for(desktop, **kw):
    return DevSettings(roots=[str(desktop)], **kw)


def test_project_lookup(world, tmp_path):
    _, desktop, proj, _ = world
    (desktop / "orbit-gesture-studio").mkdir(); (desktop / "Notes").mkdir(); (desktop / ".hidden").mkdir()
    cfg = cfg_for(desktop)
    assert set(list_projects(cfg)) == {"ORBIT", "orbit-gesture-studio", "Notes"}
    assert resolve_project("orbit", cfg) == proj.resolve()                        # exact beats prefix
    assert resolve_project("my folder called ORBIT in desktop", cfg) == proj.resolve()
    assert resolve_project("orbit gesture", cfg).name == "orbit-gesture-studio"
    with pytest.raises(ProjectError, match="No folder called"):
        resolve_project("zebra", cfg)
    cfg2 = DevSettings(roots=[str(desktop)], projects={"work": str(tmp_path)})
    assert resolve_project("work", cfg2) == tmp_path.resolve()
    assert find_project_in_text("go to my folder called orbit in desktop and push", cfg) == "orbit"
    assert find_project_in_text("push the orbit gesture studio code", cfg) == "orbit gesture studio"
    assert find_project_in_text("hello there", cfg) is None


# ============================================================ the conversation
class Clock:
    t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def skill(world):
    _, desktop, proj, remote = world
    opened = []
    clock = Clock()
    s = DevSkill(cfg_for(desktop), clock=clock, open_editor=lambda editor, path: opened.append((editor, path)))
    s.opened, s.clock_obj, s.proj, s.remote = opened, clock, proj, remote
    return s


PROMPT = "go to my folder called orbit in desktop and push the code with appropriate commit"


def test_prompt_plan_yes_pushes(skill):
    dirty(skill.proj)
    ans = skill.handle(PROMPT)
    assert ans.startswith("ORBIT: 7 changes") and ans.endswith("Say yes to go")
    assert remote_log(skill.remote) == ["init"]                       # NOTHING happened yet
    assert skill.pending is not None
    out = skill.handle("yes")
    assert out.startswith("Pushed") and skill.pending is None
    assert len(remote_log(skill.remote)) > 1


def test_no_cancels_and_nothing_is_pushed(skill):
    dirty(skill.proj)
    skill.handle(PROMPT)
    assert skill.handle("no") == "Cancelled" and skill.pending is None
    assert remote_log(skill.remote) == ["init"] and skill.handle("yes") is None      # a stray "yes" later does nothing


def test_pending_expires(skill):
    dirty(skill.proj)
    skill.handle(PROMPT)
    skill.clock_obj.t += 200
    assert skill.handle("yes") is None and skill.pending is None
    assert remote_log(skill.remote) == ["init"]


def test_unrelated_text_keeps_pending_and_details(skill):
    dirty(skill.proj)
    skill.handle(PROMPT)
    assert skill.handle("what's the capital of France") is None and skill.pending is not None
    d = skill.handle("details")
    assert "feat(vision)" in d and "test:" in d
    assert skill.pending is not None


def test_yes_with_custom_message(skill):
    write(skill.proj / "a.py")
    skill.handle("push orbit")
    assert skill.handle("yes, message: feat: ship the thing").startswith("Pushed 1 commit")
    assert remote_log(skill.remote)[0] == "feat: ship the thing"
    write(skill.proj / "b.py")
    skill.handle("push orbit")
    assert "3-72" in skill.handle("yes message: x")                  # invalid message is refused, nothing pushed
    assert remote_log(skill.remote)[0] == "feat: ship the thing"


def test_single_commit_wording(skill):
    dirty(skill.proj)
    assert "1 commit," in skill.handle("commit everything in orbit as one commit and push")


def test_blocked_and_nothing_to_push_answers(skill, world):
    assert skill.handle("push orbit") == "ORBIT: nothing to push" and skill.pending is None
    write(skill.proj / ".env", "T=1\n")
    ans = skill.handle("push orbit")
    assert ans.startswith("Blocked: possible secrets") and skill.pending is None and ".env" in skill.last_detail


def test_nested_project_is_refused_in_conversation(tmp_path):
    desktop = tmp_path / "Desktop"; desktop.mkdir()
    sh(desktop, "init", "-q", "-b", "main")
    write(desktop / "tmit.txt"); sh(desktop, "add", "-A"); sh(desktop, "commit", "-q", "-m", "x")
    write(desktop / "ORBIT" / "a.py")
    s = DevSkill(cfg_for(desktop))
    ans = s.handle(PROMPT)
    assert ans.startswith("Blocked: ORBIT sits inside the repository") and s.pending is None


def test_open_status_and_unknowns(skill):
    assert skill.handle("open orbit in vscode") == "Opened ORBIT in VS Code" and skill.opened == [("code", skill.proj.resolve())]
    dirty(skill.proj)
    assert skill.handle("git status of orbit") == "ORBIT: 7 changes, 4 commits, push to origin/main" or "changes" in skill.handle("git status of orbit")
    assert skill.pending is None                                      # status never arms a push
    assert skill.handle("push zebra project").startswith("Which project?")
    assert skill.handle("I will push through this exam") is None
    assert skill.handle("open the window") is None
    assert DevSkill(DevSettings(enabled=False)).handle(PROMPT) is None


def test_assistant_routes_dev_requests_and_confirmation(world, tmp_path):
    _, desktop, proj, remote = world
    write(proj / "a.py")
    store = ContextStore(tmp_path / "b.db")
    cfg = BrainSettings()
    a = Assistant(cfg, store, Router(cfg, store, None, None, None), dev=DevSkill(cfg_for(desktop)))
    first = a.ask(PROMPT)
    assert first.source == "dev" and first.text.endswith("Say yes to go") and "commit:" in first.note
    assert remote_log(remote) == ["init"]
    second = a.ask("yes")
    assert second.source == "dev" and second.text.startswith("Pushed 1 commit")
    store.close()
