# Dev helpers and the tracker spreadsheet (v0.4)

## "Go to my folder called orbit and push the code"

```
you:   go to my folder called orbit in desktop and push the code with appropriate commit
ORBIT: ORBIT: 7 changes, 5 commits, push to origin/main. Say yes to go      <- read-only plan, nothing has happened
you:   details            -> the exact commit messages
you:   yes                -> Pushed 5 commits to origin/main
```

Also from the terminal: `python -m orbit dev projects | status orbit | push orbit [--single] [--yes] | open orbit`.
"open orbit in vscode" launches your editor (`dev.editor`: code / codium / code-insiders; must be on PATH).

### How it stays safe

* **Nothing runs until you say yes** (valid for `dev.confirm_seconds`, default 90). "no" cancels. Any other message leaves it pending.
* **The plan is read-only** and fingerprinted; if files change before "yes", it refuses and asks you to start again.
* **The folder must be its own git repository.** If it is nested inside another repo (this actually happened with ORBIT inside
  a Desktop-wide repo), it is blocked with an explanation, because pushing would include the other projects.
* **Never forces.** No `--force`, `--no-verify`, `--amend`, deletes or resets; `git` runs only a fixed list of subcommands, with
  `shell=False`, a timeout, and no password prompts. If the remote has commits you lack, it stops: pull first.
* **Secret scan before anything is staged.** `.env`, private keys, `secrets.json`, and files containing key/token patterns block
  the push (paths only are shown, never the secret).
* **No code leaves your machine.** Commit messages are generated locally from file names (conventional style, ≤72 chars), not by a model.
* `yes, message: feat: your words` overrides the generated message (validated: one line, 3-72 chars).
* Needs credentials already set up (SSH agent or credential helper). If git would ask for a password it times out instead of hanging.

Commits are grouped by area (build files, each `src/orbit/<package>`, tests, docs), at most `dev.max_commit_groups` (6); say
"as one commit" for a single commit, or "fix" in your request for `fix:` prefixes.

Where it looks for "my folder called X": direct children of `dev.roots` (default `~/Desktop`, `~/Projects`) plus `dev.projects`
(`{"orbit": "~/Projects/orbit-gesture-studio"}`).

## One spreadsheet for everything

A single workbook (default `~/Documents/orbit_tracker.xlsx`, setting `tracker.path`); each table is a **sheet** inside it, so there
is never more than one file. The `Interviews` sheet has: Company, Role, Applied On, Status, Interview Date, Round, Result, Notes,
Updated (real dates, a Status dropdown, frozen header, filter).

```
I applied to Google for SWE intern on Oct 3     -> Added Google - SWE intern, applied 3 Oct
Google interview on 5 Nov                        -> Google: Interview Date: 5 Nov, Interview Scheduled
Google rejected me / I got an offer from Zoho    -> Google: Rejected
add interview: Flipkart, Data Analyst, applied Oct 3, interview 5 Nov
when is my next interview?                       -> Google - SWE intern: 5 Nov (in 18 days)
what's the status of Google / did Amazon reply?  -> Google: Interview Scheduled, interview 5 Nov
how many applications / interviews / offers      list my interviews          undo
```

Common phrasings use deterministic rules (instant, nothing sent anywhere). Free-form statements fall back to the **local** model
(`tracker.local_only_extraction`: true), and its output is validated (dates, statuses) before anything is written.

* **Undo** reverses the last change (rolling backups, kept in ORBIT's folder, not next to your file).
* Safe saves: temp file + rename; refuses to write while LibreOffice has the file open; never overwrites a file it can't read.
* Your own edits in Excel/LibreOffice, extra columns and extra sheets are preserved (the file is reloaded each time).
* Cell text starting with `=`, `+`, `-` or `@` is neutralised, so pasted text can't become a formula.
* Dates: day-first (`5/11` = 5 Nov), names like "Oct 3", "next monday", "in 2 weeks".

Limits: one company with several roles needs the role named ("Google rejected me" asks which). More tables can be added in code
(`DEFAULT_TABLES`); a settings-driven schema is not built yet.
