# ORBIT assistant (v0.3)

One assistant, two interchangeable models, one shared memory, answers in one line.

```
question -> instant skills (menu, time)  ──────────────► answer in ~1 ms, no model, no network
         └► model path: shared context ► router ► Gemini (online)  or  Ollama (offline / budget low / rate-limited)
```

## Setup (5 minutes)

```bash
python -m orbit brain set-key      # paste your Gemini API key (stored in a 0600 file in the config dir, never in git)
ollama pull llama3.2:3b            # local fallback model (any small model works; see `ollama list`)
python -m orbit brain doctor       # shows network, key, available Gemini models, Ollama models, budget
python -m orbit ask "capital of France?"
python -m orbit hud                # the orb
python -m orbit autostart enable   # start the orb at login (Linux)
```

Model names change often. `brain.gemini_model` is configurable, and if it isn't available ORBIT lists your models
and auto-picks a flash one. Ollama likewise falls back to the smallest installed model. Gemini thinking is turned to
the minimum the model accepts (it adds latency); ORBIT probes which setting your model allows and remembers it.

## Which model answers?

| Situation | Model |
|---|---|
| Online, key set, budget left | Gemini |
| Offline (TCP probe to Google fails) | Ollama |
| No key / key rejected (10 min cooldown) | Ollama |
| Rate-limited (429) | Ollama until `Retry-After` passes |
| Used ≥ `switch_at_fraction` (90%) of `brain.gemini_daily_requests` | Ollama until tomorrow |
| Question needs a **private** fact, or `share_memory_with_cloud` is false | Ollama only; never sent to Gemini |
| Gemini slower than `hedge_after_ms` (1.5 s) | both are asked; first answer wins |

"Token low" is tracked locally from your own request counter (Gemini doesn't expose a live balance) plus 429s.
Both models receive the **same** context pack: facts, last few turns, today's menu, date/time.

## Speed

* Menu/time questions never touch a model (parsed once at fetch time; answered by lookup).
* Persistent HTTPS connection (no handshake per question), tiny prompts, ≤60 output tokens, no thinking.
* Ollama is pre-loaded at startup and kept in memory (`keep_alive`).
* Answers are cut to one line / `max_words` (14).
* Measured here: instant skills ≈2 ms (measured: 1.8 ms average including the SQLite write). Model latency depends on your network, model and CPU/GPU and has **not** been
  measured on your machine; `python -m orbit ask` prints it per answer.

## Memory ("learns you")

```
remember that my snack time is 4:30        -> stored, shared with Gemini when relevant
remember privately: locker is in block C   -> local-only (Ollama sees it, Gemini never does)
what do you know about me / forget snack
```

Same via `python -m orbit memory add|list|forget` (`--private`). Facts that look like secrets (passwords, OTPs, card
numbers, API keys) are refused. Frequently useful facts rise to the top of future prompts. This is retrieval, not
model training: it's transparent, editable and deletable. Everything is in one SQLite file in the config dir.

## Mess menu (and other routines)

```bash
python -m orbit routine add-menu https://tikmit.com/     # recognised: uses the viewer's public JSON API
python -m orbit routine run mess_menu                    # try it now; prints today's meals with times
```

`tikmit.com` is a JavaScript viewer: its HTML contains no menu, which is why a plain page fetch finds nothing. The
site's data comes from a public API (`GET https://tikm.coolstuff.work/api/menu?date=YYYY-MM-DD`, documented in the
project's `openapi.json`), so ORBIT calls that instead: ONE tiny request per refresh, for today only. It runs when you
are online (at startup, whenever the network returns, every 6 h). If no menu is published for today it waits an hour
before asking again; other failures back off 30 s → 30 min.

Then, all answered locally in ~2 ms with no model:

```
what's for snacks      -> Snacks 4:30-5:30 PM: Samosa, Tea
when is dinner         -> Dinner 7:30-9:30 PM
what's the next meal   -> Next - Lunch 12:30-2:30 PM: Rice, Dal, Paneer      (or "Now - ...")
mess menu today        -> B: ... | L: ... | S: ... | D: ...
```

Anything fuzzier ("anything vegetarian for lunch?") goes to the model, which sees the menu with times in its context.

Other menu pages (`kind: "menu"`) are fetched as HTML/text, parsed by weekly-table rules, else by the model. PDF/image menus
and pages behind a login are not supported yet. The API adapter follows the published spec and could not be checked
against the live service while building; if it ever changes, `routine run` prints the raw response so it can be fixed.
This is a student-run public service: ORBIT identifies itself, keeps requests minimal and respects the retry delays.

## Measuring speed

```bash
python -m orbit brain bench -n 5                      # first request vs later ones, per model
python -m orbit brain bench -n 5 --model gemini-2.5-flash-lite   # compare another Gemini model
```

"first" includes connecting; "next" reuses the connection (what the orb does). Gemini thinking settings are probed once and
remembered (`gemini_state.json`), so later runs don't waste round trips.

## The orb

Tiny, translucent, bottom-right. Green = Gemini ready, amber = local only, grey = nothing available. Click to ask
(or `Ctrl+Alt+Space` on X11), Enter, one line appears and fades. Right-click: open Studio / disarm / quit. "Menu
updated" flashes when a routine runs. Always-on-top works under X11/XWayland; some Wayland compositors ignore it.

## Privacy summary

* Sent to Gemini: your question, recent turns, **shared** facts relevant to it, today's menu text. Nothing else.
* Never sent: private facts, the camera feed, anything when you are offline or sharing is off.
* API key: env var or a 0600 file; never in settings.json or the repo.
* Menu fetches go only to the URL you configured.
