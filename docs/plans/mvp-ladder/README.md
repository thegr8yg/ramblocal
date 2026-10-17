# ramblocal: MVP ladder

The roadmap in `docs/roadmap/README.md` is organised by subsystem (API, live, text, store, CUDA, UI). This plan
replaces its ordering. It is organised by **what you can use**: every rung is a working, demoable program, the
smallest step up from the one below. The subsystem plans are still the source for detail, but they get pulled
in one rung at a time, and nothing is built before a rung needs it.

## Rules for every rung

1. **Ships something usable.** A rung ends with a binary you can run and a one-line demo.
2. **Small.** Target 1-3 days of work, 1-3 PRs. If a rung is bigger, split it.
3. **Has its own gate.** Each rung has a test, an eval or a benchmark that must pass before the next rung starts.
   Gates are cumulative: earlier gates keep running in CI.
4. **No speculative work.** No scheduler before there are two concurrent callers, no CUDA kernels before the
   profile says where the time goes, no UI before the hotkey path works.
5. **Learning checkpoint.** Each rung names one concept to understand before building (quiz first, then build).

## Where we are: rung 0 (done)

`internal/asr/ortengine` runs Parakeet TDT on the GTX 1080, token-exact against NeMo on the golden clips. The Go
mel frontend is NeMo-exact and about 2.5 ms per 10 s. `rbbench` measures WER, RTF and latency.

Still open from Phase 1: the uncontended full benchmark and baseline table. This is **time-boxed to rung 1**:
it runs in the background while rung 1 is built, and publishes whenever the NeMo reference finishes. It does not
block anything.

## The ladder

### Rung 1: `ramblocal transcribe file.wav`
Smallest thing that does the product's core job.
- `cmd/ramblocal` with one subcommand, plus `ramblocal doctor` (finds the model dir, ORT DLL, CUDA DLLs, reports
  which execution provider loaded and the VRAM used).
- Config by flags and env only. Falls back cuda to cpu automatically.
- **Gate:** CLI output equals golden text for the 5 clips; WER on a 200-clip test-clean slice within 0.3 of NeMo.
- **Concept:** how a transducer decodes (encoder frames, joint network, blank and duration heads).

### Rung 2: `ramblocal serve` (OpenAI-compatible)
- `POST /v1/audio/transcriptions` with `json`, `text` and `verbose_json` (word timestamps).
  `GET /v1/models` and `GET /healthz`.
- One engine, one request at a time (the engine mutex already serialises). `http.MaxBytesReader` on uploads,
  400 for bad audio, 413 for too large, 499/503 on cancel.
- Accepts wav directly; other containers via an optional ffmpeg shell-out, off by default.
- **Gate:** the official `openai` Python SDK transcribes a file against the local server unchanged; a Go
  contract test pins request and response shapes; load test shows no leaks over 500 requests.
- **Concept:** HTTP request lifecycle, context cancellation, why a GPU wants a queue.

### Rung 3: hold-to-talk dictation with a floating pill (the MVP of the product)
*Status: first implementation merged in PR #3 (`ramblocal dictate`, `ramblocal keys`), untested on real Windows. It was pulled ahead of rungs 2 and 4 because it is the product; the Wispr Flow-style floating pill is part of it.*

- One Windows binary, console only. Hold a key (default `F9`, configurable), speak, release; text is pasted
  into the focused window.
- Pieces, each tested alone: low-level keyboard hook (hold detection, key swallow), WASAPI capture to 16 kHz mono
  PCM, `SendInput` paste via clipboard (save and restore the clipboard).
- In-process engine, no HTTP hop. The model loads at start and stays warm.
- No VAD, no streaming, no formatting yet. Just record, transcribe, paste.
- **Gate:** release-to-text latency p50 under 400 ms and p95 under 800 ms for 5 s utterances on the GTX 1080,
  measured by an injected-audio harness (so it is reproducible without a microphone); a manual checklist for
  Notepad, a browser text box, VS Code and a terminal.
- **Concept:** audio capture buffers, 16 kHz resampling, why clipboard paste beats simulated typing.

At the end of rung 3 you can dictate into any app. Everything after this makes it better.

### Rung 4: daily-driver quality
- Trim leading and trailing silence (energy VAD) before transcribing, which also cuts latency.
- Text cleanup: filler removal (`um`, `uh`, repeats), capitalisation and spacing, number and punctuation rules.
  Parakeet already emits punctuation and capitals, so this is mostly light fixes.
- Tray icon with state (idle, recording, working), a small recording beep or overlay, config file
  (`%APPDATA%/ramblocal/config.toml`), start on login.
- **Gate:** a text-cleanup table test (input and expected output, 100+ cases); an eval set of 30 self-recorded
  dictation clips scored for WER and filler rate; idle CPU under 1 percent and idle VRAM equals the model size.
- **Concept:** why normalisation for scoring differs from formatting for display.

### Rung 5: history, stats, voice notes
- SQLite (pure Go driver) store: every dictation is saved (text only; **never audio by default**) with
  timestamp, app, duration and word count.
- `ramblocal history` and `ramblocal stats` (words per minute, streaks, daily totals; the leaderboard is a local
  table for now). Voice note mode: a second hotkey that saves to a note instead of pasting.
- **Gate:** store migrations tested on a copy of the previous schema; stats computed from a fixed fixture
  match hand-calculated values.
- **Concept:** schema migration, and what privacy-by-default means for stored text.

### Rung 6: dictionary and snippets
- Static first: user word list and phrase replacements applied after decoding (`ramblocal` not `ram bull local`),
  snippets such as "my email" expanding to text.
- Then adaptive: when a user corrects a word right after pasting, offer to add it. Optional contextual biasing
  inside the decoder is a later experiment, measured against post-edit replacement.
- **Gate:** a held-out set of names and jargon; term error rate with the dictionary must beat without it by a
  stated margin.
- **Concept:** biasing in decoding versus post-processing, and what each costs in latency.

### Rung 7: hands-free, commands and routines
- Hands-free mode: VAD-segmented continuous dictation with a toggle hotkey.
- Preset commands ("new line", "delete that", "select all") and routines (named sequences of commands and
  text), with a safe parser (**transcript text is never executed**; commands map to a fixed allow-list).
- Per-app styles (e.g. no capitalisation in a terminal, casual in chat).
- **Gate:** command-grammar tests including adversarial transcripts; end-to-end test that a spoken sentence
  containing the word "delete" is not mistaken for a command without the trigger phrase.
- **Concept:** endpointing and why streaming latency is dominated by when you decide the user stopped.

### Rung 8: streaming and partial results
- Chunked or cache-aware streaming so text appears while you speak, with a stable-prefix rule so already-pasted
  text is never rewritten.
- `stream=true` on the HTTP endpoint (server-sent events) for API compatibility.
- **Gate:** streamed final text equals offline text on the golden set within a stated tolerance; first-token latency
  measured and recorded.

### Rung 9: performance (only now, driven by profiles)
Everything above runs on the ORT engine. This rung exists because by now the profile tells us the real bottlenecks.
Candidates in likely order, each one a separate PR with a before and after number:
1. Device-side decode loop (the per-token ORT call costs 31-40 ms p50 today).
2. INT8 or FP16 weights (the FP32 floor is 2.4 GB; the VRAM target is re-baselined to 4.0 GB for FP32).
3. Custom CUDA kernels for whatever the profiler shows is hot. This needs CUDA Toolkit 12.9 and MSVC, which you
   install from an admin terminal.
- **Gate:** WER within 0.1 absolute of the ORT engine; RTF and p95 latency better by a stated factor.
- **Concept:** roofline thinking, memory-bound versus compute-bound, why Pascal has no tensor cores.

### Rung 10: UI and release
- Settings window, history browser, dictionary editor, onboarding that runs `doctor` for you.
- Signed installer, model download with checksum, versioned release notes, the NVIDIA love-letter README with the
  benchmark tables.
- **Gate:** a clean-VM install test; a published benchmark table reproducible from a single command.

## Mapping from the old phases

| Old phase | Where it went |
|---|---|
| 1 Engine v1 + bench | Rung 0 (done), results published during rung 1 |
| 2 API + scheduler | Rung 2 (serial engine only; the scheduler is deferred until there are two real callers) |
| 3 Live dictation | Rungs 3, 7, 8 |
| 4 Text, commands, routines | Rungs 4, 7 |
| 5 Store, stats, dictionary | Rungs 5, 6 |
| 6 Custom CUDA | Rung 9 |
| 7 UI + release | Rung 10 |

## Execution

- One feature branch per rung, merged to `main` before the next starts. We stop stacking PRs on PRs, which is
  what made the current stack (#1 and #2) awkward to land.
- Within a rung, independent pieces (for example rung 3: hook, capture, paste) go to parallel subagents in
  worktrees. Each piece is built test first, then reviewed, then merged.
- A short ledger entry per rung records the demo, the gate numbers and anything deferred.
- A rung is done when its gate passes in CI and its demo is recorded in the rung's section of this file.

## Decisions I need from you

1. **Hotkey default.** Fn is not visible to Windows as a normal key on most keyboards, so rung 3 defaults to `F9`
   or `Right Ctrl`, and Fn support depends on your keyboard. Which do you want?
2. **Rung 3 target apps.** Is the 4-app checklist (Notepad, browser, VS Code, terminal) the right one?
3. **Merging.** Please merge PR #1 so rung 1 can branch from `main`.
