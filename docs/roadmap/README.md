# ramblocal: Roadmap and Execution Plans

**Spec:** [../spec/README.md](../spec/README.md)
**Detailed plan for Phases 0 and 1:** [../plans/phase-0-1-engine-v1/README.md](../plans/phase-0-1-engine-v1/README.md)

Each phase has a goal, an exit gate (measured, not judged), a task list, and an execution plan saying which tasks run in parallel, on which model, and in what order. Phases 2-7 each get a detailed step-by-step plan, written when the phase before them passes its gate, because their details depend on Phase 0 and 1 measurements.

---

## How execution works (all phases)

**Orchestrator:** the main session (Opus). It never writes feature code itself. It:
1. Creates one git worktree per lane: branch `lane/<phase>-<name>`, created from `main`.
2. Dispatches one **Haiku 5.5 implementer subagent** per task, with the task text, the spec, and the `Interfaces` block. Implementers follow TDD: failing test → code → passing test → commit.
3. After each task, dispatches a **reviewer subagent (Sonnet 5.5)** that checks the diff against the task's acceptance criteria and the spec. A rejected task goes back to a fresh implementer with the review attached.
4. Merges a lane into `main` only when all its tasks pass review, `go test ./...` passes on `main` after the merge, and `go mod tidy` leaves no diff.
5. At each phase end, runs the phase gate and a whole-branch review on Opus.

**Model assignment rules:**
| Work | Implementer | Reviewer |
|---|---|---|
| Go packages with a full test spec (most tasks) | Haiku 5.5 | Sonnet 5.5 |
| Interface-defining tasks (other lanes depend on them) | Haiku 5.5 | Opus 5.5 |
| CUDA kernels, numerics, quantization (Phase 6) | Opus 5.5 | Opus 5.5 + golden tests |
| Hardware gates, benchmarks on the 1080 | Orchestrator runs them directly | Human reads the numbers |

Haiku implements against tests it cannot argue with. Where correctness is subtle and the test oracle is weak (kernels), a stronger model writes the code.

**GPU lock:** subagents run on the same machine. Only one task may use the GPU for **benchmarking** at a time, so timings aren't polluted. Unit tests that touch the GPU are fine concurrently; anything that writes to `bench/results/` takes the lock. `rbbench run` acquires it automatically (lock file `%TEMP%\ramblocal-gpu.lock`, see Phase 1 Task 1.8); other benchmark scripts run only while no `rbbench` is running.

**Shared-file rule:** only interface tasks may edit `internal/asr/asr.go`. Lanes may add Go dependencies; conflicts in `go.mod`/`go.sum` are resolved at merge by the orchestrator with `go mod tidy`.

---

## Dependency graph

```
Phase 0 (gate) ──► Phase 1 engine v1 ──────────────┬──► Phase 3b live pipeline ──► Phase 6 CUDA engine ──► Phase 7 release
        │                │ (Engine interface, T1.1)  │            ▲                        ▲
        │                └──► Phase 2 API ───────────┘            │                        │
        ├──► Phase 3a Windows plumbing ───────────────────────────┤                        │
        ├──► Phase 4 text, commands, routines ────────────────────┤                        │
        └──► Phase 5a store + stats ──► Phase 5b dictionary ──────┴────────────────────────┘
```

**Wave plan** (each wave's lanes run in parallel):

| Wave | Starts when | Parallel lanes |
|---|---|---|
| W0 | now | Phase 0 (two lanes: Windows hardware lane + Lambda export lane) |
| W1 | Phase 0 gate passes | P1 engine · P2 API (on a fake engine) · P3a Windows plumbing · P4 text/commands/routines · P5a store/stats |
| W2 | P1 gate passes | P3b live pipeline · P5b adaptive dictionary · P2 switches from fake to real engine |
| W3 | P3b gate passes | P6 CUDA engine (sequential swaps, one lane) · P7a UI (parallel) |
| W4 | P6 gate passes | P7b release |

---

## Phase 0: Hardware and toolchain gate

**Goal:** prove the 1080 and the toolchain can run the plan before any product code exists.

**Exit gate (all must hold):**
- `nvidia-smi` shows the GTX 1080; driver version recorded.
- A CUDA 12.x `nvcc` build of a trivial kernel compiled for `sm_61` runs and returns correct results.
- An MSVC-built DLL exporting a C function is called from Go via `syscall.LoadDLL`.
- `onnxruntime_go` loads an ORT build, and the CUDA EP (or DirectML EP, if CUDA fails) runs the exported Parakeet encoder on the 1080. The active EP is recorded.
- Golden fixtures and the exported model exist in `models/parakeet-tdt-0.6b-v2/` and `testdata/golden/`.
- Facts recorded in `bench/results/gtx1080/hardware.json`: driver, CUDA version, ORT version, active EP, VRAM free.

**Tasks:** see the [detailed plan](../plans/phase-0-1-engine-v1/README.md), Tasks 0.1-0.5.

**Execution plan:**
| Lane | Tasks | Where | Model |
|---|---|---|---|
| `p0-win` | 0.1 repo skeleton + CI, 0.2 toolchain probe, 0.3 DLL probe, 0.5 ORT probe | Dev machine (1080) | Haiku implementer; orchestrator runs probes |
| `p0-export` | 0.4 export + golden fixtures | Lambda instance (any NVIDIA GPU, Linux) | Haiku writes the script; orchestrator runs it on Lambda |

0.5 depends on 0.4 (needs the ONNX files). If the CUDA EP fails on sm_61 and DirectML also fails, **stop**: the plan's Phase 1 runtime changes, and the orchestrator comes back to you with options before W1.

---

## Phase 1: Engine v1 (ONNX Runtime) + benchmark harness

**Goal:** correct Parakeet transcription in Go on the 1080, and the measurement tools every later phase relies on.

**Exit gate:**
- Golden test: token IDs from `ortengine` equal NeMo's for all 5 fixture clips.
- `rbbench` on LibriSpeech test-clean: WER ≤ NeMo reference + 0.3.
- `rbbench` JSON committed with RTF, per-stage latency p50/p95, VRAM, cold start.
- faster-whisper baseline (`large-v3-turbo`, INT8, CUDA) measured on the same card and committed in the same JSON format.

**Tasks:** detailed plan Tasks 1.1-1.9: Engine interface and manifest · WAV I/O · tokenizer · TDT greedy decoder · ORT sessions · `ortengine` · WER + normalizer · `rbbench` · baseline script.

**Execution plan:**
| Lane | Tasks | Model |
|---|---|---|
| `p1-core` | 1.1 (interface; **merge first**, everything else consumes it), then 1.3, 1.4 | Haiku; 1.1 reviewed by Opus |
| `p1-io` | 1.2 WAV, 1.7 WER | Haiku |
| `p1-ort` | 1.5, 1.6 (after 1.1, 1.3, 1.4 merge) | Haiku; reviewer Sonnet; orchestrator runs golden test on 1080 |
| `p1-bench` | 1.8 `rbbench`, 1.9 baseline | Haiku; orchestrator runs with GPU lock |

---

## Phase 2: OpenAI-compatible API server + scheduler

**Goal:** any OpenAI client can transcribe through ramblocal; live work always preempts API work.

**Exit gate:**
- Conformance suite: the official `openai` Python client (pinned version) calls `audio.transcriptions.create` with every `response_format` and with `stream=True`, against the server on a fake engine; all pass in CI.
- Same suite passes against the real engine on the 1080.
- Scheduler test: with 8 queued API jobs, a live job starts within one in-flight job's duration (no head-of-line blocking beyond the current batch).
- `rbbench --target http://127.0.0.1:8765/v1 --concurrency 1,4,8` throughput committed.

**Tasks:**
1. `internal/sched`: priority queue with two classes, max in-flight 1 on GPU, batch up to N API jobs of similar length. Tests with a fake engine and virtual clock.
2. `internal/api`: multipart parsing, 25 MB limit, `model` validation against `/v1/models`, OpenAI error bodies.
3. Response formatters: json, text, srt, vtt, verbose_json (segments + words). Golden-string tests.
4. SSE streaming: `transcript.text.delta` / `transcript.text.done` events, matching the OpenAI streaming transcription event shape. **Verify the exact event schema against current OpenAI docs at task start.**
5. Audio decoding for API uploads: wav natively; flac/mp3/ogg via a pure-Go decoder or a bundled `ffmpeg` (decision in the Phase 2 plan, by benchmark).
6. Long-file segmentation: VAD split ≤ 30 s chunks, timestamps offset correctly. Test with a 10-minute fixture.
7. Auth: optional bearer token; refuse non-loopback bind without one.
8. `tests/conformance/` Python suite + CI job.

**Execution plan:** one lane per task group, all Haiku: `p2-sched` (1), `p2-http` (2, 3, 7), `p2-stream` (4), `p2-decode` (5, 6), `p2-conformance` (8). Runs in W1 against `asr.FakeEngine`; switches to the real engine in W2.

---

## Phase 3: Live dictation

### 3a. Windows plumbing (W1, no engine needed)

**Tasks:**
1. `internal/win/hotkey`: `WH_KEYBOARD_LL` hook on a `runtime.LockOSThread` goroutine with a message loop; hold, release, double-tap detection; chord config. Tests: state machine unit-tested with synthetic key events; integration via `SendInput` in a test.
2. `internal/audio/capture`: malgo capture at 16 kHz mono (device-native rate resampled if needed), C callback writes into a lock-free SPSC ring buffer; the mic opens at app start and stays open (pre-roll 300 ms kept so the first word is not clipped).
3. `internal/win/inject`: clipboard save → set → `Ctrl+V` via SendInput → restore after 250 ms; SendInput `KEYEVENTF_UNICODE` fallback. Test against a helper window process that records what it receives.
4. Active-app detection: foreground window → process name, for per-app style and stats.

### 3b. Live pipeline (W2)

**Tasks:**
1. `internal/vad`: Silero VAD (ONNX, CPU EP) with energy-gate fallback; segmenter implementing spec §5 (300 ms pause close, 20 s force-close with 1 s overlap).
2. Segment dedup across overlap by word timestamps.
3. Pipeline wiring: hotkey → capture → segmenter → scheduler (live class) → text pipeline → inject; command key routes to the router instead.
4. Stage timers for every hop, into `internal/metrics`.
5. Replay harness: `rbbench live --replay <wav> --release-at <s>` drives the pipeline with a virtual clock and a fake injector; reports key-release → injected latency.
6. Minimal overlay: a small always-on-top window showing "listening / transcribing" (Wails or a raw Win32 layered window; decided in the Phase 3 plan).

**Exit gate:** replay harness on 20 recorded utterances (2-30 s): **p50 ≤ 150 ms, p95 ≤ 300 ms** key release → injected; WER on those utterances within 0.5 of offline transcription of the same audio (segmentation cost).

**Execution plan:** 3a: four Haiku lanes in W1. 3b: one lane `p3-live` (tasks in order 1 → 2 → 3 → 4 → 5), one lane `p3-overlay` (6) in parallel. Orchestrator runs the latency gate with the GPU lock. If p50 misses, the orchestrator profiles stage timers and reports which stage blew the budget **before** any optimization work starts.

---

## Phase 4: Text pipeline, commands, routines (W1)

**Tasks:**
1. `internal/text/cleanup`: filler removal (word-boundary aware; never removes "um" inside words), "scratch that" handling, whitespace and punctuation repair.
2. Replacements (case-preserving) and snippets.
3. Per-app style rules from config.
4. `internal/router`: normalization, exact match, token-set-ratio fuzzy match, threshold 90, `unmatched` log.
5. `internal/routines`: YAML schema + validation (unknown step types rejected at load), executor, dry-run, risky-step confirmation via a `Confirmer` interface (tray prompt in production, fake in tests). `run` steps never receive transcript text.
6. Preset file loader with hot reload.

**Exit gate:** all unit tests pass; a table of 100 hand-written (heard phrase → expected action or no action) cases passes with zero false positives.

**Execution plan:** three Haiku lanes: `p4-text` (1-3), `p4-router` (4, 6), `p4-routines` (5). Reviewer for `p4-routines` is Opus (it executes things).

---

## Phase 5: Store, stats, adaptive dictionary

### 5a. Store and stats (W1)

**Tasks:**
1. `internal/store`: schema + migrations (`utterances`, `notes`, `dictionary`, `daily_stats`), FTS5 index on text, WAL mode.
2. Stats queries: words per day/week/all, WPM, time saved (`words / typing_wpm - speaking_seconds`), streaks, top apps, personal bests. Tests with seeded fixtures across day boundaries and time zones.
3. Voice notes: create, title from first sentence, search across notes + history.

### 5b. Adaptive dictionary (W2)

**Tasks:**
1. `internal/dict`: manual add/remove; candidate proposals with counts; accept/reject.
2. `internal/win/uia`: read the focused element's text via UI Automation; skip password fields (`IsPassword`).
3. Correction learner: after a paste, poll the focused control for up to 10 s; align pasted text vs. current text; propose replaced words within edit distance ≤ 3 that are not dictionary words already.
4. Feed accepted terms into replacements (Phase 4) and `asr.Options.BiasWords` (consumed by Phase 6 biasing).

**Exit gate:** store + stats tests pass; correction learner proposes the right pair in 9/10 scripted correction scenarios against the helper window and proposes nothing in 10/10 no-edit scenarios.

**Execution plan:** 5a: two Haiku lanes (`p5-store`, `p5-stats`). 5b: one lane `p5-dict`, Haiku, reviewer Sonnet.

---

## Phase 6: Custom CUDA engine (W3)

**Goal:** replace ONNX Runtime with our own CUDA engine, one stage at a time, each swap proven equal or better.

**Rule for every swap:** golden test passes (tokens identical; tensors within tolerance), and `rbbench` shows no WER regression > 0.2 and no latency regression on any stage. The commit message includes the before/after numbers.

**Tasks, in order:**
1. `engine/` scaffold: CMake + nvcc + MSVC, `rb_engine.h` C ABI, error handling, weight loading from the exported model (convert ONNX initializers to a flat `.rbw` weight file with a tool in `tools/export`).
2. `internal/asr/cudaengine`: Go binding via `syscall.LoadDLL`; a hybrid engine that can run any stage on ORT or CUDA, so stages swap independently.
3. Swap 1, mel frontend: preemphasis + STFT (cuFFT) + mel + log + per-feature normalization, fused where possible. Golden mel tolerance 1e-4.
4. Swap 2, TDT decoder + joint: LSTM prediction net and joint on GPU, greedy loop on device with **CUDA graphs**, removing per-token host round trips. Tokens must equal golden.
5. Swap 3, encoder FP32: FastConformer blocks (subsampling conv, MHSA with relative positional encoding, conv module, FFN) with cuBLAS SGEMM + fused elementwise kernels. Encoder output tolerance 1e-3.
6. Swap 4, encoder INT8: per-channel weight quantization, `dp4a` GEMM kernels, FP32 accumulation and activations. Gate: WER within 0.3 of FP32; VRAM ≤ 1.5 GB.
7. Context biasing: boosting trie over `BiasWords` applied in TDT decoding. Gate: recall of biased terms on a 50-utterance jargon set improves ≥ 30% with WER on LibriSpeech not worse by > 0.1.
8. Spike: cache-aware streaming FastConformer checkpoint vs. spec §5 segmentation. Report only; adopt only if it wins on both latency and WER.
9. Lambda GPU matrix: run `rbbench` on 3-4 Lambda GPU types; commit results.

**Execution plan:** one sequential lane `p6-engine` (swaps must land in order; each changes the baseline for the next). **Opus implementer** for 3-7, Haiku for 1, 2, 9 scaffolding. Orchestrator holds the GPU lock for every gate run. Task 8 runs in a parallel spike lane on Lambda.

---

## Phase 7: UI and release

### 7a. UI (W3, parallel with Phase 6)
1. Tray icon: status (ready / listening / transcribing / error / active EP), quick toggles.
2. Settings window (Wails): hotkeys, model, presets, routines editor, dictionary review (accept/reject candidates).
3. History + voice notes view with search, re-paste, re-run.
4. Stats / leaderboard view.

### 7b. Release (W4)
1. Installer (single exe + engine DLL + model download on first run).
2. README with the benchmark table (Parakeet on GTX 1080 vs. faster-whisper on the same card, plus the Lambda matrix), `docs/engine/README.md`, API docs, contributing guide.
3. License and model-license check; credits to NVIDIA NeMo and the Parakeet authors.
4. Tag v0.1.0.

**Exit gate:** fresh Windows 11 VM with a Pascal or newer GPU: install → first dictation in under 5 minutes following only the README; all phase gates re-run green on the release commit.

**Execution plan:** 7a: four Haiku lanes. 7b: one lane, Haiku for docs drafts, final README review by you.

---

## What I need from you at each gate

| Gate | Your call |
|---|---|
| Phase 0 | Approve the active EP and toolchain versions, or choose the fallback if CUDA EP fails on sm_61. |
| Phase 1 | Approve the numbers vs. the Whisper baseline before building the app on top. |
| Phase 3 | Try dictation yourself for a day; report anything that feels slow or wrong. |
| Phase 6 | Approve each swap's before/after numbers. |
| Phase 7 | Final README and release. |
