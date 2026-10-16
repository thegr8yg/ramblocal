# ramblocal: Design Spec

**Date:** 2026-10-15
**Status:** Approved in conversation; written spec pending review.
**One line:** A fully local, GPU-accelerated voice dictation and command tool for Windows, built on NVIDIA Parakeet, with an OpenAI-compatible transcription API, tuned to run extremely well on cheap hardware (baseline: GTX 1080, Pascal, 8 GB).

Related: [Roadmap and execution plans](../roadmap/README.md) · [Phase 0-1 plan](../plans/phase-0-1-engine-v1/README.md)

---

## 1. Goals

1. **Fast.** Key release to text pasted: **p50 ≤ 150 ms, p95 ≤ 300 ms** for utterances up to 30 s on a GTX 1080.
2. **Accurate.** WER on LibriSpeech test-clean within **0.3 absolute** of the NeMo reference implementation for the same checkpoint, using our normalizer.
3. **Local.** No audio, text, or telemetry leaves the machine. No cloud ASR. No LLM decides what to execute.
4. **Compatible.** `POST /v1/audio/transcriptions` works with the official `openai` Python and JS clients unmodified (only `base_url` changes).
5. **Feature parity with commercial dictation apps** (Wispr Flow / SuperWhisper class): hold-to-talk, hands-free mode, history, voice notes, stats, adaptive dictionary, snippets, per-app styles, filler removal.
6. **NVIDIA-first.** Parakeet (FastConformer + TDT) is the primary engine. Whisper is a measured baseline in the evals, not a runtime option.
7. **Open-source quality.** Reproducible benchmarks, CI, a published GPU comparison matrix, docs a stranger can follow.

## 2. Non-goals (v1)

- macOS / Linux desktop agent. (The API server and engine must build on Linux for Lambda benchmarking; the desktop agent is Windows-only.)
- Training or fine-tuning models.
- Any cloud fallback.
- LLM-based rewriting of dictated text.
- Wake words.

## 3. Hardware and toolchain baseline

| Item | Value |
|---|---|
| GPU | GTX 1080, sm_61, 8 GB, ~320 GB/s, no tensor cores, FP16 at 1/64 rate, INT8 `dp4a` available |
| OS | Windows 11 |
| CUDA toolkit | 12.x, pinned. **Verify in Phase 0:** CUDA 13 removed sm_61 support. |
| Host language | Go, latest stable (≥ 1.24) |
| C toolchain for cgo | MinGW-w64 gcc (needed by `onnxruntime_go` and `malgo`) |
| Engine language | CUDA C++, built as a DLL with a plain C ABI, compiled with nvcc + MSVC. Loaded from Go with `syscall.LoadDLL` (no cgo across the MSVC/MinGW boundary). |
| Phase-1 runtime | ONNX Runtime via `github.com/yalue/onnxruntime_go`. Execution provider order: CUDA → DirectML → CPU. **Verify in Phase 0:** which ORT GPU build runs on sm_61. |
| Benchmark GPUs | Lambda cloud instances, comparison matrix only. The product targets the 1080. |

## 4. Architecture

One Go binary, `ramblocal.exe`, containing the desktop agent and the API server. One engine instance; one copy of the model in VRAM.

```
   ┌─ Live path ─────────────────────┐    ┌─ API path ────────────────────┐
   │ hotkey → mic (WASAPI, C cb)     │    │ HTTP /v1/audio/transcriptions │
   │ → ring buffer → VAD segmenter   │    │ → decode/resample → segment   │
   └───────────────┬─────────────────┘    └───────────────┬───────────────┘
                   └──────────► Scheduler ◄───────────────┘
                     (live = high priority; API = low, batched)
                                  │
        Engine interface ── ortengine (Phase 1) / cudaengine (Phase 6)
        mel → FastConformer encoder → TDT decoder+joint → detokenize
                                  │
   ┌─ Live ───────────────────────┴───────┐    ┌─ API ──────────────────┐
   │ text pipeline → router               │    │ OpenAI formatter       │
   │   ├─ dictation → inject              │    │ json/text/srt/vtt,     │
   │   └─ command → matcher → action      │    │ verbose_json, SSE      │
   └───────────────┬──────────────────────┘    └────────────────────────┘
                   ▼
     SQLite store: history, voice notes, stats, dictionary
                   │
     adaptive dictionary ──► engine biasing + text replacements
```

Every stage records a monotonic timestamp; `/metrics` and the bench harness read them.

### 4.1 Packages

| Package | Responsibility |
|---|---|
| `cmd/ramblocal` | Main binary: wires agent + API + engine. |
| `cmd/rbbench` | Benchmark/eval CLI: WER, RTF, latency percentiles, VRAM. Targets an in-process engine or any OpenAI-compatible URL. |
| `internal/audio` | WAV read/write, resampling, ring buffer, capture (malgo / miniaudio; the callback runs in C). |
| `internal/vad` | Voice activity detection and segmentation. |
| `internal/asr` | `Engine` interface, `Result` types, model manifest, tokenizer, TDT greedy decoder (Go reference). |
| `internal/asr/ortengine` | ONNX Runtime implementation of `Engine`. |
| `internal/asr/cudaengine` | Go binding to the `engine/` DLL; implementation of `Engine`. |
| `engine/` | CUDA C++ engine DLL: `include/rb_engine.h`, kernels, CMake build. |
| `internal/sched` | Priority scheduler between live and API work; batching. |
| `internal/api` | OpenAI-compatible HTTP handlers. |
| `internal/text` | WER normalizer, cleanup (fillers, "scratch that"), replacements, snippets, per-app style. |
| `internal/router` | Command matching: exact, then fuzzy with threshold. |
| `internal/routines` | YAML routine executor: `notify`, `run`, `url`, `hotkey`, `wait`; dry-run; confirmation for risky steps. |
| `internal/win/hotkey` | Low-level keyboard hook (`WH_KEYBOARD_LL`) on a locked OS thread with its own message loop. |
| `internal/win/inject` | Clipboard paste with restore; SendInput Unicode fallback. |
| `internal/win/uia` | UI Automation read of the focused text control, for correction learning. |
| `internal/store` | SQLite (`modernc.org/sqlite`, pure Go): utterances, notes, stats, dictionary; FTS5 search. |
| `internal/dict` | Adaptive dictionary: learn from corrections and manual adds; emit biasing list and replacements. |
| `internal/metrics` | Stage timers, histograms, `/metrics` in Prometheus text format. |
| `internal/config` | YAML config, defaults, validation. |
| `tools/export` | Python, run on Linux/Lambda: export Parakeet to ONNX, write the manifest, generate golden fixtures. |
| `tools/baseline` | Python: faster-whisper baseline that writes the same result JSON as `rbbench`. |
| `bench/` | Dataset manifests, prep scripts, committed result JSON per GPU. |

### 4.2 Core interfaces (Go)

```go
package asr

type Word struct {
	Text       string
	Start, End float64 // seconds
}

type StageTimings struct {
	Mel, Encoder, Decoder, Detok time.Duration
}

type Result struct {
	Text     string
	Tokens   []int
	Words    []Word
	AudioSec float64
	Timings  StageTimings
}

type Options struct {
	Language  string   // "" = model default
	BiasWords []string // adaptive dictionary terms; ignored by engines without biasing
}

type Engine interface {
	// Transcribe a complete 16 kHz mono float32 clip.
	Transcribe(ctx context.Context, pcm []float32, opts Options) (Result, error)
	Name() string
	Close() error
}
```

The engine DLL C ABI (Phase 6) mirrors this: `rb_engine_create`, `rb_engine_transcribe`, `rb_result_free`, `rb_engine_destroy`, each returning an error code, plus `rb_last_error`.

### 4.3 Model

- Default: `nvidia/parakeet-tdt-0.6b-v2` (English). Option: `nvidia/parakeet-tdt-0.6b-v3` (multilingual).
- `tools/export` writes `models/<name>/`: `preprocessor.onnx`, `encoder.onnx`, `decoder_joint.onnx`, `vocab.txt`, `manifest.json` (tensor names, blank id, TDT durations, subsampling factor, sample rate, feature count, prediction-net state shapes).
- Golden fixtures generated by NeMo for 5 clips: mel features, encoder output, token IDs, text. Every engine implementation is tested against them.
- Models are not committed to git. `tools/export` output is uploaded as a release asset; `scripts/fetch-model.ps1` downloads it.

## 5. Streaming strategy (decision)

Parakeet TDT is a full-context model. v1 uses **VAD-segmented incremental transcription**:

1. While the key is held, the VAD segmenter closes a segment at each pause ≥ 300 ms.
2. Closed segments are transcribed immediately, in the background, while the user keeps talking.
3. A segment with no pause is force-closed at 20 s with 1 s overlap; overlap words are deduplicated by timestamp.
4. On key release, only the open trailing segment remains to transcribe.

Latency after release is bounded by the length of the last segment, not the whole utterance. A cache-aware streaming FastConformer checkpoint is evaluated as a spike in Phase 6 and adopted only if it beats this on both WER and latency in the bench.

## 6. Features

| Feature | Behavior |
|---|---|
| Hold-to-talk | Default `Ctrl+Win`, configurable. Fn only where the keyboard reports it to Windows. |
| Hands-free | Double-tap the dictation key (within 300 ms) to start; tap again to stop. |
| Command key | Separate key (default `Ctrl+Alt+Win`). Transcript goes to the command router, never to injection. |
| Voice note key | Toggle (default `Ctrl+Win+N`). Long-form; saved as a note, not injected. |
| Injection | Clipboard paste, previous clipboard restored after 250 ms. SendInput Unicode fallback for apps on a config list. |
| Cleanup | Filler removal (`um`, `uh`, `erm`; configurable). "Scratch that" deletes the previous sentence of the current utterance. Dictionary replacements. Snippet expansion (`"insert signature"` → stored text). |
| Per-app style | Rules keyed by process name, e.g. no trailing period in chat apps, no capitalization in terminals. |
| Commands | Exact match on normalized text, then fuzzy (token-set ratio) ≥ 90. Below threshold: no action, logged as `unmatched`. |
| Routines | YAML steps `notify`, `run`, `url`, `hotkey`, `wait`. Steps with `risky: true`, or whose `run` contains delete/send/shutdown keywords, require confirmation. `--dry-run` prints steps only. |
| History | Every utterance: final text, raw text, app, stage timings, optional audio (off by default). FTS5 search. Re-paste. |
| Voice notes | Saved with title = first sentence; searchable alongside history. |
| Stats / leaderboard | Words dictated (day / week / all time), average dictation WPM vs. configured typing WPM, time saved, streak days, top apps, personal bests (longest utterance, fastest WPM, longest streak). |
| Adaptive dictionary | Sources: manual add; correction learning (after a paste, read the focused control via UIA for up to 10 s; if a pasted word was replaced by a word within edit distance ≤ 3, propose the pair). Accepted terms feed replacements and engine biasing. Never reads password fields. |
| API | `POST /v1/audio/transcriptions` with `file`, `model`, `language`, `prompt`, `response_format` (json, text, srt, vtt, verbose_json), `timestamp_granularities[]`, `stream`. `GET /v1/models`. Binds 127.0.0.1 by default; optional bearer token. |

## 7. Performance requirements

| Metric | Target (GTX 1080) | Measured by |
|---|---|---|
| Key release → pasted, p50 / p95 | ≤ 150 ms / ≤ 300 ms | stage timers; replay harness with recorded audio |
| RTF, offline, 1 stream | reported; must not regress > 5% | `rbbench` |
| WER, LibriSpeech test-clean | ≤ NeMo reference + 0.3 | `rbbench` |
| VRAM, engine resident | ≤ 3.0 GB (Phase 1), ≤ 1.5 GB (Phase 6, INT8) | NVML in `rbbench` |
| Cold start to ready | ≤ 5 s | `rbbench --cold` |
| Idle CPU | < 1% | Windows perf counter, manual check per release |
| API throughput | reported, audio-seconds per second at concurrency 1/4/8 | `rbbench --target <url> --concurrency N` |

Bench results are committed as JSON under `bench/results/<gpu>/<date>.json`. A regression > 5% on any latency metric or > 0.2 WER fails the gate.

## 8. Error handling

- Engine init failure: fall down the EP chain (CUDA → DirectML → CPU). Log the active EP and expose it in the tray and `/v1/models`.
- Mic unavailable: tray error state; the hotkey shows a toast instead of recording.
- Inference error mid-utterance: keep audio in memory, retry once, then save to history as `failed` with audio so it can be re-run.
- Injection failure (focus lost): text stays on the clipboard and in history; a toast says so.
- API: OpenAI-shaped error bodies (`{"error":{"message","type","code"}}`). 400 bad input, 413 over the 25 MB default limit, 503 when the queue is full.

## 9. Security

- The transcript is never passed to a shell, interpreter, or LLM for execution. Commands only select preset actions.
- `run` steps execute only commands written in the routines file; no transcript interpolation.
- API binds 127.0.0.1 unless configured; a non-loopback bind requires a token.
- No telemetry.

## 10. Testing strategy

- Unit tests per package (Go `testing`, table-driven).
- Golden tests: every engine matches NeMo fixtures (token IDs exact for greedy decoding; features and encoder outputs within tolerance).
- Conformance: the official `openai` Python client runs in CI against the server backed by a fake engine.
- Replay harness: recorded WAVs played through the live pipeline with a virtual clock, for deterministic latency and segmentation tests.
- Bench gate: `rbbench` on the 1080 before merging any engine change.
- Windows integration (hooks, injection, UIA) tested against a test-window helper process, on the dev machine or a self-hosted Windows runner.

## 11. Open source and NVIDIA framing

- License: Apache-2.0 (matches NeMo).
- README leads with the benchmark table: Parakeet on a 2016 GPU vs. Whisper on the same card.
- `docs/engine/README.md`: how the CUDA engine works, the Pascal-specific choices (`dp4a`, FP32 activations), what was learned.
- Credits NVIDIA NeMo and the Parakeet authors; states each checkpoint's license (verify per checkpoint before release).
