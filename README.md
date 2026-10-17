# ramblocal

Local, GPU-accelerated voice dictation for Windows, built on NVIDIA Parakeet, with an OpenAI-compatible transcription API.

Status: pre-alpha. See [docs/spec](docs/spec/README.md), [docs/roadmap](docs/roadmap/README.md) and the [MVP ladder](docs/plans/mvp-ladder/README.md).

## Rung 1: transcribe a file

Build (Windows needs MinGW-w64 gcc on PATH for cgo):

    go build -o ramblocal.exe ./cmd/ramblocal

Point it at the exported model (`tools/export`) and an ONNX Runtime build with the CUDA provider:

    set RAMBLOCAL_MODEL_DIR=C:\path\to\models\parakeet-tdt-0.6b-v2
    set RAMBLOCAL_ORT_LIB=C:\path\to\onnxruntime.dll

    ramblocal doctor                      # checks model, ORT, GPU, loads the engine, runs a self-test
    ramblocal transcribe clip.wav         # prints the text (16 kHz WAV)
    ramblocal transcribe --json clip.wav  # text, audio_sec, ep and word timestamps
    ramblocal transcribe --ep cpu clip.wav

Execution providers fall back cuda, then dml, then cpu. `doctor` and `--timings` show which one loaded.
See `bench/results/gtx1080/environment.json` for the CUDA and cuDNN versions known to work on a GTX 1080
(cuDNN 9.10.2; 9.27 fails on Pascal).

Verify against NeMo on the golden clips (skips without a model):

    go test ./cmd/ramblocal -run TestTranscribeGolden -v
