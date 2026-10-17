package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/ortengine"
)

// doctorManifest is a valid manifest with a legacy ONNX preprocessor (no frontend section).
const doctorManifest = `{
  "name": "test", "sample_rate": 16000, "num_features": 128, "hop_seconds": 0.01,
  "subsampling": 8, "encoder_dim": 1024, "vocab_size": 1024, "blank_id": 1024,
  "tdt_durations": [0,1,2,3,4], "pred_layers": 2, "pred_hidden": 640,
  "preprocessor": {"file":"p.onnx","inputs":["a","b"],"outputs":["c","d"],"dtypes":{}},
  "encoder": {"file":"e.onnx","inputs":["a","b"],"outputs":["c","d"],"dtypes":{}},
  "decoder_joint": {"file":"d.onnx","inputs":["a","b","c","d","e"],"outputs":["f","g","h","i"],"dtypes":{}}
}`

// fakeEngine stands in for ortengine.Engine.
type fakeEngine struct {
	ep      string
	err     error
	samples int
	calls   int
}

func (f *fakeEngine) Transcribe(_ context.Context, pcm []float32, _ asr.Options) (asr.Result, error) {
	f.calls++
	f.samples = len(pcm)
	return asr.Result{}, f.err
}
func (f *fakeEngine) Name() string { return "fake-" + f.ep }
func (f *fakeEngine) Close() error { return nil }
func (f *fakeEngine) EP() string   { return f.ep }

// doctorEnv records what doctor passed to the engine constructor.
type doctorEnv struct {
	cfg      *ortengine.Config
	gpuCalls int
}

// stubDoctor swaps the package hooks for fakes and clears the environment variables that
// findModelDir and findORTLib read. A nil eng makes the engine constructor fail.
func stubDoctor(t *testing.T, eng *fakeEngine, gpu func(call int) (gpuInfo, error)) *doctorEnv {
	t.Helper()
	t.Setenv("RAMBLOCAL_MODEL_DIR", "")
	t.Setenv("RAMBLOCAL_ORT_LIB", "")
	env := &doctorEnv{}
	oldEngine, oldGPU := newDoctorEngine, queryGPU
	t.Cleanup(func() { newDoctorEngine, queryGPU = oldEngine, oldGPU })
	newDoctorEngine = func(cfg ortengine.Config) (doctorEngine, error) {
		env.cfg = &cfg
		if eng == nil {
			return nil, errors.New("no execution provider worked")
		}
		return eng, nil
	}
	queryGPU = func() (gpuInfo, error) {
		call := env.gpuCalls
		env.gpuCalls++
		return gpu(call)
	}
	return env
}

// gpuBeforeAfter reports 100 MB used before the engine loads and 350 MB after.
func gpuBeforeAfter(call int) (gpuInfo, error) {
	used := 100
	if call > 0 {
		used = 350
	}
	return gpuInfo{Name: "GTX 1080", Driver: "550.54", TotalMB: 8192, UsedMB: used}, nil
}

func gpuDown(int) (gpuInfo, error) {
	return gpuInfo{}, errors.New("nvidia-smi: executable file not found")
}

// fixtureModel writes a model directory with manifest.json and every file the manifest needs,
// except those named in skip.
func fixtureModel(t *testing.T, skip ...string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(doctorManifest), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"vocab.txt", "p.onnx", "e.onnx", "d.onnx"} {
		if containsString(skip, f) {
			continue
		}
		if err := os.WriteFile(filepath.Join(dir, f), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func fixtureORT(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "libonnxruntime.so")
	if err := os.WriteFile(p, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func containsString(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func TestDoctorMissingModelDir(t *testing.T) {
	env := stubDoctor(t, &fakeEngine{ep: "cuda"}, gpuBeforeAfter)
	var out bytes.Buffer
	missing := filepath.Join(t.TempDir(), "absent")
	err := runDoctor([]string{"--model", missing, "--ort", fixtureORT(t)}, &out, io.Discard)
	if err == nil {
		t.Fatal("want error for missing model directory")
	}
	if !strings.Contains(out.String(), "FAIL  model") || !strings.Contains(out.String(), "model directory not found") {
		t.Fatalf("report lacks model FAIL:\n%s", out.String())
	}
	if env.cfg != nil {
		t.Fatal("engine was built without a model directory")
	}
}

func TestDoctorMissingModelFiles(t *testing.T) {
	env := stubDoctor(t, &fakeEngine{ep: "cuda"}, gpuBeforeAfter)
	dir := fixtureModel(t, "vocab.txt", "e.onnx")
	var out bytes.Buffer
	err := runDoctor([]string{"--model", dir, "--ort", fixtureORT(t)}, &out, io.Discard)
	if err == nil {
		t.Fatal("want error for missing model files")
	}
	got := out.String()
	for _, want := range []string{"missing vocab.txt", "missing e.onnx"} {
		if !strings.Contains(got, want) {
			t.Errorf("report lacks %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "missing d.onnx") || strings.Contains(got, "missing p.onnx") {
		t.Errorf("report lists files that exist:\n%s", got)
	}
	if !strings.Contains(got, "skip  engine") {
		t.Errorf("engine check should be skipped:\n%s", got)
	}
	if env.cfg != nil {
		t.Fatal("engine was built with missing model files")
	}
}

func TestModelFilesFrontend(t *testing.T) {
	m := &asr.Manifest{
		Encoder:      asr.Graph{File: "e.onnx"},
		DecoderJoint: asr.Graph{File: "d.onnx"},
		Preprocessor: asr.Graph{File: "p.onnx"},
		Frontend:     &asr.Frontend{Filterbank: "fb.f32", Window: "w.f32"},
	}
	got := strings.Join(modelFiles(m), " ")
	if got != "vocab.txt e.onnx d.onnx fb.f32 w.f32" {
		t.Fatalf("modelFiles = %q", got)
	}
}

func TestDoctorORTMissing(t *testing.T) {
	env := stubDoctor(t, &fakeEngine{ep: "cuda"}, gpuBeforeAfter)
	var out bytes.Buffer
	err := runDoctor([]string{"--model", fixtureModel(t), "--ort", filepath.Join(t.TempDir(), "nolib.so")}, &out, io.Discard)
	if err == nil {
		t.Fatal("want error for missing ONNX Runtime library")
	}
	if !strings.Contains(out.String(), "FAIL  onnxruntime") {
		t.Fatalf("report lacks onnxruntime FAIL:\n%s", out.String())
	}
	if env.cfg != nil {
		t.Fatal("engine was built without ONNX Runtime")
	}
}

func TestDoctorGPUFailureWarns(t *testing.T) {
	env := stubDoctor(t, &fakeEngine{ep: "cpu"}, gpuDown)
	var out bytes.Buffer
	err := runDoctor([]string{"--model", fixtureModel(t), "--ort", fixtureORT(t)}, &out, io.Discard)
	if err != nil {
		t.Fatalf("GPU failure must not fail doctor: %v\n%s", err, out.String())
	}
	got := out.String()
	if !strings.Contains(got, "warn  gpu") || !strings.Contains(got, "CPU still works") {
		t.Errorf("report lacks gpu warn:\n%s", got)
	}
	if strings.Contains(got, "FAIL") || strings.Contains(got, "fell back") {
		t.Errorf("unexpected FAIL or fallback warn on CPU-only machine:\n%s", got)
	}
	if env.cfg == nil {
		t.Fatal("engine was not built")
	}
}

func TestDoctorAllOK(t *testing.T) {
	eng := &fakeEngine{ep: "cuda"}
	env := stubDoctor(t, eng, gpuBeforeAfter)
	dir := fixtureModel(t)
	lib := fixtureORT(t)
	var out bytes.Buffer
	if err := runDoctor([]string{"--model", dir, "--ort", lib}, &out, io.Discard); err != nil {
		t.Fatalf("runDoctor: %v\n%s", err, out.String())
	}
	got := out.String()
	for _, want := range []string{"ep cuda", "+250 MB since before load", "self-test", "1 s of silence"} {
		if !strings.Contains(got, want) {
			t.Errorf("report lacks %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "FAIL") || strings.Contains(got, "warn") {
		t.Errorf("unexpected FAIL or warn:\n%s", got)
	}
	if env.cfg.ModelDir != dir || env.cfg.ORTLib != lib || len(env.cfg.EPs) != 0 {
		t.Errorf("engine config = %+v", *env.cfg)
	}
	if eng.calls != 1 || eng.samples != asr.SampleRate {
		t.Errorf("self-test: %d calls with %d samples, want 1 call with %d", eng.calls, eng.samples, asr.SampleRate)
	}
}

func TestDoctorCUDAFallbackWarns(t *testing.T) {
	stubDoctor(t, &fakeEngine{ep: "cpu"}, gpuBeforeAfter)
	var out bytes.Buffer
	err := runDoctor([]string{"--model", fixtureModel(t), "--ort", fixtureORT(t)}, &out, io.Discard)
	if err != nil {
		t.Fatalf("fallback is a warning, not a failure: %v\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "warn  ep") || !strings.Contains(out.String(), "fell back to cpu") {
		t.Errorf("report lacks fallback warn:\n%s", out.String())
	}
}

func TestDoctorEPFlagLimitsProviders(t *testing.T) {
	env := stubDoctor(t, &fakeEngine{ep: "cpu"}, gpuDown)
	var out bytes.Buffer
	err := runDoctor([]string{"--model", fixtureModel(t), "--ort", fixtureORT(t), "--ep", "cuda, cpu"}, &out, io.Discard)
	if err != nil {
		t.Fatalf("runDoctor: %v\n%s", err, out.String())
	}
	if env.cfg == nil || strings.Join(env.cfg.EPs, ",") != "cuda,cpu" {
		t.Fatalf("EPs = %v, want [cuda cpu]", env.cfg)
	}
	// cuda was requested explicitly, so the CPU result is reported as a fallback.
	if !strings.Contains(out.String(), "fell back to cpu") {
		t.Errorf("report lacks fallback warn:\n%s", out.String())
	}
}

func TestDoctorEngineFailureFails(t *testing.T) {
	stubDoctor(t, nil, gpuDown)
	var out bytes.Buffer
	err := runDoctor([]string{"--model", fixtureModel(t), "--ort", fixtureORT(t)}, &out, io.Discard)
	if err == nil {
		t.Fatal("want error when the engine does not load")
	}
	if !strings.Contains(out.String(), "FAIL  engine") {
		t.Errorf("report lacks engine FAIL:\n%s", out.String())
	}
}

func TestDoctorSelfTestFailureFails(t *testing.T) {
	stubDoctor(t, &fakeEngine{ep: "cpu", err: errors.New("decode: boom")}, gpuDown)
	var out bytes.Buffer
	err := runDoctor([]string{"--model", fixtureModel(t), "--ort", fixtureORT(t)}, &out, io.Discard)
	if err == nil {
		t.Fatal("want error when the self-test fails")
	}
	if !strings.Contains(out.String(), "FAIL  self-test") {
		t.Errorf("report lacks self-test FAIL:\n%s", out.String())
	}
}
