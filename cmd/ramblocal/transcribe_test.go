package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/ortengine"
	"github.com/thegr8yg/ramblocal/internal/audio"
)

// fixtures writes a WAV at the given sample rate plus a fake model dir and ORT library, so
// path resolution succeeds without a real model. It returns the WAV path, model dir and library path.
func fixtures(t *testing.T, rate int) (string, string, string) {
	t.Helper()
	dir := t.TempDir()

	wav := filepath.Join(dir, "clip.wav")
	f, err := os.Create(wav)
	if err != nil {
		t.Fatal(err)
	}
	if err := audio.WriteWAV16(f, make([]float32, 1600), rate); err != nil {
		t.Fatal(err)
	}
	f.Close()

	model := filepath.Join(dir, "model")
	if err := os.Mkdir(model, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(model, "manifest.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	lib := filepath.Join(dir, "libonnxruntime.so")
	if err := os.WriteFile(lib, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	return wav, model, lib
}

// stubEngine replaces newEngine for the duration of the test and returns a pointer to the
// config the command passed in.
func stubEngine(t *testing.T, eng asr.Engine) *ortengine.Config {
	t.Helper()
	var got ortengine.Config
	old := newEngine
	newEngine = func(cfg ortengine.Config) (asr.Engine, error) {
		got = cfg
		return eng, nil
	}
	t.Cleanup(func() { newEngine = old })
	return &got
}

func TestTranscribeRequiresOneFile(t *testing.T) {
	var out, errOut bytes.Buffer
	if err := runTranscribe(nil, &out, &errOut); err == nil {
		t.Fatal("expected error with no file argument")
	}
	if err := runTranscribe([]string{"a.wav", "b.wav"}, &out, &errOut); err == nil {
		t.Fatal("expected error with two file arguments")
	}
}

func TestTranscribeRejectsWrongSampleRate(t *testing.T) {
	stubEngine(t, &asr.FakeEngine{})
	wav, model, lib := fixtures(t, 8000)
	var out, errOut bytes.Buffer
	err := runTranscribe([]string{"--model", model, "--ort", lib, wav}, &out, &errOut)
	if err == nil {
		t.Fatal("expected sample rate error")
	}
	if !strings.Contains(err.Error(), "8000") || !strings.Contains(err.Error(), "16000") {
		t.Fatalf("error should name the actual and expected rate, got: %v", err)
	}
	if out.Len() != 0 {
		t.Fatalf("no output expected on error, got %q", out.String())
	}
}

func TestTranscribePlainOutput(t *testing.T) {
	var gotLen int
	stubEngine(t, &asr.FakeEngine{Fn: func(pcm []float32, _ asr.Options) (asr.Result, error) {
		gotLen = len(pcm)
		return asr.Result{Text: "hello world", AudioSec: 0.1}, nil
	}})
	wav, model, lib := fixtures(t, asr.SampleRate)
	var out, errOut bytes.Buffer
	if err := runTranscribe([]string{"--model", model, "--ort", lib, wav}, &out, &errOut); err != nil {
		t.Fatal(err)
	}
	if out.String() != "hello world\n" {
		t.Fatalf("stdout = %q, want %q", out.String(), "hello world\n")
	}
	if gotLen != 1600 {
		t.Fatalf("engine got %d samples, want 1600", gotLen)
	}
}

func TestTranscribeJSONShape(t *testing.T) {
	stubEngine(t, &asr.FakeEngine{Fn: func(pcm []float32, _ asr.Options) (asr.Result, error) {
		return asr.Result{
			Text:     "hi there",
			AudioSec: 0.1,
			Words: []asr.Word{
				{Text: "hi", Start: 0.08, End: 0.32},
				{Text: "there", Start: 0.40, End: 0.72},
			},
		}, nil
	}})
	wav, model, lib := fixtures(t, asr.SampleRate)
	var out, errOut bytes.Buffer
	if err := runTranscribe([]string{"--json", "--model", model, "--ort", lib, wav}, &out, &errOut); err != nil {
		t.Fatal(err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out.Bytes(), &raw); err != nil {
		t.Fatalf("invalid JSON %q: %v", out.String(), err)
	}
	var keys []string
	for k := range raw {
		keys = append(keys, k)
	}
	if len(keys) != 4 {
		t.Fatalf("keys = %v, want exactly text, audio_sec, ep, words", keys)
	}

	var got struct {
		Text     string  `json:"text"`
		AudioSec float64 `json:"audio_sec"`
		EP       string  `json:"ep"`
		Words    []struct {
			Text  string  `json:"text"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"words"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Text != "hi there" || got.AudioSec != 0.1 {
		t.Fatalf("got text %q audio_sec %v", got.Text, got.AudioSec)
	}
	if len(got.Words) != 2 || got.Words[0].Text != "hi" || got.Words[1].End != 0.72 {
		t.Fatalf("words = %+v", got.Words)
	}
}

func TestTranscribeJSONEmptyWordsIsArray(t *testing.T) {
	stubEngine(t, &asr.FakeEngine{})
	wav, model, lib := fixtures(t, asr.SampleRate)
	var out, errOut bytes.Buffer
	if err := runTranscribe([]string{"--json", "--model", model, "--ort", lib, wav}, &out, &errOut); err != nil {
		t.Fatal(err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	if string(raw["words"]) != "[]" {
		t.Fatalf("words = %s, want []", raw["words"])
	}
}

func TestTranscribeEngineErrorPropagates(t *testing.T) {
	boom := errors.New("boom")
	stubEngine(t, &asr.FakeEngine{Fn: func([]float32, asr.Options) (asr.Result, error) {
		return asr.Result{}, boom
	}})
	wav, model, lib := fixtures(t, asr.SampleRate)
	var out, errOut bytes.Buffer
	err := runTranscribe([]string{"--model", model, "--ort", lib, wav}, &out, &errOut)
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want it to wrap boom", err)
	}
	if out.Len() != 0 {
		t.Fatalf("no stdout expected on engine error, got %q", out.String())
	}
}

func TestTranscribeEPFlagAndPaths(t *testing.T) {
	got := stubEngine(t, &asr.FakeEngine{})
	wav, model, lib := fixtures(t, asr.SampleRate)
	var out, errOut bytes.Buffer
	if err := runTranscribe([]string{"--ep", "cpu", "--model", model, "--ort", lib, wav}, &out, &errOut); err != nil {
		t.Fatal(err)
	}
	want := ortengine.Config{ModelDir: model, ORTLib: lib, EPs: []string{"cpu"}}
	if !reflect.DeepEqual(*got, want) {
		t.Fatalf("engine config = %+v, want %+v", *got, want)
	}
}

func TestTranscribeEPDefaultsToFallback(t *testing.T) {
	got := stubEngine(t, &asr.FakeEngine{})
	wav, model, lib := fixtures(t, asr.SampleRate)
	var out, errOut bytes.Buffer
	if err := runTranscribe([]string{"--model", model, "--ort", lib, wav}, &out, &errOut); err != nil {
		t.Fatal(err)
	}
	if len(got.EPs) != 0 {
		t.Fatalf("EPs = %v, want empty so ortengine falls back", got.EPs)
	}
}

func TestTranscribeTimingsGoToStderr(t *testing.T) {
	stubEngine(t, &asr.FakeEngine{Fn: func([]float32, asr.Options) (asr.Result, error) {
		return asr.Result{Text: "x"}, nil
	}})
	wav, model, lib := fixtures(t, asr.SampleRate)
	var out, errOut bytes.Buffer
	if err := runTranscribe([]string{"--timings", "--model", model, "--ort", lib, wav}, &out, &errOut); err != nil {
		t.Fatal(err)
	}
	if out.String() != "x\n" {
		t.Fatalf("stdout = %q, timings must not leak into it", out.String())
	}
	if !strings.Contains(errOut.String(), "encoder") {
		t.Fatalf("stderr missing timings: %q", errOut.String())
	}
}
