package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/asr/golden"
)

// TestTranscribeGolden runs the real engine through the CLI on the NeMo golden clips.
// It skips unless RAMBLOCAL_MODEL_DIR and RAMBLOCAL_ORT_LIB are set.
func TestTranscribeGolden(t *testing.T) {
	model, lib := os.Getenv("RAMBLOCAL_MODEL_DIR"), os.Getenv("RAMBLOCAL_ORT_LIB")
	if model == "" || lib == "" {
		t.Skip("set RAMBLOCAL_MODEL_DIR and RAMBLOCAL_ORT_LIB to run against a real model")
	}
	dir := filepath.Join("..", "..", "testdata", "golden")
	names, err := golden.List(dir)
	if err != nil || len(names) == 0 {
		t.Fatalf("no golden fixtures in %s: %v", dir, err)
	}
	for _, name := range names {
		fx, err := golden.Load(dir, name)
		if err != nil {
			t.Fatal(err)
		}
		var out, errb bytes.Buffer
		err = run([]string{"transcribe", "--model", model, "--ort", lib, filepath.Join(dir, name+".wav")}, &out, &errb)
		if err != nil {
			t.Fatalf("%s: %v\n%s", name, err, errb.String())
		}
		if got := strings.TrimSpace(out.String()); got != fx.Text {
			t.Errorf("%s: got %q, want %q", name, got, fx.Text)
		}
	}
}
