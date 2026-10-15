package ortengine

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/golden"
)

var _ asr.Engine = (*Engine)(nil)

func testEngine(t *testing.T) *Engine {
	t.Helper()
	lib, dir := os.Getenv("RAMBLOCAL_ORT_LIB"), os.Getenv("RAMBLOCAL_MODEL_DIR")
	if lib == "" || dir == "" {
		t.Skip("RAMBLOCAL_ORT_LIB / RAMBLOCAL_MODEL_DIR not set")
	}
	cfg := Config{ModelDir: dir, ORTLib: lib}
	if ep := os.Getenv("RAMBLOCAL_EP"); ep != "" {
		cfg.EPs = []string{ep}
	}
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() })
	t.Logf("engine %s", e.Name())
	return e
}

func goldenDir() string { return filepath.Join("..", "..", "..", "testdata", "golden") }

func TestGoldenTokens(t *testing.T) {
	e := testEngine(t)
	names, _ := golden.List(goldenDir())
	for _, n := range names {
		t.Run(n, func(t *testing.T) {
			fx, err := golden.Load(goldenDir(), n)
			if err != nil {
				t.Fatal(err)
			}
			res, err := e.Transcribe(context.Background(), fx.PCM, asr.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(res.Tokens, fx.Tokens) {
				t.Fatalf("tokens differ\n got  %q\n want %q", res.Text, fx.Text)
			}
			if res.Text != fx.Text {
				t.Errorf("text %q, want %q", res.Text, fx.Text)
			}
			if len(res.Words) == 0 || res.Words[len(res.Words)-1].End > res.AudioSec+0.1 {
				t.Errorf("bad word timestamps: %+v (audio %.2fs)", res.Words, res.AudioSec)
			}
		})
	}
}

func TestTranscribeEdgeAudio(t *testing.T) {
	e := testEngine(t)
	ctx := context.Background()

	for name, pcm := range map[string][]float32{
		"empty":   nil,
		"50ms":    make([]float32, 800),
		"silence": make([]float32, 2*asr.SampleRate),
	} {
		res, err := e.Transcribe(ctx, pcm, asr.Options{})
		if err != nil {
			t.Errorf("%s: %v", name, err)
		}
		if len(strings.Fields(res.Text)) > 2 {
			t.Errorf("%s: hallucinated %q", name, res.Text)
		}
	}

	// 40 s: concatenate golden clips until long enough.
	names, _ := golden.List(goldenDir())
	var long []float32
	for len(long) < 40*asr.SampleRate {
		for _, n := range names {
			fx, err := golden.Load(goldenDir(), n)
			if err != nil {
				t.Fatal(err)
			}
			long = append(long, fx.PCM...)
		}
	}
	res, err := e.Transcribe(ctx, long, asr.Options{})
	if err != nil || len(strings.Fields(res.Text)) < 40 {
		t.Fatalf("40s clip: %d words, err %v", len(strings.Fields(res.Text)), err)
	}
}

func TestCancelledContext(t *testing.T) {
	e := testEngine(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.Transcribe(ctx, make([]float32, asr.SampleRate), asr.Options{}); err == nil {
		t.Fatal("expected error from cancelled context")
	}
}
