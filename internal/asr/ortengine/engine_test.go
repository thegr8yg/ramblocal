package ortengine

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/golden"
	"github.com/thegr8yg/ramblocal/internal/eval"
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

// goldenNames lists the golden fixtures and fails the test if there are none,
// so that loops over them cannot pass vacuously.
func goldenNames(t *testing.T) []string {
	t.Helper()
	names, err := golden.List(goldenDir())
	if err != nil {
		t.Fatal(err)
	}
	if len(names) == 0 {
		t.Fatal("no golden fixtures found")
	}
	return names
}

// testCtx bounds every integration call so a hung model fails instead of blocking the run.
func testCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	t.Cleanup(cancel)
	return ctx
}

func TestGoldenTokens(t *testing.T) {
	e := testEngine(t)
	names := goldenNames(t)
	for _, n := range names {
		t.Run(n, func(t *testing.T) {
			fx, err := golden.Load(goldenDir(), n)
			if err != nil {
				t.Fatal(err)
			}
			res, err := e.Transcribe(testCtx(t), fx.PCM, asr.Options{})
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
	ctx := testCtx(t)

	for name, pcm := range map[string][]float32{
		"empty":        nil,
		"50ms":         make([]float32, 800),
		"minSamples-1": make([]float32, MinSamples-1),
		"minSamples":   make([]float32, MinSamples),
		"500ms":        make([]float32, asr.SampleRate/2),
		"silence":      make([]float32, 2*asr.SampleRate),
	} {
		res, err := e.Transcribe(ctx, pcm, asr.Options{})
		if err != nil {
			t.Errorf("%s: %v", name, err)
		}
		if len(strings.Fields(res.Text)) > 2 {
			t.Errorf("%s: hallucinated %q", name, res.Text)
		}
	}

	// 40 s: concatenate golden clips until long enough; the reference is built
	// from the golden texts in the same order and repetition.
	names := goldenNames(t)
	var long []float32
	var refs []string
	for len(long) < 40*asr.SampleRate {
		for _, n := range names {
			fx, err := golden.Load(goldenDir(), n)
			if err != nil {
				t.Fatal(err)
			}
			long = append(long, fx.PCM...)
			refs = append(refs, fx.Text)
		}
	}
	res, err := e.Transcribe(ctx, long, asr.Options{})
	if err != nil {
		t.Fatalf("40s clip: %v", err)
	}
	if len(res.Words) == 0 {
		t.Fatalf("40s clip: no words, text %q", res.Text)
	}
	if last := res.Words[len(res.Words)-1]; last.End > res.AudioSec+0.1 {
		t.Errorf("40s clip: last word %q ends at %.2fs, audio %.2fs", last.Text, last.End, res.AudioSec)
	}
	var c eval.Corpus
	c.Add(strings.Join(refs, " "), res.Text)
	if w := c.WER(); w > 5.0 {
		t.Errorf("40s clip: WER %.2f%% > 5%%\n got  %q", w, res.Text)
	}
}

func TestConcurrentTranscribe(t *testing.T) {
	e := testEngine(t)
	ctx := testCtx(t)
	names := goldenNames(t)

	var wg sync.WaitGroup
	for i := range 4 {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			fx, err := golden.Load(goldenDir(), name)
			if err != nil {
				t.Errorf("%s: load: %v", name, err)
				return
			}
			for range 2 {
				res, err := e.Transcribe(ctx, fx.PCM, asr.Options{})
				if err != nil {
					t.Errorf("%s: %v", name, err)
					return
				}
				if !slices.Equal(res.Tokens, fx.Tokens) {
					t.Errorf("%s: tokens differ\n got  %q\n want %q", name, res.Text, fx.Text)
				}
			}
		}(names[i%len(names)])
	}
	wg.Wait()
}

func TestCancelledContext(t *testing.T) {
	e := testEngine(t)
	ctx, cancel := context.WithCancel(testCtx(t))
	cancel()
	if _, err := e.Transcribe(ctx, make([]float32, asr.SampleRate), asr.Options{}); err == nil {
		t.Fatal("expected error from cancelled context")
	}
}
