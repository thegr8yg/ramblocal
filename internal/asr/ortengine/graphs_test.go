package ortengine

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/golden"
)

// Tolerances are fixed here; loosening them requires orchestrator approval with evidence.
const (
	melTol = 1e-3
	encTol = 5e-3
)

func testGraphs(t *testing.T) (*graphs, string) {
	t.Helper()
	lib, dir := os.Getenv("RAMBLOCAL_ORT_LIB"), os.Getenv("RAMBLOCAL_MODEL_DIR")
	if lib == "" || dir == "" {
		t.Skip("RAMBLOCAL_ORT_LIB / RAMBLOCAL_MODEL_DIR not set")
	}
	if err := InitRuntime(lib); err != nil {
		t.Fatal(err)
	}
	m, err := asr.LoadManifest(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	ep := os.Getenv("RAMBLOCAL_EP")
	if ep == "" {
		ep = "cuda"
	}
	g, err := openGraphs(dir, m, ep)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.close)
	return g, filepath.Join("..", "..", "..", "testdata", "golden")
}

func TestGoldenMelAndEncoder(t *testing.T) {
	g, gdir := testGraphs(t)
	names, err := golden.List(gdir)
	if err != nil || len(names) == 0 {
		t.Fatalf("no golden fixtures in %s: %v", gdir, err)
	}
	for _, n := range names {
		t.Run(n, func(t *testing.T) {
			fx, err := golden.Load(gdir, n)
			if err != nil {
				t.Fatal(err)
			}
			feats, frames, err := g.preprocess(fx.PCM)
			if err != nil {
				t.Fatal(err)
			}
			if frames != fx.MelShape[1] {
				t.Fatalf("mel frames %d, golden %d", frames, fx.MelShape[1])
			}
			if d := golden.MaxAbsDiff(feats, fx.Mel); d > melTol {
				t.Errorf("mel max abs diff %g > %g", d, melTol)
			}
			// Encoder is checked on the golden mel so its error is not compounded by preprocessing.
			enc, outFrames, err := g.encode(fx.Mel, fx.MelShape[1])
			if err != nil {
				t.Fatal(err)
			}
			if outFrames != fx.EncShape[1] {
				t.Fatalf("encoder frames %d, golden %d", outFrames, fx.EncShape[1])
			}
			if d := golden.MaxAbsDiff(enc, fx.Enc); d > encTol {
				t.Errorf("encoder max abs diff %g > %g", d, encTol)
			}
		})
	}
}
