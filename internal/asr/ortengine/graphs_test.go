package ortengine

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/golden"
	"github.com/thegr8yg/ramblocal/internal/mel"
)

// Tolerances are fixed here; loosening them requires orchestrator approval with evidence.
const (
	melTol = 2e-4
	encTol = 5e-3
)

func testGraphs(t *testing.T) (*graphs, *asr.Manifest, string, string) {
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
	return g, m, dir, filepath.Join("..", "..", "..", "testdata", "golden")
}

func TestGoldenMelAndEncoder(t *testing.T) {
	g, m, dir, gdir := testGraphs(t)
	if m.Frontend == nil {
		t.Skip("manifest has no frontend section")
	}
	if g.pre != nil {
		t.Error("preprocessor graph opened although the manifest has a frontend section")
	}
	fe, err := mel.New(dir, m)
	if err != nil {
		t.Fatal(err)
	}
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
			rows, valid := fx.MelShape[0], fx.MelShape[1]

			// (a) Go frontend vs NeMo mel.
			feats, width, gotValid := fe.Compute(fx.PCM)
			if gotValid != valid {
				t.Fatalf("mel valid %d, golden %d", gotValid, valid)
			}
			if width != valid+1 {
				t.Fatalf("mel width %d, want %d", width, valid+1)
			}
			var melDiff float64
			for r := 0; r < rows; r++ {
				if v := feats[r*width+valid]; v != 0 {
					t.Fatalf("row %d last column = %v, want exactly 0", r, v)
				}
				d := golden.MaxAbsDiff(feats[r*width:r*width+valid], fx.Mel[r*valid:(r+1)*valid])
				melDiff = max(melDiff, float64(d))
			}
			t.Logf("mel max abs diff %g", melDiff)
			if melDiff > melTol {
				t.Errorf("mel max abs diff %g > %g", melDiff, melTol)
			}

			// (b) Encoder fed the golden mel plus one zero column, length = valid.
			padded := make([]float32, rows*(valid+1))
			for r := 0; r < rows; r++ {
				copy(padded[r*(valid+1):], fx.Mel[r*valid:(r+1)*valid])
			}
			enc, outFrames, err := g.encode(padded, valid+1, valid)
			if err != nil {
				t.Fatal(err)
			}
			if outFrames != fx.EncShape[1] {
				t.Fatalf("encoder frames %d, golden %d", outFrames, fx.EncShape[1])
			}
			encDiff := golden.MaxAbsDiff(enc, fx.Enc)
			t.Logf("encoder max abs diff %g", encDiff)
			if encDiff > encTol {
				t.Errorf("encoder max abs diff %g > %g", encDiff, encTol)
			}
		})
	}
}
