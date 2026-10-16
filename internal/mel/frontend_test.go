package mel

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/golden"
)

// Tiny synthetic frontend: n_fft 8, win 6, hop 2, 3 mels (5 bins).
var (
	tinyFB = []float32{
		1, 0.5, 0, 0, 0,
		0, 0.5, 1, 0.5, 0,
		0, 0, 0, 0.5, 1,
	}
	tinyWin = []float32{0.1, 0.4, 0.9, 0.9, 0.4, 0.1}
)

func tinyManifest() *asr.Manifest {
	return &asr.Manifest{
		NumFeatures: 3,
		Frontend: &asr.Frontend{
			NFFT: 8, WinLength: 6, HopLength: 2, Preemph: 0.97,
			LogGuard: 1e-7, NormEps: 1e-5, Filterbank: "fb.f32", Window: "w.f32",
		},
	}
}

func writeF32(t *testing.T, path string, v []float32) {
	t.Helper()
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func tinyFrontend(t *testing.T) *Frontend {
	t.Helper()
	dir := t.TempDir()
	writeF32(t, filepath.Join(dir, "fb.f32"), tinyFB)
	writeF32(t, filepath.Join(dir, "w.f32"), tinyWin)
	f, err := New(dir, tinyManifest())
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// naiveCompute is written independently of the production code: direct DFT, no FFT,
// explicit zero padding of the window.
func naiveCompute(pcm []float32, fb, win []float32, nfft, hop, nmel int, preemph, guard, eps float64) ([]float32, int, int) {
	n := len(pcm)
	valid := n / hop
	width := 1 + n/hop
	y := make([]float64, n)
	for i := range y {
		y[i] = float64(pcm[i])
		if i > 0 {
			y[i] -= preemph * float64(pcm[i-1])
		}
	}
	padded := make([]float64, n+nfft)
	copy(padded[nfft/2:], y)
	fullWin := make([]float64, nfft)
	off := (nfft - len(win)) / 2
	for i, w := range win {
		fullWin[off+i] = float64(w)
	}
	bins := nfft/2 + 1
	mel := make([][]float64, nmel)
	for m := range mel {
		mel[m] = make([]float64, width)
	}
	for tt := 0; tt < width; tt++ {
		power := make([]float64, bins)
		for k := 0; k < bins; k++ {
			var re, im float64
			for j := 0; j < nfft; j++ {
				v := padded[tt*hop+j] * fullWin[j]
				a := -2 * math.Pi * float64(k*j) / float64(nfft)
				re += v * math.Cos(a)
				im += v * math.Sin(a)
			}
			power[k] = re*re + im*im
		}
		for m := 0; m < nmel; m++ {
			var s float64
			for k := 0; k < bins; k++ {
				s += float64(fb[m*bins+k]) * power[k]
			}
			mel[m][tt] = math.Log(s + guard)
		}
	}
	out := make([]float32, nmel*width)
	for m := 0; m < nmel; m++ {
		var mean float64
		for tt := 0; tt < valid; tt++ {
			mean += mel[m][tt]
		}
		mean /= float64(valid)
		var ss float64
		for tt := 0; tt < valid; tt++ {
			d := mel[m][tt] - mean
			ss += d * d
		}
		std := math.Sqrt(ss/float64(valid-1)) + eps
		for tt := 0; tt < valid; tt++ {
			out[m*width+tt] = float32((mel[m][tt] - mean) / std)
		}
	}
	return out, width, valid
}

func randPCM(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	pcm := make([]float32, n)
	for i := range pcm {
		pcm[i] = float32(rng.Float64()*2 - 1)
	}
	return pcm
}

func TestComputeMatchesNaive(t *testing.T) {
	f := tinyFrontend(t)
	for _, n := range []int{4, 5, 17, 100, 1000} {
		pcm := randPCM(n, int64(n))
		got, width, valid := f.Compute(pcm)
		want, wantW, wantV := naiveCompute(pcm, tinyFB, tinyWin, 8, 2, 3, 0.97, 1e-7, 1e-5)
		if width != wantW || valid != wantV {
			t.Fatalf("n=%d: width/valid = %d/%d, want %d/%d", n, width, valid, wantW, wantV)
		}
		if width != valid+1 || valid != n/2 {
			t.Fatalf("n=%d: width %d valid %d inconsistent", n, width, valid)
		}
		if len(got) != 3*width {
			t.Fatalf("n=%d: len %d want %d", n, len(got), 3*width)
		}
		if d := golden.MaxAbsDiff(got, want); d > 1e-4 {
			t.Fatalf("n=%d: max diff vs naive %g", n, d)
		}
		for m := 0; m < 3; m++ {
			if got[m*width+valid] != 0 {
				t.Fatalf("n=%d: row %d last column = %v, want exactly 0", n, m, got[m*width+valid])
			}
		}
	}
}

func TestComputeNormalization(t *testing.T) {
	f := tinyFrontend(t)
	pcm := randPCM(16000, 7)
	got, width, valid := f.Compute(pcm)
	if valid != 8000 || width != 8001 {
		t.Fatalf("valid/width = %d/%d, want 8000/8001", valid, width)
	}
	for m := 0; m < 3; m++ {
		row := got[m*width : m*width+valid]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(valid)
		var ss float64
		for _, v := range row {
			ss += (float64(v) - mean) * (float64(v) - mean)
		}
		std := math.Sqrt(ss / float64(valid-1))
		if math.Abs(mean) > 1e-4 || math.Abs(std-1) > 1e-3 {
			t.Errorf("row %d: mean %g std %g", m, mean, std)
		}
	}
}

func TestComputeHundredFrames(t *testing.T) {
	// The brief's example uses the production hop (160): 16000 samples -> valid 100, width 101.
	fb := make([]float32, 2*(512/2+1))
	for i := range fb {
		fb[i] = 1 / 257.0
	}
	dir := t.TempDir()
	writeF32(t, filepath.Join(dir, "fb.f32"), fb)
	writeF32(t, filepath.Join(dir, "w.f32"), make([]float32, 400))
	m := &asr.Manifest{NumFeatures: 2, Frontend: &asr.Frontend{NFFT: 512, WinLength: 400, HopLength: 160,
		Preemph: 0.97, LogGuard: 1e-7, NormEps: 1e-5, Filterbank: "fb.f32", Window: "w.f32"}}
	f, err := New(dir, m)
	if err != nil {
		t.Fatal(err)
	}
	feats, width, valid := f.Compute(randPCM(16000, 3))
	if valid != 100 || width != 101 || len(feats) != 2*101 {
		t.Fatalf("valid/width/len = %d/%d/%d, want 100/101/202", valid, width, len(feats))
	}
}

func TestComputeShortInput(t *testing.T) {
	f := tinyFrontend(t)
	// hop 2: valid < 2 means fewer than 4 samples.
	for _, n := range []int{0, 1, 2, 3} {
		feats, width, valid := f.Compute(randPCM(n, 1))
		if feats != nil || width != 0 || valid != 0 {
			t.Errorf("n=%d: got (%v, %d, %d), want (nil, 0, 0)", n, feats, width, valid)
		}
	}
	if feats, _, valid := f.Compute(randPCM(4, 1)); feats == nil || valid != 2 {
		t.Errorf("n=4: valid %d, want 2", valid)
	}
}

func TestNewErrors(t *testing.T) {
	if _, err := New(t.TempDir(), &asr.Manifest{NumFeatures: 3}); err == nil {
		t.Error("nil Frontend: expected error")
	}
	for name, c := range map[string]struct{ fb, win int }{
		"short filterbank": {14, 6},
		"long filterbank":  {16, 6},
		"short window":     {15, 5},
		"long window":      {15, 7},
	} {
		dir := t.TempDir()
		writeF32(t, filepath.Join(dir, "fb.f32"), make([]float32, c.fb))
		writeF32(t, filepath.Join(dir, "w.f32"), make([]float32, c.win))
		if _, err := New(dir, tinyManifest()); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	if _, err := New(t.TempDir(), tinyManifest()); err == nil {
		t.Error("missing files: expected error")
	}
}

const melTol = 2e-4

func TestGoldenMel(t *testing.T) {
	dir := os.Getenv("RAMBLOCAL_MODEL_DIR")
	if dir == "" {
		t.Skip("RAMBLOCAL_MODEL_DIR not set")
	}
	m, err := asr.LoadManifest(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	f, err := New(dir, m)
	if err != nil {
		t.Fatal(err)
	}
	gdir := filepath.Join("..", "..", "testdata", "golden")
	names, err := golden.List(gdir)
	if err != nil || len(names) != 5 {
		t.Fatalf("golden fixtures: %v, %v", names, err)
	}
	for _, n := range names {
		t.Run(n, func(t *testing.T) {
			fx, err := golden.Load(gdir, n)
			if err != nil {
				t.Fatal(err)
			}
			feats, width, valid := f.Compute(fx.PCM)
			if valid != fx.MelShape[1] {
				t.Fatalf("valid %d, golden %d", valid, fx.MelShape[1])
			}
			if width != valid+1 {
				t.Fatalf("width %d, want %d", width, valid+1)
			}
			rows := fx.MelShape[0]
			var maxd float64
			for r := 0; r < rows; r++ {
				if v := feats[r*width+valid]; v != 0 {
					t.Fatalf("row %d last column = %v, want exactly 0", r, v)
				}
				for c := 0; c < valid; c++ {
					d := math.Abs(float64(feats[r*width+c]) - float64(fx.Mel[r*valid+c]))
					if math.IsNaN(d) {
						d = math.Inf(1)
					}
					maxd = math.Max(maxd, d)
				}
			}
			t.Logf("mel max abs diff %g (valid %d)", maxd, valid)
			if maxd > melTol {
				t.Errorf("mel max abs diff %g > %g", maxd, melTol)
			}
		})
	}
}
