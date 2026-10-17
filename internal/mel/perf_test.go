package mel

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/asr"
)

// realFrontend loads the production frontend from RAMBLOCAL_MODEL_DIR, or skips.
func realFrontend(tb testing.TB) *Frontend {
	tb.Helper()
	dir := os.Getenv("RAMBLOCAL_MODEL_DIR")
	if dir == "" {
		tb.Skip("RAMBLOCAL_MODEL_DIR not set")
	}
	m, err := asr.LoadManifest(filepath.Join(dir, "manifest.json"))
	if err != nil {
		tb.Fatal(err)
	}
	f, err := New(dir, m)
	if err != nil {
		tb.Fatal(err)
	}
	return f
}

// noise returns deterministic pseudo-random audio in [-0.5, 0.5].
func noise(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	pcm := make([]float32, n)
	for i := range pcm {
		pcm[i] = float32(rng.Float64() - 0.5)
	}
	return pcm
}

// syntheticFB returns a production-shaped filterbank (128 mels x 257 bins) banded
// like Slaney's: each row is non-zero on a short random span, zero elsewhere, with
// some interior zeros inside the span.
func syntheticFB() []float32 {
	const nmel, bins = 128, 257
	rng := rand.New(rand.NewSource(11))
	fb := make([]float32, nmel*bins)
	for m := 0; m < nmel; m++ {
		start := rng.Intn(bins - 8)
		end := start + 2 + rng.Intn(min(40, bins-start-2))
		for k := start; k < end; k++ {
			if rng.Intn(9) != 0 {
				fb[m*bins+k] = float32(rng.Float64() * 0.05)
			}
		}
	}
	return fb
}

// syntheticFrontend builds a production-shaped frontend (n_fft 512, win 400, hop 160, 128 mels).
func syntheticFrontend(t *testing.T) *Frontend {
	t.Helper()
	const nfft, win, hop, nmel = 512, 400, 160, 128
	w := make([]float32, win)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(win)))
	}
	dir := t.TempDir()
	writeF32(t, filepath.Join(dir, "fb.f32"), syntheticFB())
	writeF32(t, filepath.Join(dir, "w.f32"), w)
	m := &asr.Manifest{NumFeatures: nmel, Frontend: &asr.Frontend{NFFT: nfft, WinLength: win, HopLength: hop,
		Preemph: 0.97, LogGuard: 1e-7, NormEps: 1e-5, Filterbank: "fb.f32", Window: "w.f32"}}
	f, err := New(dir, m)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

type result struct {
	feats        []float32
	width, valid int
}

func compute(f *Frontend, pcm []float32) result {
	feats, w, v := f.Compute(pcm)
	return result{feats, w, v}
}

func equalBits(a, b result) bool {
	if a.width != b.width || a.valid != b.valid || len(a.feats) != len(b.feats) {
		return false
	}
	for i := range a.feats {
		if math.Float32bits(a.feats[i]) != math.Float32bits(b.feats[i]) {
			return false
		}
	}
	return true
}

// withProcs runs fn with GOMAXPROCS set to n, restoring it afterwards.
func withProcs(n int, fn func()) {
	old := runtime.GOMAXPROCS(n)
	defer runtime.GOMAXPROCS(old)
	fn()
}

// checkDeterministic asserts Compute is bit-identical between 1 and 8 procs
// (hence between 1 and N workers) and between consecutive calls, and that
// every call returns freshly allocated feats.
func checkDeterministic(t *testing.T, f *Frontend, pcm []float32) {
	t.Helper()
	var one, many, again result
	withProcs(1, func() { one = compute(f, pcm) })
	withProcs(8, func() { many = compute(f, pcm); again = compute(f, pcm) })
	if one.feats == nil {
		t.Fatal("no output")
	}
	if !equalBits(one, many) {
		t.Error("output differs between 1 and 8 procs")
	}
	if !equalBits(many, again) {
		t.Error("output differs between consecutive calls")
	}
	if &one.feats[0] == &many.feats[0] || &many.feats[0] == &again.feats[0] {
		t.Error("Compute returned an aliased feats slice")
	}
}

func TestComputeDeterministicSynthetic(t *testing.T) {
	checkDeterministic(t, tinyFrontend(t), noise(4000, 1))         // 2000 frames, tiny dims
	checkDeterministic(t, syntheticFrontend(t), noise(16000*5, 2)) // production dims
	checkDeterministic(t, syntheticFrontend(t), noise(160*10, 3))  // below the parallel threshold
}

func TestComputeDeterministicReal30s(t *testing.T) {
	checkDeterministic(t, realFrontend(t), noise(16000*30, 4))
}

func TestComputeConcurrent(t *testing.T) {
	for name, f := range map[string]*Frontend{"synthetic": syntheticFrontend(t), "tiny": tinyFrontend(t)} {
		t.Run(name, func(t *testing.T) {
			const g = 8
			inputs := make([][]float32, g)
			want := make([]result, g)
			for i := range inputs {
				inputs[i] = noise(16000+i*3777, int64(100+i))
				withProcs(1, func() { want[i] = compute(f, inputs[i]) })
			}
			got := make([]result, g)
			var wg sync.WaitGroup
			for i := range inputs {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for rep := 0; rep < 3; rep++ {
						got[i] = compute(f, inputs[i])
					}
				}()
			}
			wg.Wait()
			for i := range got {
				if !equalBits(got[i], want[i]) {
					t.Errorf("goroutine %d: result differs from single-threaded", i)
				}
			}
		})
	}
}

func BenchmarkCompute10s(b *testing.B) {
	f := realFrontend(b)
	pcm := noise(16000*10, 42)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if feats, _, _ := f.Compute(pcm); feats == nil {
			b.Fatal("nil feats")
		}
	}
}
