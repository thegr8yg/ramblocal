package mel

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

func TestFFTMatchesNaiveDFT(t *testing.T) {
	const n = 64
	rng := rand.New(rand.NewSource(1))
	re := make([]float64, n)
	im := make([]float64, n)
	x := make([]complex128, n)
	for i := range re {
		re[i] = rng.NormFloat64()
		x[i] = complex(re[i], 0)
	}
	if err := fft(re, im); err != nil {
		t.Fatal(err)
	}
	for k := 0; k < n; k++ {
		var want complex128
		for j := 0; j < n; j++ {
			want += x[j] * cmplx.Exp(complex(0, -2*math.Pi*float64(j*k)/n))
		}
		got := complex(re[k], im[k])
		if cmplx.Abs(got-want) > 1e-9 {
			t.Fatalf("bin %d: got %v want %v", k, got, want)
		}
	}
}

func TestFFTRejectsNonPowerOfTwo(t *testing.T) {
	for _, n := range []int{0, 3, 6, 100} {
		if err := fft(make([]float64, n), make([]float64, n)); err == nil {
			t.Errorf("n=%d: expected error", n)
		}
	}
	if err := fft(make([]float64, 8), make([]float64, 4)); err == nil {
		t.Error("mismatched lengths: expected error")
	}
}
