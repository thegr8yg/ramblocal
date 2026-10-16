package mel

import (
	"math"
	"math/rand"
	"testing"
)

// TestRealFFTPowerMatchesComplexFFT checks the N/2-point real-FFT trick against the
// full complex FFT of the same real input.
func TestRealFFTPowerMatchesComplexFFT(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for _, n := range []int{2, 4, 8, 16, 512, 1024} {
		rp, err := newRealFFTPlan(n)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		full, _ := newFFTPlan(n)
		x := make([]float64, n)
		for trial := 0; trial < 20; trial++ {
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			re := append([]float64(nil), x...)
			im := make([]float64, n)
			full.transform(re, im)

			zr := make([]float64, n/2)
			zi := make([]float64, n/2)
			power := make([]float64, n/2+1)
			rp.loadReal(x, zr, zi)
			rp.power(zr, zi, power)
			for k := 0; k <= n/2; k++ {
				want := re[k]*re[k] + im[k]*im[k]
				if d := math.Abs(power[k] - want); d > 1e-9*(1+want) {
					t.Fatalf("n=%d bin %d: real-FFT power %g, complex %g", n, k, power[k], want)
				}
			}
		}
	}
}

func TestRealFFTPlanRejectsBadSizes(t *testing.T) {
	for _, n := range []int{0, 1, 3, 6, 100} {
		if _, err := newRealFFTPlan(n); err == nil {
			t.Errorf("n=%d: expected error", n)
		}
	}
}
