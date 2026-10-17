package mel

import (
	"math"
	"math/rand"
	"testing"
)

// denseMel is the reference dense filterbank product.
func denseMel(fb, power []float64, nMel, bins int) []float64 {
	out := make([]float64, nMel)
	for m := range out {
		var s float64
		for k, w := range fb[m*bins : (m+1)*bins] {
			s += w * power[k]
		}
		out[m] = s
	}
	return out
}

func checkBandedEqualsDense(t *testing.T, f *Frontend) {
	t.Helper()
	rng := rand.New(rand.NewSource(5))
	power := make([]float64, f.bins)
	got := make([]float64, f.nMel)
	for trial := 0; trial < 50; trial++ {
		for k := range power {
			power[k] = rng.Float64() * math.Pow(10, float64(rng.Intn(8)-3))
		}
		f.applyFilterbank(power, got)
		want := denseMel(f.fb, power, f.nMel, f.bins)
		for m := range want {
			if got[m] != want[m] {
				t.Fatalf("trial %d mel %d: banded %v != dense %v", trial, m, got[m], want[m])
			}
		}
	}
}

func TestBandedFilterbankEqualsDenseSynthetic(t *testing.T) {
	checkBandedEqualsDense(t, syntheticFrontend(t))
	checkBandedEqualsDense(t, tinyFrontend(t))
}

func TestBandedFilterbankEqualsDenseReal(t *testing.T) {
	f := realFrontend(t)
	checkBandedEqualsDense(t, f)
	nz := 0
	for _, w := range f.fb {
		if w != 0 {
			nz++
		}
	}
	t.Logf("filterbank non-zero weights: %d of %d", nz, len(f.fb))
}

func TestBandedFilterbankAllZeroRow(t *testing.T) {
	f := tinyFrontend(t)
	for k := 0; k < f.bins; k++ {
		f.fb[k] = 0 // zero out row 0
	}
	f.buildBands()
	checkBandedEqualsDense(t, f)
}
