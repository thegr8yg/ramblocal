// Package mel computes the NeMo-exact log-mel features Parakeet expects.
package mel

import (
	"fmt"
	"math"
)

// fftPlan holds the bit-reversal permutation and twiddle factors for one power-of-two size.
type fftPlan struct {
	n        int
	rev      []int
	cos, sin []float64 // cos/sin(-2*pi*k/n) for k in [0, n/2)
}

func isPow2(n int) bool { return n >= 1 && n&(n-1) == 0 }

func newFFTPlan(n int) (*fftPlan, error) {
	if !isPow2(n) {
		return nil, fmt.Errorf("fft size %d is not a power of two", n)
	}
	p := &fftPlan{n: n, rev: make([]int, n), cos: make([]float64, n/2), sin: make([]float64, n/2)}
	bits := 0
	for 1<<bits < n {
		bits++
	}
	for i := 0; i < n; i++ {
		r := 0
		for b := 0; b < bits; b++ {
			if i&(1<<b) != 0 {
				r |= 1 << (bits - 1 - b)
			}
		}
		p.rev[i] = r
	}
	for k := 0; k < n/2; k++ {
		a := -2 * math.Pi * float64(k) / float64(n)
		p.cos[k], p.sin[k] = math.Cos(a), math.Sin(a)
	}
	return p, nil
}

// transform is an in-place iterative radix-2 decimation-in-time forward FFT.
func (p *fftPlan) transform(re, im []float64) {
	n := p.n
	for i := 0; i < n; i++ {
		if j := p.rev[i]; j > i {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		half, step := size/2, n/size
		for start := 0; start < n; start += size {
			for k := 0; k < half; k++ {
				wr, wi := p.cos[k*step], p.sin[k*step]
				a, b := start+k, start+k+half
				tr := re[b]*wr - im[b]*wi
				ti := re[b]*wi + im[b]*wr
				re[b], im[b] = re[a]-tr, im[a]-ti
				re[a], im[a] = re[a]+tr, im[a]+ti
			}
		}
	}
}

// fft computes the forward FFT of (re, im) in place; len must be a power of two.
func fft(re, im []float64) error {
	if len(re) != len(im) {
		return fmt.Errorf("fft: re/im length mismatch %d vs %d", len(re), len(im))
	}
	p, err := newFFTPlan(len(re))
	if err != nil {
		return err
	}
	p.transform(re, im)
	return nil
}
