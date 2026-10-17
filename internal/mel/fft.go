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
	p.butterflies(re, im)
}

// butterflies runs the FFT stages on input that is already in bit-reversed order.
func (p *fftPlan) butterflies(re, im []float64) {
	n := p.n
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

// realFFTPlan computes the power spectrum |X[k]|^2, k in [0, n/2], of a real signal of
// length n using one n/2-point complex FFT: even samples go to the real part and odd
// samples to the imaginary part, and the two interleaved spectra are untangled
// afterwards. All arithmetic is float64.
type realFFTPlan struct {
	n, m     int       // n real samples, m = n/2 complex points
	half     *fftPlan  // size m
	cos, sin []float64 // cos/sin(-2*pi*k/n) for k in [0, m]
}

func newRealFFTPlan(n int) (*realFFTPlan, error) {
	if !isPow2(n) || n < 2 {
		return nil, fmt.Errorf("real fft size %d is not a power of two >= 2", n)
	}
	m := n / 2
	half, err := newFFTPlan(m)
	if err != nil {
		return nil, err
	}
	p := &realFFTPlan{n: n, m: m, half: half, cos: make([]float64, m+1), sin: make([]float64, m+1)}
	for k := 0; k <= m; k++ {
		a := -2 * math.Pi * float64(k) / float64(n)
		p.cos[k], p.sin[k] = math.Cos(a), math.Sin(a)
	}
	return p, nil
}

// slot returns where real sample j lands in the bit-reversed complex input:
// index slot(j) of the real array when j is even, of the imaginary array when odd.
func (p *realFFTPlan) slot(j int) int { return p.half.rev[j>>1] }

// loadReal packs the n real samples x into the (bit-reversed) complex input zr, zi.
func (p *realFFTPlan) loadReal(x, zr, zi []float64) {
	for j, v := range x {
		if j&1 == 0 {
			zr[p.slot(j)] = v
		} else {
			zi[p.slot(j)] = v
		}
	}
}

// power transforms the packed input (zr, zi, length n/2, bit-reversed, as filled by
// loadReal) in place and writes |X[k]|^2 for k in [0, n/2] to out (length n/2+1).
func (p *realFFTPlan) power(zr, zi, out []float64) {
	m := p.m
	p.half.butterflies(zr, zi)
	for k := 0; k <= m; k++ {
		i, j := k&(m-1), (m-k)&(m-1) // Z is m-periodic: Z[m] = Z[0]
		a, b := zr[i], zi[i]
		c, d := zr[j], zi[j]
		// Fe = (Z[k] + conj(Z[m-k]))/2, Fo = -i*(Z[k] - conj(Z[m-k]))/2
		fer, fei := 0.5*(a+c), 0.5*(b-d)
		for_, foi := 0.5*(b+d), -0.5*(a-c)
		wc, ws := p.cos[k], p.sin[k]
		xr := fer + wc*for_ - ws*foi
		xi := fei + wc*foi + ws*for_
		out[k] = xr*xr + xi*xi
	}
}
