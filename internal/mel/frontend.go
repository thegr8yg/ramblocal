package mel

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/thegr8yg/ramblocal/internal/asr"
)

// Frontend is NeMo's FilterbankFeatures in eval mode (no dither, batch of 1).
// It is immutable after New and safe for concurrent use.
type Frontend struct {
	cfg  asr.Frontend
	nMel int
	bins int
	fb   []float64 // [nMel, bins] row-major
	win  []float64 // WinLength samples
	plan *fftPlan
}

// New loads the filterbank and window files named by the manifest's frontend section.
func New(dir string, m *asr.Manifest) (*Frontend, error) {
	if m == nil || m.Frontend == nil {
		return nil, errors.New("mel: manifest has no frontend section")
	}
	c := *m.Frontend
	plan, err := newFFTPlan(c.NFFT)
	if err != nil {
		return nil, fmt.Errorf("mel: %w", err)
	}
	f := &Frontend{cfg: c, nMel: m.NumFeatures, bins: c.NFFT/2 + 1, plan: plan}
	if f.nMel <= 0 || c.WinLength <= 0 || c.WinLength > c.NFFT || c.HopLength <= 0 {
		return nil, errors.New("mel: invalid frontend dimensions")
	}
	if f.fb, err = readF32(filepath.Join(dir, c.Filterbank), f.nMel*f.bins); err != nil {
		return nil, fmt.Errorf("mel filterbank: %w", err)
	}
	if f.win, err = readF32(filepath.Join(dir, c.Window), c.WinLength); err != nil {
		return nil, fmt.Errorf("mel window: %w", err)
	}
	return f, nil
}

func readF32(path string, n int) ([]float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) != 4*n {
		return nil, fmt.Errorf("%s: %d bytes, want %d (%d float32)", path, len(raw), 4*n, n)
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
	}
	return out, nil
}

// Compute returns row-major feats [NumFeatures, width], plus width and valid.
// width = 1 + len(pcm)/hop STFT frames; only the first valid = len(pcm)/hop columns
// are normalized and meaningful, the remaining column is exactly zero (NeMo's
// masked padding). Inputs with valid < 2 yield (nil, 0, 0), as NeMo rejects them.
func (f *Frontend) Compute(pcm []float32) (feats []float32, width, valid int) {
	c := f.cfg
	n, hop, nfft := len(pcm), c.HopLength, c.NFFT
	valid = n / hop
	if valid < 2 {
		return nil, 0, 0
	}
	width = 1 + n/hop

	// Preemphasis then zero padding of nfft/2 on both sides (center=True, constant).
	pad := nfft / 2
	padded := make([]float64, n+nfft)
	padded[pad] = float64(pcm[0])
	for i := 1; i < n; i++ {
		padded[pad+i] = float64(pcm[i]) - c.Preemph*float64(pcm[i-1])
	}

	off := (nfft - c.WinLength) / 2
	re := make([]float64, nfft)
	im := make([]float64, nfft)
	mel := make([]float64, f.nMel*width)
	for t := 0; t < width; t++ {
		frame := padded[t*hop : t*hop+nfft]
		clear(re)
		clear(im)
		for j, w := range f.win {
			re[off+j] = frame[off+j] * w
		}
		f.plan.transform(re, im)
		power := re[:f.bins] // reuse: re[k]^2+im[k]^2 into re[k]
		for k := 0; k < f.bins; k++ {
			power[k] = re[k]*re[k] + im[k]*im[k]
		}
		for m := 0; m < f.nMel; m++ {
			row := f.fb[m*f.bins : (m+1)*f.bins]
			var s float64
			for k, w := range row {
				s += w * power[k]
			}
			mel[m*width+t] = math.Log(s + c.LogGuard)
		}
	}

	// Per-feature normalization over the valid frames only (Bessel-corrected std).
	feats = make([]float32, f.nMel*width)
	for m := 0; m < f.nMel; m++ {
		row := mel[m*width : m*width+valid]
		var mean float64
		for _, v := range row {
			mean += v
		}
		mean /= float64(valid)
		var ss float64
		for _, v := range row {
			d := v - mean
			ss += d * d
		}
		std := math.Sqrt(ss/float64(valid-1)) + c.NormEps
		out := feats[m*width : m*width+valid]
		for t, v := range row {
			out[t] = float32((v - mean) / std)
		}
		// column t = valid stays exactly 0
	}
	return feats, width, valid
}
