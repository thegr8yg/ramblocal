package mel

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/thegr8yg/ramblocal/internal/asr"
)

// Frontend is NeMo's FilterbankFeatures in eval mode (no dither, batch of 1).
// It is immutable after New and safe for concurrent use.
type Frontend struct {
	cfg  asr.Frontend
	nMel int
	bins int
	fb   []float64 // [nMel, bins] row-major (dense reference; Compute uses the banded copy)
	win  []float64 // WinLength samples
	plan *realFFTPlan

	// Banded filterbank: row m's non-zero weights live in bins [bandLo[m], bandHi[m])
	// and are stored contiguously in bandW starting at bandOff[m]. Bins outside the
	// band are exactly 0.0, so skipping them leaves the dense sum unchanged bit for bit.
	bandLo, bandHi, bandOff []int
	bandW                   []float64
}

// New loads the filterbank and window files named by the manifest's frontend section.
func New(dir string, m *asr.Manifest) (*Frontend, error) {
	if m == nil || m.Frontend == nil {
		return nil, errors.New("mel: manifest has no frontend section")
	}
	c := *m.Frontend
	plan, err := newRealFFTPlan(c.NFFT)
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
	f.buildBands()
	return f, nil
}

// buildBands derives the per-row non-zero band of the filterbank from f.fb.
func (f *Frontend) buildBands() {
	f.bandLo = make([]int, f.nMel)
	f.bandHi = make([]int, f.nMel)
	f.bandOff = make([]int, f.nMel)
	f.bandW = f.bandW[:0]
	for m := 0; m < f.nMel; m++ {
		row := f.fb[m*f.bins : (m+1)*f.bins]
		lo, hi := 0, 0
		for k, w := range row {
			if w != 0 {
				if hi == 0 {
					lo = k
				}
				hi = k + 1
			}
		}
		f.bandLo[m], f.bandHi[m], f.bandOff[m] = lo, hi, len(f.bandW)
		f.bandW = append(f.bandW, row[lo:hi]...)
	}
}

// applyFilterbank writes out[m] = sum_k fb[m][k]*power[k] for every mel row, visiting
// only each row's non-zero band (same summation order as the dense product).
func (f *Frontend) applyFilterbank(power, out []float64) {
	for m := 0; m < f.nMel; m++ {
		lo, hi, off := f.bandLo[m], f.bandHi[m], f.bandOff[m]
		w := f.bandW[off : off+hi-lo]
		p := power[lo:hi]
		var s float64
		for k, wk := range w {
			s += wk * p[k]
		}
		out[m] = s
	}
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

const (
	// minParallelFrames is the frame count below which Compute stays serial.
	minParallelFrames = 64
	// maxWorkers caps the worker pool.
	maxWorkers = 8
	// frameChunk is the number of consecutive frames a worker claims at a time.
	frameChunk = 16
	// maxPooledElems bounds the call-level buffers kept in the pool (float64 elements).
	maxPooledElems = 1 << 22
)

// frameScratch is the per-worker scratch of the per-frame path.
type frameScratch struct {
	zr, zi, power, melCol []float64
}

// callBuffers are the large per-call buffers (padded signal, mel matrix).
type callBuffers struct {
	padded, mel []float64
}

var (
	frameScratchPool sync.Pool // *frameScratch
	callBuffersPool  sync.Pool // *callBuffers
)

func grow(s []float64, n int) []float64 {
	if cap(s) < n {
		return make([]float64, n)
	}
	return s[:n]
}

func (f *Frontend) getScratch() *frameScratch {
	s, _ := frameScratchPool.Get().(*frameScratch)
	if s == nil {
		s = new(frameScratch)
	}
	s.zr = grow(s.zr, f.plan.m)
	s.zi = grow(s.zi, f.plan.m)
	s.power = grow(s.power, f.bins)
	s.melCol = grow(s.melCol, f.nMel)
	return s
}

// Compute returns row-major feats [NumFeatures, width], plus width and valid.
// width = 1 + len(pcm)/hop STFT frames; only the first valid = len(pcm)/hop columns
// are normalized and meaningful, the remaining column is exactly zero (NeMo's
// masked padding). Inputs with valid < 2 yield (nil, 0, 0), as NeMo rejects them.
//
// Frames are independent, so they are computed by a bounded worker pool; every
// output element is produced by the same sequential arithmetic regardless of the
// worker count, so the result is bit-identical for any GOMAXPROCS.
func (f *Frontend) Compute(pcm []float32) (feats []float32, width, valid int) {
	c := f.cfg
	n, hop, nfft := len(pcm), c.HopLength, c.NFFT
	valid = n / hop
	if valid < 2 {
		return nil, 0, 0
	}
	width = 1 + n/hop

	buf, _ := callBuffersPool.Get().(*callBuffers)
	if buf == nil {
		buf = new(callBuffers)
	}
	buf.padded = grow(buf.padded, n+nfft)
	buf.mel = grow(buf.mel, f.nMel*width)
	padded, mel := buf.padded, buf.mel

	// Preemphasis then zero padding of nfft/2 on both sides (center=True, constant).
	pad := nfft / 2
	clear(padded[:pad])
	clear(padded[pad+n:])
	padded[pad] = float64(pcm[0])
	for i := 1; i < n; i++ {
		padded[pad+i] = float64(pcm[i]) - c.Preemph*float64(pcm[i-1])
	}

	workers := min(runtime.GOMAXPROCS(0), maxWorkers, (width+frameChunk-1)/frameChunk)
	if width < minParallelFrames {
		workers = 1
	}
	if workers <= 1 {
		s := f.getScratch()
		f.frames(padded, mel, width, 0, width, s)
		frameScratchPool.Put(s)
	} else {
		var next atomic.Int64
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				s := f.getScratch()
				defer frameScratchPool.Put(s)
				for {
					lo := int(next.Add(frameChunk)) - frameChunk
					if lo >= width {
						return
					}
					f.frames(padded, mel, width, lo, min(lo+frameChunk, width), s)
				}
			}()
		}
		wg.Wait()
	}

	// Per-feature normalization over the valid frames only (Bessel-corrected std).
	feats = make([]float32, f.nMel*width)
	f.normalize(mel, feats, width, valid, workers)

	if len(buf.padded) <= maxPooledElems && len(buf.mel) <= maxPooledElems {
		callBuffersPool.Put(buf)
	}
	return feats, width, valid
}

// frames computes log-mel columns [lo, hi) into mel (row-major [nMel, width]).
func (f *Frontend) frames(padded, mel []float64, width, lo, hi int, s *frameScratch) {
	hop, nfft := f.cfg.HopLength, f.cfg.NFFT
	off := (nfft - f.cfg.WinLength) / 2
	for t := lo; t < hi; t++ {
		frame := padded[t*hop : t*hop+nfft]
		clear(s.zr)
		clear(s.zi)
		for j, w := range f.win {
			// even samples -> real part, odd -> imaginary part, in bit-reversed slots
			if v, idx := frame[off+j]*w, f.plan.slot(off+j); (off+j)&1 == 0 {
				s.zr[idx] = v
			} else {
				s.zi[idx] = v
			}
		}
		f.plan.power(s.zr, s.zi, s.power)
		f.applyFilterbank(s.power, s.melCol)
		for m, v := range s.melCol {
			mel[m*width+t] = math.Log(v + f.cfg.LogGuard)
		}
	}
}

// normalize applies per-row mean/std normalization over the first valid columns;
// column t = valid stays exactly 0. Rows are independent, so they are split across
// workers without affecting the result.
func (f *Frontend) normalize(mel []float64, feats []float32, width, valid, workers int) {
	rows := func(lo, hi int) {
		for m := lo; m < hi; m++ {
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
			std := math.Sqrt(ss/float64(valid-1)) + f.cfg.NormEps
			out := feats[m*width : m*width+valid]
			for t, v := range row {
				out[t] = float32((v - mean) / std)
			}
		}
	}
	if workers <= 1 {
		rows(0, f.nMel)
		return
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*f.nMel/workers, (w+1)*f.nMel/workers
		wg.Add(1)
		go func() {
			defer wg.Done()
			rows(lo, hi)
		}()
	}
	wg.Wait()
}
