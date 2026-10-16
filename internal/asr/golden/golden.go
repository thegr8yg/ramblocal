// Package golden loads NeMo-generated reference fixtures written by tools/export.
package golden

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/thegr8yg/ramblocal/internal/audio"
)

type Fixture struct {
	Name     string
	PCM      []float32
	Mel      []float32 // [MelShape[0], MelShape[1]] = [features, frames]
	MelShape []int
	Enc      []float32 // [EncShape[0], EncShape[1]] = [dim, frames]
	EncShape []int
	Tokens   []int
	Text     string
}

func List(dir string) ([]string, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	var names []string
	for _, m := range matches {
		names = append(names, strings.TrimSuffix(filepath.Base(m), ".json"))
	}
	sort.Strings(names)
	return names, nil
}

func Load(dir, name string) (*Fixture, error) {
	base := filepath.Join(dir, name)
	raw, err := os.ReadFile(base + ".json")
	if err != nil {
		return nil, err
	}
	var meta struct {
		Tokens   []int  `json:"tokens"`
		Text     string `json:"text"`
		MelShape []int  `json:"mel_shape"`
		EncShape []int  `json:"enc_shape"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, fmt.Errorf("%s.json: %w", base, err)
	}
	f, err := os.Open(base + ".wav")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	pcm, rate, err := audio.ReadWAV(f)
	if err != nil {
		return nil, fmt.Errorf("%s.wav: %w", base, err)
	}
	if rate != 16000 {
		return nil, fmt.Errorf("%s.wav: rate %d, want 16000", base, rate)
	}
	mel, err := readF32(base+".mel.f32", product(meta.MelShape))
	if err != nil {
		return nil, err
	}
	enc, err := readF32(base+".enc.f32", product(meta.EncShape))
	if err != nil {
		return nil, err
	}
	return &Fixture{Name: name, PCM: pcm, Mel: mel, MelShape: meta.MelShape, Enc: enc,
		EncShape: meta.EncShape, Tokens: meta.Tokens, Text: meta.Text}, nil
}

func readF32(path string, n int) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) != n*4 {
		return nil, fmt.Errorf("%s: %d bytes, want %d", path, len(raw), n*4)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}

func product(shape []int) int {
	p := 1
	for _, s := range shape {
		p *= s
	}
	return p
}

// MaxAbsDiff returns the largest |a[i]-b[i]| over the shorter length.
// If any compared pair is NaN (or the difference is NaN), it returns +Inf so
// that a comparison against a finite tolerance always fails.
func MaxAbsDiff(a, b []float32) float32 {
	var m float32
	for i := 0; i < min(len(a), len(b)); i++ {
		if math.IsNaN(float64(a[i])) || math.IsNaN(float64(b[i])) {
			return float32(math.Inf(1))
		}
		d := a[i] - b[i]
		if math.IsNaN(float64(d)) {
			return float32(math.Inf(1))
		}
		if d < 0 {
			d = -d
		}
		m = max(m, d)
	}
	return m
}
