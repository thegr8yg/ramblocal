package bench

import (
	"math"
	"sort"
)

// Pct holds latency percentiles in milliseconds.
type Pct struct {
	P50 float64 `json:"p50"`
	P95 float64 `json:"p95"`
	Max float64 `json:"max"`
}

// Percentiles uses the nearest-rank method.
func Percentiles(ms []float64) Pct {
	if len(ms) == 0 {
		return Pct{}
	}
	s := append([]float64(nil), ms...)
	sort.Float64s(s)
	rank := func(p float64) float64 {
		i := int(math.Ceil(p/100*float64(len(s)))) - 1
		return s[max(0, min(i, len(s)-1))]
	}
	return Pct{P50: rank(50), P95: rank(95), Max: s[len(s)-1]}
}
