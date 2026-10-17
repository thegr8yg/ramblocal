package capture

import (
	"math"
	"sync/atomic"
)

const (
	meterFloorDB = -60.0 // at or below this RMS level the meter reads 0; 0 dBFS reads 1
	meterAttack  = 0.6   // fraction of the gap to the target closed per block while rising
	meterRelease = 0.15  // fraction closed per block while falling
	silenceDB    = -120.0
)

// meter tracks a smoothed 0..1 microphone level. update is called from the capture goroutine;
// get may be called from any goroutine.
type meter struct {
	cur   float32 // smoothed value; touched only by update
	shown atomic.Uint32
}

// update folds one block of samples into the smoothed level. Empty blocks are ignored.
func (m *meter) update(block []float32) {
	if len(block) == 0 {
		return
	}
	target := dbToUnit(rmsDB(block))
	k := float32(meterRelease)
	if target > m.cur {
		k = meterAttack
	}
	m.cur += (target - m.cur) * k
	m.shown.Store(math.Float32bits(m.cur))
}

func (m *meter) get() float32 {
	return math.Float32frombits(m.shown.Load())
}

// rmsDB returns the RMS level of block in dBFS (full scale sine is about -3 dB). Silence is -120 dB.
func rmsDB(block []float32) float32 {
	if len(block) == 0 {
		return silenceDB
	}
	var sum float64
	for _, v := range block {
		sum += float64(v) * float64(v)
	}
	ms := sum / float64(len(block))
	if ms <= 1e-12 {
		return silenceDB
	}
	return float32(10 * math.Log10(ms))
}

// dbToUnit maps a level in dB to 0..1 linearly across meterFloorDB..0.
func dbToUnit(db float32) float32 {
	u := (db - meterFloorDB) / -meterFloorDB
	switch {
	case u < 0:
		return 0
	case u > 1:
		return 1
	}
	return u
}
