// Package capture records the default microphone as 16 kHz mono float32.
package capture

import "errors"

type Recorder struct{}

// Open starts the microphone and keeps it running, holding a 300 ms pre-roll so the first word is not clipped.
func Open() (*Recorder, error) { return nil, errors.New("capture: not implemented") }

// Begin marks the start of an utterance, including the pre-roll.
func (r *Recorder) Begin() {}

// End returns all samples since Begin (16 kHz mono, -1..1) and stops collecting them.
func (r *Recorder) End() []float32 { return nil }

// Level is the recent RMS level, 0..1, for the overlay meter.
func (r *Recorder) Level() float32 { return 0 }

func (r *Recorder) Close() error { return nil }
