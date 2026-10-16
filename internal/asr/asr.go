// Package asr defines the speech-recognition engine contract shared by every backend.
package asr

import (
	"context"
	"time"
)

// SampleRate is the only rate engines accept: 16 kHz mono float32 in [-1, 1].
const SampleRate = 16000

type Word struct {
	Text       string
	Start, End float64 // seconds from the start of the clip
}

type StageTimings struct {
	Mel, Encoder, Decoder, Detok time.Duration
}

type Result struct {
	Text     string
	Tokens   []int
	Words    []Word
	AudioSec float64
	Timings  StageTimings
}

type Options struct {
	Language  string   // "" = model default
	BiasWords []string // adaptive dictionary terms; engines without biasing ignore them
}

// Engine transcribes complete clips. Implementations must be safe for concurrent use.
type Engine interface {
	Transcribe(ctx context.Context, pcm []float32, opts Options) (Result, error)
	Name() string
	Close() error
}
