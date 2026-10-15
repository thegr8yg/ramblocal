package asr

import (
	"context"
	"sync"
)

// FakeEngine is a test double. With Fn nil it returns an empty Result with AudioSec set.
type FakeEngine struct {
	Fn    func(pcm []float32, opts Options) (Result, error)
	mu    sync.Mutex
	calls int
}

func (f *FakeEngine) Transcribe(ctx context.Context, pcm []float32, opts Options) (Result, error) {
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	f.mu.Lock()
	f.calls++
	fn := f.Fn
	f.mu.Unlock()
	if fn == nil {
		return Result{AudioSec: float64(len(pcm)) / SampleRate}, nil
	}
	return fn(pcm, opts)
}

func (f *FakeEngine) Calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func (f *FakeEngine) Name() string { return "fake" }
func (f *FakeEngine) Close() error { return nil }
