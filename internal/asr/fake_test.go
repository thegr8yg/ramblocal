package asr

import (
	"context"
	"errors"
	"testing"
)

func TestFakeEngineDefaultAndFn(t *testing.T) {
	var f FakeEngine
	res, err := f.Transcribe(context.Background(), make([]float32, SampleRate), Options{})
	if err != nil || res.AudioSec != 1 {
		t.Fatalf("default: %+v, %v", res, err)
	}
	f.Fn = func(pcm []float32, _ Options) (Result, error) { return Result{Text: "hi"}, nil }
	res, _ = f.Transcribe(context.Background(), nil, Options{})
	if res.Text != "hi" || f.Calls() != 2 {
		t.Fatalf("fn: %+v calls=%d", res, f.Calls())
	}
}

func TestFakeEngineHonorsCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var f FakeEngine
	if _, err := f.Transcribe(ctx, nil, Options{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}
