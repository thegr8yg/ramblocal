package asr

import (
	"context"
	"fmt"
)

// DecoderState is the prediction-network state, opaque to the decoding loop.
type DecoderState any

// Joint runs one prediction-network + joint step for a TDT model.
type Joint interface {
	InitialState() DecoderState
	Step(t, lastToken int, state DecoderState) (tokenLogits, durLogits []float32, next DecoderState, err error)
}

// Hyp is a decoded hypothesis: Frames[i] is the encoder frame at which Tokens[i] was emitted.
type Hyp struct {
	Tokens []int
	Frames []int
}

// GreedyTDT runs greedy Token-and-Duration Transducer decoding over numFrames encoder frames.
func GreedyTDT(ctx context.Context, j Joint, numFrames, blankID int, durations []int, maxSymbolsPerFrame int) (Hyp, error) {
	if len(durations) == 0 {
		return Hyp{}, fmt.Errorf("tdt: durations must not be empty")
	}
	for _, d := range durations {
		if d < 0 {
			return Hyp{}, fmt.Errorf("tdt: negative duration %d", d)
		}
	}
	if blankID < 0 {
		return Hyp{}, fmt.Errorf("tdt: blankID %d must be >= 0", blankID)
	}
	if maxSymbolsPerFrame < 1 {
		return Hyp{}, fmt.Errorf("tdt: maxSymbolsPerFrame %d must be >= 1", maxSymbolsPerFrame)
	}
	var h Hyp
	state := j.InitialState()
	last := blankID
	emitted := 0
	for t := 0; t < numFrames; {
		if err := ctx.Err(); err != nil {
			return Hyp{}, err
		}
		tok, dur, next, err := j.Step(t, last, state)
		if err != nil {
			return Hyp{}, fmt.Errorf("joint step at frame %d: %w", t, err)
		}
		if len(tok) != blankID+1 || len(dur) != len(durations) {
			return Hyp{}, fmt.Errorf("joint returned %d token / %d duration logits, want %d / %d",
				len(tok), len(dur), blankID+1, len(durations))
		}
		k := argmax(tok)
		d := durations[argmax(dur)]
		if k != blankID {
			h.Tokens = append(h.Tokens, k)
			h.Frames = append(h.Frames, t)
			state = next
			last = k
			emitted++
		} else if d == 0 {
			d = 1
		}
		if d == 0 && emitted >= maxSymbolsPerFrame {
			d = 1
		}
		if d > 0 {
			t += d
			emitted = 0
		}
	}
	return h, nil
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}
