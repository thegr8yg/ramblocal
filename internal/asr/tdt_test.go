package asr

import (
	"context"
	"errors"
	"slices"
	"testing"
)

// scriptedJoint returns (token, durationIndex) pairs in order, recording the frames it was asked about.
type scriptedJoint struct {
	vocab  int // blank id = vocab
	nDur   int
	script [][2]int
	calls  int
	frames []int
	lasts  []int
	states []int
}

func (s *scriptedJoint) InitialState() DecoderState { return 0 }

func (s *scriptedJoint) Step(t, last int, st DecoderState) ([]float32, []float32, DecoderState, error) {
	s.frames = append(s.frames, t)
	s.lasts = append(s.lasts, last)
	s.states = append(s.states, st.(int))
	step := s.script[min(s.calls, len(s.script)-1)]
	s.calls++
	tok := make([]float32, s.vocab+1)
	tok[step[0]] = 1
	dur := make([]float32, s.nDur)
	dur[step[1]] = 1
	return tok, dur, st.(int) + 1, nil
}

var durs = []int{0, 1, 2, 3, 4}

func TestGreedyTDTBasic(t *testing.T) {
	// vocab 10, blank 10. Emit 3 (dur 0 -> stay), emit 4 (dur 2), blank (dur 1), emit 5 (dur 4) -> past end.
	j := &scriptedJoint{vocab: 10, nDur: 5, script: [][2]int{{3, 0}, {4, 2}, {10, 1}, {5, 4}}}
	h, err := GreedyTDT(context.Background(), j, 5, 10, durs, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(h.Tokens, []int{3, 4, 5}) || !slices.Equal(h.Frames, []int{0, 0, 3}) {
		t.Fatalf("hyp = %+v", h)
	}
	if !slices.Equal(j.frames, []int{0, 0, 2, 3}) {
		t.Fatalf("visited frames %v", j.frames)
	}
	if !slices.Equal(j.lasts, []int{10, 3, 4, 4}) {
		t.Fatalf("lastToken sequence %v (blank must not change lastToken)", j.lasts)
	}
	if !slices.Equal(j.states, []int{0, 1, 2, 2}) {
		t.Fatalf("state sequence %v (want [0 1 2 2]: blank must not advance state)", j.states)
	}
}

func TestGreedyTDTRejectsBadConfig(t *testing.T) {
	cases := []struct {
		name      string
		blankID   int
		durations []int
		maxSym    int
	}{
		{"empty durations", 10, nil, 10},
		{"negative duration", 10, []int{0, -1, 2}, 10},
		{"negative blank id", -1, durs, 10},
		{"zero maxSymbolsPerFrame", 10, durs, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			j := &scriptedJoint{vocab: 10, nDur: 5, script: [][2]int{{10, 1}}}
			if _, err := GreedyTDT(context.Background(), j, 3, tc.blankID, tc.durations, tc.maxSym); err == nil {
				t.Fatal("expected error")
			}
			if j.calls != 0 {
				t.Fatalf("joint called %d times, want 0", j.calls)
			}
		})
	}
}

func TestGreedyTDTBlankDurationZeroAdvances(t *testing.T) {
	j := &scriptedJoint{vocab: 10, nDur: 5, script: [][2]int{{10, 0}}}
	h, err := GreedyTDT(context.Background(), j, 3, 10, durs, 10)
	if err != nil || len(h.Tokens) != 0 || j.calls != 3 {
		t.Fatalf("hyp=%+v calls=%d err=%v", h, j.calls, err)
	}
}

func TestGreedyTDTSymbolCap(t *testing.T) {
	// Always non-blank with duration 0: must emit maxSymbols per frame, then advance.
	j := &scriptedJoint{vocab: 10, nDur: 5, script: [][2]int{{7, 0}}}
	h, err := GreedyTDT(context.Background(), j, 2, 10, durs, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(h.Frames, []int{0, 0, 0, 1, 1, 1}) {
		t.Fatalf("frames = %v", h.Frames)
	}
}

func TestGreedyTDTEmptyAndErrors(t *testing.T) {
	j := &scriptedJoint{vocab: 10, nDur: 5, script: [][2]int{{10, 1}}}
	if h, err := GreedyTDT(context.Background(), j, 0, 10, durs, 10); err != nil || len(h.Tokens) != 0 || j.calls != 0 {
		t.Fatalf("zero frames: %+v %v calls=%d", h, err, j.calls)
	}
	bad := &scriptedJoint{vocab: 5, nDur: 5, script: [][2]int{{1, 1}}}
	if _, err := GreedyTDT(context.Background(), bad, 3, 10, durs, 10); err == nil {
		t.Fatal("token logits length mismatch: expected error")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := GreedyTDT(ctx, j, 3, 10, durs, 10); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
}
