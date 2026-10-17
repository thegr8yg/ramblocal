package overlay

import (
	"math"
	"runtime"
	"testing"
)

func TestSetLevelClamps(t *testing.T) {
	o := &Overlay{quit: make(chan struct{})}
	for in, want := range map[float32]float32{-1: 0, 0.25: 0.25, 3: 1, float32(math.NaN()): 0} {
		o.SetLevel(in)
		if got := math.Float32frombits(o.level.Load()); got != want {
			t.Errorf("SetLevel(%v): stored %v, want %v", in, got, want)
		}
	}
}

func TestSetStateStoresAndSanitises(t *testing.T) {
	o := &Overlay{quit: make(chan struct{})}
	o.SetState(Working)
	if got := State(o.state.Load()); got != Working {
		t.Errorf("state=%v, want %v", got, Working)
	}
	o.SetState(State(42))
	if got := State(o.state.Load()); got != Idle {
		t.Errorf("out-of-range state stored as %v, want %v", got, Idle)
	}
}

func TestQuitClosesOnceAndSafeToRepeat(t *testing.T) {
	o := &Overlay{quit: make(chan struct{})}
	o.signalQuit()
	o.signalQuit()
	select {
	case <-o.Quit():
	default:
		t.Fatal("Quit channel not closed after signalQuit")
	}
}

func TestCloseTwiceWithoutWindow(t *testing.T) {
	o := &Overlay{quit: make(chan struct{})}
	o.Close()
	o.Close()
}

func TestNewUnsupportedOffWindows(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("window creation is covered by overlay_windows_test.go")
	}
	o, err := New()
	if err == nil || o != nil {
		t.Fatalf("New() on %s = (%v, %v), want an error", runtime.GOOS, o, err)
	}
}

func TestStateString(t *testing.T) {
	for s, want := range map[State]string{Idle: "idle", Listening: "listening", Working: "working", Failed: "failed"} {
		if got := s.String(); got != want {
			t.Errorf("%d.String()=%q, want %q", int(s), got, want)
		}
	}
}
