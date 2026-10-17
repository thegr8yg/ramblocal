package hotkey

import "testing"

type step struct {
	down bool
	want Event // zero means no event
}

func runSteps(t *testing.T, steps []step) {
	t.Helper()
	var tr Tracker
	for i, s := range steps {
		ev, ok := tr.Feed(s.down)
		switch {
		case s.want == 0 && ok:
			t.Errorf("step %d (down=%v): got event %v, want none", i, s.down, ev)
		case s.want != 0 && !ok:
			t.Errorf("step %d (down=%v): got no event, want %v", i, s.down, s.want)
		case s.want != 0 && ev != s.want:
			t.Errorf("step %d (down=%v): got %v, want %v", i, s.down, ev, s.want)
		}
	}
}

func TestTrackerSinglePress(t *testing.T) {
	runSteps(t, []step{
		{true, Down},
		{false, Up},
	})
}

func TestTrackerIgnoresAutoRepeat(t *testing.T) {
	runSteps(t, []step{
		{true, Down},
		{true, 0},
		{true, 0},
		{true, 0},
		{false, Up},
	})
}

func TestTrackerRepeatedPresses(t *testing.T) {
	runSteps(t, []step{
		{true, Down},
		{false, Up},
		{true, Down},
		{true, 0},
		{false, Up},
		{false, 0}, // spurious extra up
	})
}

func TestTrackerIgnoresUpWithoutDown(t *testing.T) {
	runSteps(t, []step{
		{false, 0},
		{false, 0},
		{true, Down},
		{false, Up},
	})
}

func TestTrackerZeroValueIdle(t *testing.T) {
	var tr Tracker
	if ev, ok := tr.Feed(false); ok {
		t.Fatalf("first up produced %v", ev)
	}
}
