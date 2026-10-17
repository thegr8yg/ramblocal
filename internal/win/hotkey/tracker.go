package hotkey

// Tracker turns raw down/up transitions of one key into Down and Up events. Auto-repeat downs produce nothing,
// and an up without a preceding down is ignored. The zero value is ready to use.
type Tracker struct {
	down bool
}

// Feed records one transition (down=true for a key-down message, including auto-repeat) and returns the event it
// produces, if any.
func (t *Tracker) Feed(down bool) (Event, bool) {
	switch {
	case down && !t.down:
		t.down = true
		return Down, true
	case !down && t.down:
		t.down = false
		return Up, true
	}
	return 0, false
}
