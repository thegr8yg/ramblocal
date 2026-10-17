// Package overlay is the small always-on-top status widget (like Wispr Flow's pill).
//
// The window lives on its own locked OS thread with its own message loop (Windows only).
// This file holds the portable state that the window thread reads: SetState and SetLevel
// only store into atomics and wake the window, so they are safe to call from any goroutine.
package overlay

import (
	"math"
	"sync"
	"sync/atomic"
)

// State is what the pill is showing.
type State int

const (
	Idle      State = iota // waiting; small, dim
	Listening              // key held; expanded, shows the mic level
	Working                // transcribing; animated dots
	Failed                 // brief red flash, then back to Idle
)

func (s State) String() string {
	switch s {
	case Idle:
		return "idle"
	case Listening:
		return "listening"
	case Working:
		return "working"
	case Failed:
		return "failed"
	}
	return "unknown"
}

// Overlay is a handle to the floating pill. Create one with New and release it with Close.
type Overlay struct {
	state atomic.Int32  // State
	level atomic.Uint32 // float32 bits, 0..1

	quit      chan struct{}
	quitOnce  sync.Once
	closeOnce sync.Once

	pl platform // window-thread state, see overlay_windows.go
}

// New creates the window on its own locked OS thread with its own message loop.
// It returns once the window exists. On platforms other than Windows it returns an error.
func New() (*Overlay, error) {
	o := &Overlay{quit: make(chan struct{})}
	if err := o.start(); err != nil {
		return nil, err
	}
	return o, nil
}

// SetState changes what the pill shows. Safe for concurrent use.
func (o *Overlay) SetState(s State) {
	if s < Idle || s > Failed {
		s = Idle
	}
	o.state.Store(int32(s))
	o.wake()
}

// SetLevel sets the microphone level, 0..1. It is only drawn while Listening. Safe for concurrent use.
func (o *Overlay) SetLevel(l float32) {
	if !(l > 0) { // also catches NaN
		l = 0
	}
	if l > 1 {
		l = 1
	}
	o.level.Store(math.Float32bits(l))
	o.wake()
}

// Quit is closed when the user picks Quit from the widget's right-click menu.
func (o *Overlay) Quit() <-chan struct{} { return o.quit }

// Close destroys the window and ends its thread. It is safe to call more than once.
func (o *Overlay) Close() { o.closeOnce.Do(o.destroy) }

// signalQuit closes the Quit channel exactly once.
func (o *Overlay) signalQuit() {
	o.quitOnce.Do(func() { close(o.quit) })
}
