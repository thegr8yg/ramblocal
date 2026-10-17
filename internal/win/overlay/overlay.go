// Package overlay is the small always-on-top status widget (like Wispr Flow's pill).
package overlay

import "errors"

type State int

const (
	Idle      State = iota // waiting; small, dim
	Listening              // key held; expanded, shows the mic level
	Working                // transcribing; spinner or pulse
	Failed                 // brief error flash
)

type Overlay struct{}

// New creates the window on its own locked OS thread with its own message loop.
func New() (*Overlay, error) { return nil, errors.New("overlay: not implemented") }

func (o *Overlay) SetState(s State)   {}
func (o *Overlay) SetLevel(l float32) {} // 0..1 microphone level, only drawn while Listening

// Quit is closed when the user picks Quit from the widget's right-click menu.
func (o *Overlay) Quit() <-chan struct{} { return nil }

func (o *Overlay) Close() {}
