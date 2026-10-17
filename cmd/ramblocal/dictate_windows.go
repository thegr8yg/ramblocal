package main

import (
	"fmt"
	"io"

	"github.com/thegr8yg/ramblocal/internal/audio/capture"
	"github.com/thegr8yg/ramblocal/internal/win/hotkey"
	"github.com/thegr8yg/ramblocal/internal/win/inject"
	"github.com/thegr8yg/ramblocal/internal/win/overlay"
)

// openDevices opens the overlay, the microphone and the hotkey listener, in that order.
// On error everything opened so far is closed again.
func openDevices(opts dictateOptions, stderr io.Writer) (*devices, error) {
	vk, err := hotkey.ParseKey(opts.key)
	if err != nil {
		return nil, fmt.Errorf("key %q: %w", opts.key, err)
	}

	d := &devices{inj: winInjector{}}
	fail := func(err error) (*devices, error) {
		d.close()
		return nil, err
	}

	if opts.overlay {
		ov, err := overlay.New()
		if err != nil {
			return fail(fmt.Errorf("overlay: %w", err))
		}
		d.ovl = winOverlay{ov}
		d.closers = append(d.closers, ov.Close)
	} else {
		d.ovl = nopOverlay{}
	}

	rec, err := capture.Open()
	if err != nil {
		return fail(fmt.Errorf("microphone: %w", err))
	}
	d.rec = rec
	d.closers = append(d.closers, func() { _ = rec.Close() })

	events, stop, err := hotkey.Listen(vk, opts.swallow)
	if err != nil {
		return fail(fmt.Errorf("hotkey %q: %w", opts.key, err))
	}
	d.closers = append(d.closers, stop)

	keys := make(chan KeyEvent)
	go forwardKeys(events, keys)
	d.keys = keys
	return d, nil
}

// forwardKeys converts hotkey events to KeyEvent until the hotkey listener closes its channel.
func forwardKeys(events <-chan hotkey.Event, keys chan<- KeyEvent) {
	defer close(keys)
	for ev := range events {
		switch ev {
		case hotkey.Down:
			keys <- KeyDown
		case hotkey.Up:
			keys <- KeyUp
		}
	}
}

type winInjector struct{}

func (winInjector) Paste(text string) error { return inject.Paste(text) }

type winOverlay struct{ o *overlay.Overlay }

func (w winOverlay) SetState(s State) {
	var st overlay.State
	switch s {
	case StateListening:
		st = overlay.Listening
	case StateWorking:
		st = overlay.Working
	case StateFailed:
		st = overlay.Failed
	default:
		st = overlay.Idle
	}
	w.o.SetState(st)
}

func (w winOverlay) SetLevel(l float32)    { w.o.SetLevel(l) }
func (w winOverlay) Quit() <-chan struct{} { return w.o.Quit() }
