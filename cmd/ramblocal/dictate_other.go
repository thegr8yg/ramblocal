//go:build !windows

package main

import (
	"errors"
	"io"
)

var errDictateNotWindows = errors.New("dictate needs Windows: the hotkey, microphone, overlay and paste are Windows-only")

// openDevices is Windows-only; other platforms get a clear error.
func openDevices(opts dictateOptions, stderr io.Writer) (*devices, error) {
	return nil, errDictateNotWindows
}
