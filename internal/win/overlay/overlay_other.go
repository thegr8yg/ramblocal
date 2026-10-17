//go:build !windows

package overlay

import (
	"errors"
	"runtime"
)

var errUnsupported = errors.New("overlay: the floating status pill is Windows-only (GOOS=" + runtime.GOOS + ")")

// platform holds no window state off Windows.
type platform struct{}

func (o *Overlay) start() error { return errUnsupported }
func (o *Overlay) wake()        {}
func (o *Overlay) destroy()     {}
