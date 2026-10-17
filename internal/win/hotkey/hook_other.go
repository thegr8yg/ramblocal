//go:build !windows

package hotkey

import (
	"context"
	"errors"
	"io"
)

var errUnsupported = errors.New("hotkey: the keyboard hook is only available on Windows")

func installHook(sink func(rawKey) bool) (func(), error) { return nil, errUnsupported }

// Probe is only implemented on Windows; on other systems it returns an error.
func Probe(ctx context.Context, w io.Writer) error { return errUnsupported }
