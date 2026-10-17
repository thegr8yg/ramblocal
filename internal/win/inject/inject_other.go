//go:build !windows

package inject

import "errors"

var errUnsupported = errors.New("inject: paste and SendInput are only supported on Windows")

// platform reports that there is no clipboard or input backend on this OS.
func platform() (osAPI, error) {
	return nil, errUnsupported
}
