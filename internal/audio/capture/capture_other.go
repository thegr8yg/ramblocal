//go:build !windows

package capture

import "errors"

var errUnsupported = errors.New("capture: microphone capture is only implemented on Windows (winmm)")

func openDevice(onPCM func([]byte)) (device, error) {
	return nil, errUnsupported
}
