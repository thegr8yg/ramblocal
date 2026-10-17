//go:build !windows

package capture

import "testing"

func TestOpenUnsupportedOffWindows(t *testing.T) {
	r, err := Open()
	if err == nil {
		r.Close()
		t.Fatal("Open succeeded on a non-Windows platform")
	}
	if r != nil {
		t.Fatal("Open returned a recorder alongside an error")
	}
}
