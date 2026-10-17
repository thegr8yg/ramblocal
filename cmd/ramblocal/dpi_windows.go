package main

import "syscall"

// The overlay draws in device pixels, so opt out of DPI virtualization before any window exists.
func init() {
	syscall.NewLazyDLL("user32.dll").NewProc("SetProcessDPIAware").Call()
}
