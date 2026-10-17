// Package inject types text into the focused window by pasting it through the clipboard.
package inject

import "errors"

// Paste puts text on the clipboard, sends Ctrl+V, then restores the previous clipboard text.
func Paste(text string) error { return errors.New("inject: not implemented") }
