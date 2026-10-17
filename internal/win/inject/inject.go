// Package inject types text into the focused window by pasting it through the clipboard.
//
// Paste saves the clipboard's CF_UNICODETEXT, places the new text on the clipboard, sends
// Ctrl+V, waits briefly, and restores the saved text only if the clipboard still holds the
// pasted text. If the clipboard path fails before Ctrl+V is delivered, Paste types the text
// with SendInput instead (see TypeUnicode).
//
// This file is platform independent: the OS calls go through osAPI, whose Windows
// implementation is in inject_windows.go. Everything else here is covered by unit tests
// that run on any OS.
package inject

import (
	"errors"
	"fmt"
	"slices"
	"time"
	"unicode/utf16"
)

// Win32 constants (winuser.h).
const (
	cfUnicodeText    = 13     // CF_UNICODETEXT
	inputKeyboard    = 1      // INPUT_KEYBOARD
	keyeventfKeyUp   = 0x0002 // KEYEVENTF_KEYUP
	keyeventfUnicode = 0x0004 // KEYEVENTF_UNICODE
	vkControl        = 0x11   // VK_CONTROL
	vkV              = 0x56   // 'V'
)

const (
	openRetries    = 10                     // OpenClipboard retries after the first attempt
	openRetryDelay = 10 * time.Millisecond  // pause between OpenClipboard attempts
	restoreDelay   = 250 * time.Millisecond // time the target app gets to read the clipboard
)

// osAPI is the set of OS operations the paste logic needs.
type osAPI interface {
	// OpenClipboard makes a single attempt to open the clipboard.
	OpenClipboard() error
	// CloseClipboard closes the clipboard opened by OpenClipboard.
	CloseClipboard() error
	// ReadText returns the CF_UNICODETEXT contents without the terminating NUL. ok is false
	// when the clipboard holds no CF_UNICODETEXT data. Requires an open clipboard.
	ReadText() (units []uint16, ok bool, err error)
	// WriteText empties the clipboard and sets CF_UNICODETEXT to units; the implementation
	// adds the terminating NUL. Requires an open clipboard.
	WriteText(units []uint16) error
	// SendInput injects the events and returns how many the system accepted.
	SendInput(in []input) (int, error)
	// Sleep pauses for d.
	Sleep(d time.Duration)
}

// input mirrors the Win32 INPUT structure as used for keyboard events on 64-bit Windows,
// where its size is 40 bytes: type (4) + padding (4) + union at offset 8. The union's
// largest member is MOUSEINPUT (32 bytes), hence the trailing 8 bytes after keybdInput (24).
type input struct {
	inputType uint32
	_         uint32
	ki        keybdInput
	_         [8]byte
}

// keybdInput mirrors KEYBDINPUT (24 bytes on 64-bit Windows).
type keybdInput struct {
	wVk         uint16
	wScan       uint16
	dwFlags     uint32
	time        uint32
	dwExtraInfo uintptr
}

// Paste puts text on the clipboard, sends Ctrl+V to the focused window, and then restores
// the previous CF_UNICODETEXT if the clipboard still holds the pasted text. Empty text is a
// no-op. If the clipboard path fails before Ctrl+V is delivered, Paste types the text with
// TypeUnicode instead.
func Paste(text string) error {
	if text == "" {
		return nil
	}
	api, err := platform()
	if err != nil {
		return err
	}
	return pasteWith(api, text)
}

// TypeUnicode types text into the focused window with SendInput KEYEVENTF_UNICODE events,
// one key-down and one key-up per UTF-16 code unit. Characters outside the BMP are sent as
// their two surrogate code units, which the receiving window combines.
func TypeUnicode(text string) error {
	if text == "" {
		return nil
	}
	api, err := platform()
	if err != nil {
		return err
	}
	return typeWith(api, encodeUTF16(text))
}

func pasteWith(api osAPI, text string) error {
	units := encodeUTF16(text)
	if len(units) == 0 {
		return nil
	}

	var saved []uint16
	var hadText bool
	err := withClipboard(api, func() error {
		var err error
		if saved, hadText, err = api.ReadText(); err != nil {
			return fmt.Errorf("read clipboard: %w", err)
		}
		if err := api.WriteText(units); err != nil {
			return fmt.Errorf("set clipboard text: %w", err)
		}
		return nil
	})
	if err != nil {
		return typeFallback(api, units, fmt.Errorf("paste: %w", err))
	}

	if err := sendInputs(api, ctrlVInputs()); err != nil {
		// Nothing was pasted, so put the old clipboard back now and type the text instead.
		rerr := restoreIfUnchanged(api, units, saved, hadText)
		return typeFallback(api, units, errors.Join(fmt.Errorf("paste: send Ctrl+V: %w", err), rerr))
	}

	api.Sleep(restoreDelay)
	if err := restoreIfUnchanged(api, units, saved, hadText); err != nil {
		return fmt.Errorf("paste: text was pasted but the previous clipboard was not restored: %w", err)
	}
	return nil
}

// typeFallback types units after the clipboard path failed before Ctrl+V was sent. cause is
// reported only together with a failure of the fallback itself.
func typeFallback(api osAPI, units []uint16, cause error) error {
	if err := typeWith(api, units); err != nil {
		return errors.Join(cause, fmt.Errorf("paste: typing fallback: %w", err))
	}
	return nil
}

func typeWith(api osAPI, units []uint16) error {
	if len(units) == 0 {
		return nil
	}
	if err := sendInputs(api, unicodeInputs(units)); err != nil {
		return fmt.Errorf("type unicode: %w", err)
	}
	return nil
}

// restoreIfUnchanged puts saved back on the clipboard, but only if the clipboard still holds
// ours. If the user copied something else in the meantime, or the clipboard has no text now,
// it is left alone. When the previous clipboard had no text, there is nothing to restore.
func restoreIfUnchanged(api osAPI, ours, saved []uint16, hadText bool) error {
	if !hadText {
		return nil
	}
	err := withClipboard(api, func() error {
		cur, ok, err := api.ReadText()
		if err != nil {
			return fmt.Errorf("read clipboard: %w", err)
		}
		if !ok || !slices.Equal(cur, ours) {
			return nil
		}
		if err := api.WriteText(saved); err != nil {
			return fmt.Errorf("write clipboard text: %w", err)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("restore clipboard: %w", err)
	}
	return nil
}

// withClipboard opens the clipboard, runs fn, and always closes it again.
func withClipboard(api osAPI, fn func() error) error {
	if err := openClipboardWithRetry(api); err != nil {
		return err
	}
	ferr := fn()
	if cerr := api.CloseClipboard(); cerr != nil {
		return errors.Join(ferr, fmt.Errorf("close clipboard: %w", cerr))
	}
	return ferr
}

// openClipboardWithRetry tries OpenClipboard once and then up to openRetries more times,
// because another application may hold the clipboard briefly.
func openClipboardWithRetry(api osAPI) error {
	var err error
	for attempt := 0; attempt <= openRetries; attempt++ {
		if err = api.OpenClipboard(); err == nil {
			return nil
		}
		if attempt < openRetries {
			api.Sleep(openRetryDelay)
		}
	}
	return fmt.Errorf("open clipboard (%d attempts): %w", openRetries+1, err)
}

// sendInputs sends in as one SendInput call and checks that every event was accepted.
func sendInputs(api osAPI, in []input) error {
	n, err := api.SendInput(in)
	if err != nil {
		return err
	}
	if n != len(in) {
		return fmt.Errorf("SendInput accepted %d of %d events", n, len(in))
	}
	return nil
}

// ctrlVInputs returns the key-down Ctrl, V down, V up, Ctrl up sequence.
func ctrlVInputs() []input {
	return []input{
		keyEvent(vkControl, 0),
		keyEvent(vkV, 0),
		keyEvent(vkV, keyeventfKeyUp),
		keyEvent(vkControl, keyeventfKeyUp),
	}
}

func keyEvent(vk uint16, flags uint32) input {
	return input{inputType: inputKeyboard, ki: keybdInput{wVk: vk, dwFlags: flags}}
}

// unicodeInputs returns a down and an up KEYEVENTF_UNICODE event for each UTF-16 unit.
func unicodeInputs(units []uint16) []input {
	out := make([]input, 0, 2*len(units))
	for _, u := range units {
		out = append(out,
			input{inputType: inputKeyboard, ki: keybdInput{wScan: u, dwFlags: keyeventfUnicode}},
			input{inputType: inputKeyboard, ki: keybdInput{wScan: u, dwFlags: keyeventfUnicode | keyeventfKeyUp}},
		)
	}
	return out
}

// encodeUTF16 converts s to UTF-16 code units without a terminator. NUL characters are
// dropped because the clipboard stores text as NUL-terminated strings. Invalid UTF-8 becomes
// U+FFFD.
func encodeUTF16(s string) []uint16 {
	units := make([]uint16, 0, len(s))
	for _, r := range s {
		if r == 0 {
			continue
		}
		units = utf16.AppendRune(units, r)
	}
	return units
}
