package hotkey

import (
	"strings"
	"testing"
)

func TestParseKey(t *testing.T) {
	ok := map[string]uint32{
		"f1": 0x70, "f9": 0x78, "f12": 0x7B, "f24": 0x87,
		"ctrl": 0x11, "lctrl": 0xA2, "rctrl": 0xA3,
		"alt": 0x12, "lalt": 0xA4, "ralt": 0xA5,
		"shift": 0x10, "lshift": 0xA0, "rshift": 0xA1,
		"lwin": 0x5B, "rwin": 0x5C,
		"capslock": 0x14, "scrolllock": 0x91, "pause": 0x13, "insert": 0x2D,
		"home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
		"a": 0x41, "f": 0x46, "m": 0x4D, "z": 0x5A,
		"0": 0x30, "9": 0x39,
		"vk:0x7A": 0x7A, "vk:122": 0x7A, "VK:0X7a": 0x7A, "vk:0x01": 0x01, "vk:0xFE": 0xFE,
		"RCTRL": 0xA3, "  F9 ": 0x78, "CapsLock": 0x14,
	}
	for name, want := range ok {
		got, err := ParseKey(name)
		if err != nil {
			t.Errorf("ParseKey(%q) error: %v", name, err)
			continue
		}
		if got != want {
			t.Errorf("ParseKey(%q) = 0x%02X, want 0x%02X", name, got, want)
		}
	}

	bad := []string{"", "  ", "f0", "f25", "f1x", "f+5", "aa", "-", "bogus",
		"vk:", "vk:0", "vk:0x00", "vk:0xFF", "vk:0x100", "vk:zz"}
	for _, name := range bad {
		if _, err := ParseKey(name); err == nil {
			t.Errorf("ParseKey(%q) succeeded, want error", name)
		}
	}
}

func TestParseKeyFnPointsAtDiagnostic(t *testing.T) {
	_, err := ParseKey("fn")
	if err == nil {
		t.Fatal(`ParseKey("fn") succeeded, want error`)
	}
	for _, want := range []string{"ramblocal keys", "rctrl", "f9"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("fn error %q does not mention %q", err, want)
		}
	}
	if _, err := ParseKey("FN"); err == nil {
		t.Error(`ParseKey("FN") succeeded, want error`)
	}
}

func TestKeyNameRoundTrip(t *testing.T) {
	names := []string{"f1", "f24", "rctrl", "lctrl", "ctrl", "ralt", "lshift", "rwin", "capslock",
		"scrolllock", "pause", "insert", "home", "end", "pageup", "pagedown", "a", "z", "0", "9"}
	for _, name := range names {
		vk, err := ParseKey(name)
		if err != nil {
			t.Fatalf("ParseKey(%q): %v", name, err)
		}
		if got := keyName(vk); got != name {
			t.Errorf("keyName(0x%02X) = %q, want %q", vk, got, name)
		}
	}
	if got := keyName(0x07); got != "" {
		t.Errorf("keyName(0x07) = %q, want empty", got)
	}
}

func TestMatches(t *testing.T) {
	cases := []struct {
		target, vk uint32
		want       bool
	}{
		{0x78, 0x78, true},
		{0x78, 0x79, false},
		{0xA3, 0xA3, true},
		{0xA3, 0xA2, false}, // rctrl does not match lctrl
		{0x11, 0xA2, true},  // generic ctrl matches lctrl
		{0x11, 0xA3, true},  // generic ctrl matches rctrl
		{0x11, 0x11, true},  // exact
		{0x11, 0x5B, false}, // not a modifier family member
		{0x10, 0xA1, true},  // generic shift
		{0x12, 0xA5, true},  // generic alt
		{0xA5, 0x12, false}, // specific target does not match generic code
		{0xA5, 0xA4, false}, // ralt does not match lalt
	}
	for _, c := range cases {
		if got := matches(c.target, c.vk); got != c.want {
			t.Errorf("matches(0x%02X, 0x%02X) = %v, want %v", c.target, c.vk, got, c.want)
		}
	}
}
