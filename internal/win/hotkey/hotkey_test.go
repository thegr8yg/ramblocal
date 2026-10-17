package hotkey

import (
	"runtime"
	"strings"
	"testing"
)

func TestFormatKey(t *testing.T) {
	got := formatKey(rawKey{VK: 0xA3, Scan: 0x11D, Flags: 0x01, Down: true})
	want := "down vk=0xA3 scan=0x11D flags=0x01 name=rctrl"
	if got != want {
		t.Errorf("formatKey = %q, want %q", got, want)
	}
	got = formatKey(rawKey{VK: 0xFF, Scan: 0x00, Flags: 0x90, Down: false, Injected: true})
	if !strings.HasPrefix(got, "up   vk=0xFF") || !strings.HasSuffix(got, "name=- injected") {
		t.Errorf("formatKey unknown key = %q", got)
	}
}

func TestListenRejectsBadVK(t *testing.T) {
	for _, vk := range []uint32{0, 0xFF, 0x1000} {
		if _, stop, err := Listen(vk, false); err == nil {
			stop()
			t.Errorf("Listen(0x%X) succeeded, want error", vk)
		}
	}
}

func TestListenUnsupportedOffWindows(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("hook is available on Windows; not exercised here")
	}
	if _, _, err := Listen(0x7A, false); err == nil {
		t.Error("Listen succeeded on a non-Windows platform")
	}
}
