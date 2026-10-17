package hotkey

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var errFn = errors.New(`hotkey: the fn key is normally handled in keyboard firmware and is not visible to Windows keyboard hooks; ` +
	`run "ramblocal keys" and press Fn to see what, if anything, your keyboard reports, or use a key such as rctrl or f9 instead`)

// namedKeys are the spelled-out names. Left/right variants are the exact VK_L*/VK_R* codes the low-level hook reports.
// The generic ctrl/alt/shift entries also match either side (see matches).
var namedKeys = []struct {
	name string
	vk   uint32
}{
	{"ctrl", 0x11}, {"lctrl", 0xA2}, {"rctrl", 0xA3},
	{"alt", 0x12}, {"lalt", 0xA4}, {"ralt", 0xA5},
	{"shift", 0x10}, {"lshift", 0xA0}, {"rshift", 0xA1},
	{"lwin", 0x5B}, {"rwin", 0x5C},
	{"capslock", 0x14}, {"scrolllock", 0x91}, {"pause", 0x13},
	{"insert", 0x2D}, {"home", 0x24}, {"end", 0x23}, {"pageup", 0x21}, {"pagedown", 0x22},
}

func parseKey(name string) (uint32, error) {
	s := strings.ToLower(strings.TrimSpace(name))
	switch {
	case s == "":
		return 0, errors.New("hotkey: empty key name")
	case s == "fn":
		return 0, errFn
	case strings.HasPrefix(s, "vk:"):
		v, err := strconv.ParseUint(s[len("vk:"):], 0, 8)
		if err != nil || v == 0 || v == 0xFF {
			return 0, fmt.Errorf("hotkey: bad virtual-key literal %q (want vk:0x01 through vk:0xFE)", name)
		}
		return uint32(v), nil
	case len(s) >= 2 && len(s) <= 3 && s[0] == 'f' && isDigits(s[1:]):
		if n, _ := strconv.Atoi(s[1:]); n >= 1 && n <= 24 {
			return 0x70 + uint32(n-1), nil // VK_F1 is 0x70
		}
	case len(s) == 1 && s[0] >= 'a' && s[0] <= 'z':
		return 0x41 + uint32(s[0]-'a'), nil // VK_A..VK_Z
	case len(s) == 1 && s[0] >= '0' && s[0] <= '9':
		return 0x30 + uint32(s[0]-'0'), nil // VK_0..VK_9
	}
	for _, k := range namedKeys {
		if k.name == s {
			return k.vk, nil
		}
	}
	return 0, fmt.Errorf("hotkey: unknown key %q (try f9, rctrl, capslock, or vk:0xNN; run \"ramblocal keys\" to find codes)", name)
}

// keyName returns the canonical name for vk, or "" if it has none.
func keyName(vk uint32) string {
	for _, k := range namedKeys {
		if k.vk == vk {
			return k.name
		}
	}
	switch {
	case vk >= 0x70 && vk <= 0x87:
		return fmt.Sprintf("f%d", vk-0x70+1)
	case vk >= 0x41 && vk <= 0x5A:
		return string(rune('a' + vk - 0x41))
	case vk >= 0x30 && vk <= 0x39:
		return string(rune('0' + vk - 0x30))
	}
	return ""
}

// matches reports whether a transition for raw key vk is the configured target.
func matches(target, vk uint32) bool {
	if vk == target {
		return true
	}
	switch target {
	case 0x10: // shift
		return vk == 0xA0 || vk == 0xA1
	case 0x11: // ctrl
		return vk == 0xA2 || vk == 0xA3
	case 0x12: // alt
		return vk == 0xA4 || vk == 0xA5
	}
	return false
}

func isDigits(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return s != ""
}
