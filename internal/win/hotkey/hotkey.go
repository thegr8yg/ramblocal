// Package hotkey detects press and release of one push-to-talk key through a Windows low-level keyboard hook.
package hotkey

import (
	"fmt"
	"sync"
)

type Event int

const (
	Down Event = iota + 1 // key pressed (auto-repeat is filtered out)
	Up                    // key released
)

// eventBuffer is the capacity of the channel returned by Listen. Sends from the hook callback never block.
const eventBuffer = 16

// rawKey is one keyboard transition as seen by the low-level hook.
type rawKey struct {
	VK, Scan, Flags uint32
	Down            bool // false for key-up
	Injected        bool // synthesized by SendInput or similar; never a physical press
}

// ParseKey maps a name such as "f9", "rctrl", "capslock", "fn" or a "vk:0x7A" literal to a virtual-key code.
func ParseKey(name string) (uint32, error) { return parseKey(name) }

// Listen installs the hook for vk on its own locked OS thread. When swallow is true the key never reaches the
// focused application. Events arrive on the channel; call stop to remove the hook and close the channel.
func Listen(vk uint32, swallow bool) (events <-chan Event, stop func(), err error) {
	if vk == 0 || vk > 0xFE {
		return nil, nil, fmt.Errorf("hotkey: invalid virtual-key code 0x%X", vk)
	}
	out := make(chan Event, eventBuffer)
	var tr Tracker // touched only from the hook thread
	stopHook, err := installHook(func(k rawKey) bool {
		if k.Injected || !matches(vk, k.VK) {
			return false
		}
		if ev, ok := tr.Feed(k.Down); ok {
			select {
			case out <- ev:
			default: // consumer is not keeping up; never block inside the hook
			}
		}
		return swallow
	})
	if err != nil {
		return nil, nil, err
	}
	var once sync.Once
	stop = func() {
		once.Do(func() {
			stopHook() // waits for the hook thread to exit, so no send can race the close
			close(out)
		})
	}
	return out, stop, nil
}

// formatKey renders one transition as a single line for Probe.
func formatKey(k rawKey) string {
	state := "up  "
	if k.Down {
		state = "down"
	}
	name := keyName(k.VK)
	if name == "" {
		name = "-"
	}
	inj := ""
	if k.Injected {
		inj = " injected"
	}
	return fmt.Sprintf("%s vk=0x%02X scan=0x%03X flags=0x%02X name=%s%s", state, k.VK, k.Scan, k.Flags, name, inj)
}
