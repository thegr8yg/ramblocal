// Package hotkey detects press and release of one push-to-talk key through a Windows low-level keyboard hook.
package hotkey

import "errors"

type Event int

const (
	Down Event = iota + 1 // key pressed (auto-repeat is filtered out)
	Up                    // key released
)

var errNotImplemented = errors.New("hotkey: not implemented")

// ParseKey maps a name such as "f9", "rctrl", "capslock", "fn" or a "vk:0x7A" literal to a virtual-key code.
func ParseKey(name string) (uint32, error) { return 0, errNotImplemented }

// Listen installs the hook for vk on its own locked OS thread. When swallow is true the key never reaches the
// focused application. Events arrive on the channel; call stop to remove the hook.
func Listen(vk uint32, swallow bool) (events <-chan Event, stop func(), err error) {
	return nil, nil, errNotImplemented
}
