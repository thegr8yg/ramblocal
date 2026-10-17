//go:build windows

package overlay

import (
	"testing"
	"time"
)

// TestWindowLifecycle needs a Windows desktop session. It creates the real pill, drives every state and level,
// and closes it twice.
func TestWindowLifecycle(t *testing.T) {
	o, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, s := range []State{Listening, Working, Failed, Idle} {
		o.SetState(s)
		o.SetLevel(0.6)
		time.Sleep(20 * time.Millisecond)
	}
	select {
	case <-o.Quit():
		t.Fatal("Quit closed without a menu choice")
	default:
	}
	o.Close()
	o.Close()
}
