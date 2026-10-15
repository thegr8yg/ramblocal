package bench

import (
	"path/filepath"
	"testing"
	"time"
)

func TestGPULock(t *testing.T) {
	p := filepath.Join(t.TempDir(), "gpu.lock")
	release, err := AcquireGPULock(p, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := AcquireGPULock(p, 200*time.Millisecond); err == nil {
		t.Fatal("second acquire should time out")
	}
	release()
	release2, err := AcquireGPULock(p, time.Second)
	if err != nil {
		t.Fatalf("after release: %v", err)
	}
	release2()
}
