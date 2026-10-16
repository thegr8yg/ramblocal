package bench

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestReleaseKeepsForeignLock(t *testing.T) {
	p := filepath.Join(t.TempDir(), "gpu.lock")
	release, err := AcquireGPULock(p, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("someone-else\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	release()
	b, err := os.ReadFile(p)
	if err != nil || string(b) != "someone-else\n" {
		t.Fatalf("release removed a lock it does not own: %q %v", b, err)
	}
}

func TestStaleLockTakeover(t *testing.T) {
	p := filepath.Join(t.TempDir(), "gpu.lock")
	if err := os.WriteFile(p, []byte("crashed-run\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-3 * time.Hour)
	if err := os.Chtimes(p, old, old); err != nil {
		t.Fatal(err)
	}
	release, err := AcquireGPULock(p, time.Second)
	if err != nil {
		t.Fatalf("stale lock not taken over: %v", err)
	}
	release()
}

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
