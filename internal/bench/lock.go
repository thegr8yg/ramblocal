package bench

import (
	"errors"
	"fmt"
	"os"
	"time"
)

const staleLock = 2 * time.Hour

// AcquireGPULock serializes GPU benchmarks across processes using an exclusive lock file.
func AcquireGPULock(path string, timeout time.Duration) (func(), error) {
	deadline := time.Now().Add(timeout)
	for {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			fmt.Fprintf(f, "%d\n", os.Getpid())
			f.Close()
			return func() { os.Remove(path) }, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, err
		}
		if st, serr := os.Stat(path); serr == nil && time.Since(st.ModTime()) > staleLock {
			os.Remove(path)
			continue
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("GPU lock %s held by another benchmark", path)
		}
		time.Sleep(100 * time.Millisecond)
	}
}
