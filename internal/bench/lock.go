package bench

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"time"
)

const staleLock = 2 * time.Hour

// lockToken returns a per-acquisition token: pid plus a random nonce.
func lockToken() (string, error) {
	var nonce [8]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return "", err
	}
	return fmt.Sprintf("%d-%s", os.Getpid(), hex.EncodeToString(nonce[:])), nil
}

// AcquireGPULock serializes GPU benchmarks across processes using an exclusive lock file.
// The returned release removes the file only if it still holds this acquisition's token.
func AcquireGPULock(path string, timeout time.Duration) (func(), error) {
	token, err := lockToken()
	if err != nil {
		return nil, err
	}
	deadline := time.Now().Add(timeout)
	for {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			_, werr := fmt.Fprintf(f, "%s\n", token)
			cerr := f.Close()
			if werr != nil || cerr != nil {
				os.Remove(path)
				return nil, errors.Join(werr, cerr)
			}
			return func() {
				if b, err := os.ReadFile(path); err == nil && string(b) == token+"\n" {
					os.Remove(path)
				}
			}, nil
		}
		// Windows reports access-denied while a deleted lock file is still pending removal; treat it as busy.
		if !errors.Is(err, os.ErrExist) && !errors.Is(err, os.ErrPermission) {
			return nil, err
		}
		if st, serr := os.Stat(path); serr == nil && time.Since(st.ModTime()) > staleLock {
			// Rename is the atomic claim: only one contender can move the stale file aside.
			aside := path + ".stale-" + token
			if os.Rename(path, aside) == nil {
				os.Remove(aside)
				continue
			}
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("GPU lock %s held by another benchmark", path)
		}
		time.Sleep(100 * time.Millisecond)
	}
}
