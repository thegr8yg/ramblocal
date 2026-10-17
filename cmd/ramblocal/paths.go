package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

const defaultModelName = "parakeet-tdt-0.6b-v2"

// findModelDir resolves the model directory: flag, then RAMBLOCAL_MODEL_DIR, then models/<default>
// next to the executable and in the working directory. A directory counts only if it has manifest.json.
func findModelDir(flagVal string) (string, error) {
	return firstExisting("model directory", "RAMBLOCAL_MODEL_DIR", flagVal, "manifest.json",
		candidates(filepath.Join("models", defaultModelName)))
}

// findORTLib resolves the ONNX Runtime shared library: flag, then RAMBLOCAL_ORT_LIB, then the
// platform library name next to the executable and in the working directory.
func findORTLib(flagVal string) (string, error) {
	name := "libonnxruntime.so"
	switch runtime.GOOS {
	case "windows":
		name = "onnxruntime.dll"
	case "darwin":
		name = "libonnxruntime.dylib"
	}
	return firstExisting("ONNX Runtime library", "RAMBLOCAL_ORT_LIB", flagVal, "", candidates(name))
}

func candidates(rel string) []string {
	var out []string
	if exe, err := os.Executable(); err == nil {
		out = append(out, filepath.Join(filepath.Dir(exe), rel))
	}
	return append(out, rel)
}

func firstExisting(what, env, flagVal, mustContain string, defaults []string) (string, error) {
	var tried []string
	for _, p := range append([]string{flagVal, os.Getenv(env)}, defaults...) {
		if p == "" {
			continue
		}
		check := p
		if mustContain != "" {
			check = filepath.Join(p, mustContain)
		}
		if _, err := os.Stat(check); err == nil {
			return p, nil
		}
		tried = append(tried, p)
	}
	return "", fmt.Errorf("%s not found (set the flag or %s); tried %v", what, env, tried)
}
