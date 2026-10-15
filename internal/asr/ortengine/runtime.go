// Package ortengine implements asr.Engine on ONNX Runtime.
package ortengine

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	initOnce sync.Once
	initErr  error
)

// InitRuntime loads onnxruntime.dll once per process.
func InitRuntime(libPath string) error {
	initOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

func sessionOptions(ep string) (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	switch ep {
	case "cuda":
		c, err := ort.NewCUDAProviderOptions()
		if err != nil {
			opts.Destroy()
			return nil, err
		}
		defer c.Destroy()
		if err := opts.AppendExecutionProviderCUDA(c); err != nil {
			opts.Destroy()
			return nil, err
		}
	case "dml":
		if err := opts.AppendExecutionProviderDirectML(0); err != nil {
			opts.Destroy()
			return nil, err
		}
	case "cpu":
	default:
		opts.Destroy()
		return nil, fmt.Errorf("unknown execution provider %q", ep)
	}
	return opts, nil
}
