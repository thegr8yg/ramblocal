package asr

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const validManifest = `{
  "name": "test", "sample_rate": 16000, "num_features": 128, "hop_seconds": 0.01,
  "subsampling": 8, "encoder_dim": 1024, "vocab_size": 1024, "blank_id": 1024,
  "tdt_durations": [0,1,2,3,4], "pred_layers": 2, "pred_hidden": 640,
  "preprocessor": {"file":"p.onnx","inputs":["a","b"],"outputs":["c","d"],"dtypes":{}},
  "encoder": {"file":"e.onnx","inputs":["a","b"],"outputs":["c","d"],"dtypes":{}},
  "decoder_joint": {"file":"d.onnx","inputs":["a","b","c","d","e"],"outputs":["f","g","h","i"],"dtypes":{}}
}`

func writeManifest(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadManifestValid(t *testing.T) {
	m, err := LoadManifest(writeManifest(t, validManifest))
	if err != nil {
		t.Fatal(err)
	}
	if m.BlankID != 1024 || len(m.Durations) != 5 || m.DecoderJoint.File != "d.onnx" {
		t.Fatalf("unexpected manifest: %+v", m)
	}
	if got := m.FrameSeconds(); got < 0.0799 || got > 0.0801 {
		t.Fatalf("FrameSeconds = %v, want 0.08", got)
	}
}

func TestLoadManifestRejects(t *testing.T) {
	cases := map[string]struct{ from, to string }{
		"blank not last":     {`"blank_id": 1024`, `"blank_id": 0`},
		"no durations":       {`[0,1,2,3,4]`, `[]`},
		"wrong sample rate":  {`"sample_rate": 16000`, `"sample_rate": 8000`},
		"joint inputs short": {`["a","b","c","d","e"]`, `["a","b"]`},
		"zero subsampling":   {`"subsampling": 8`, `"subsampling": 0`},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := LoadManifest(writeManifest(t, strings.Replace(validManifest, c.from, c.to, 1)))
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

const frontendJSON = `, "frontend": {"n_fft": 512, "win_length": 400, "hop_length": 160, "preemph": 0.97,
  "log_guard": 5.960464477539063e-08, "norm_eps": 1e-05, "filterbank": "fb.f32", "window": "w.f32"}
}`

func withFrontend(body string) string {
	return strings.TrimSuffix(strings.TrimSpace(body), "}") + frontendJSON
}

func TestLoadManifestFrontendAbsent(t *testing.T) {
	m, err := LoadManifest(writeManifest(t, validManifest))
	if err != nil {
		t.Fatal(err)
	}
	if m.Frontend != nil {
		t.Fatalf("Frontend = %+v, want nil", m.Frontend)
	}
}

func TestLoadManifestFrontendValid(t *testing.T) {
	m, err := LoadManifest(writeManifest(t, withFrontend(validManifest)))
	if err != nil {
		t.Fatal(err)
	}
	f := m.Frontend
	if f == nil || f.NFFT != 512 || f.WinLength != 400 || f.HopLength != 160 || f.Preemph != 0.97 ||
		f.LogGuard != 5.960464477539063e-08 || f.NormEps != 1e-05 || f.Filterbank != "fb.f32" || f.Window != "w.f32" {
		t.Fatalf("unexpected frontend: %+v", f)
	}
}

func TestLoadManifestFrontendRejects(t *testing.T) {
	cases := map[string]struct{ from, to string }{
		"n_fft not power of two": {`"n_fft": 512`, `"n_fft": 500`},
		"n_fft one":              {`"n_fft": 512`, `"n_fft": 1`},
		"win_length zero":        {`"win_length": 400`, `"win_length": 0`},
		"win_length > n_fft":     {`"win_length": 400`, `"win_length": 513`},
		"hop zero":               {`"hop_length": 160`, `"hop_length": 0`},
		"log_guard zero":         {`"log_guard": 5.960464477539063e-08`, `"log_guard": 0`},
		"norm_eps negative":      {`"norm_eps": 1e-05`, `"norm_eps": -1`},
		"filterbank empty":       {`"filterbank": "fb.f32"`, `"filterbank": ""`},
		"window empty":           {`"window": "w.f32"`, `"window": ""`},
	}
	base := withFrontend(validManifest)
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(base, c.from) {
				t.Fatalf("test bug: %q not in manifest", c.from)
			}
			_, err := LoadManifest(writeManifest(t, strings.Replace(base, c.from, c.to, 1)))
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
