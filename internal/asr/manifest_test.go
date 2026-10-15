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
