package golden

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/thegr8yg/ramblocal/internal/audio"
)

func writeF32(t *testing.T, path string, v []float32) {
	t.Helper()
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, v)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadSyntheticFixture(t *testing.T) {
	dir := t.TempDir()
	var wav bytes.Buffer
	audio.WriteWAV16(&wav, []float32{0, 0.5}, 16000)
	os.WriteFile(filepath.Join(dir, "c1.wav"), wav.Bytes(), 0o644)
	writeF32(t, filepath.Join(dir, "c1.mel.f32"), []float32{1, 2, 3, 4, 5, 6})
	writeF32(t, filepath.Join(dir, "c1.enc.f32"), []float32{7, 8})
	os.WriteFile(filepath.Join(dir, "c1.json"),
		[]byte(`{"tokens":[5,6],"text":"hi","mel_shape":[2,3],"enc_shape":[2,1]}`), 0o644)

	names, err := List(dir)
	if err != nil || len(names) != 1 || names[0] != "c1" {
		t.Fatalf("List = %v, %v", names, err)
	}
	f, err := Load(dir, "c1")
	if err != nil {
		t.Fatal(err)
	}
	if f.Text != "hi" || len(f.PCM) != 2 || len(f.Mel) != 6 || f.Enc[1] != 8 || f.Tokens[1] != 6 {
		t.Fatalf("fixture = %+v", f)
	}

	writeF32(t, filepath.Join(dir, "c1.enc.f32"), []float32{7})
	if _, err := Load(dir, "c1"); err == nil {
		t.Fatal("size mismatch: expected error")
	}
}

func TestMaxAbsDiff(t *testing.T) {
	if d := MaxAbsDiff([]float32{1, 2, 3}, []float32{1, 2.5, 2}); d != 1 {
		t.Fatalf("MaxAbsDiff = %v", d)
	}
}
