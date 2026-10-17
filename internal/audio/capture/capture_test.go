package capture

import (
	"encoding/binary"
	"reflect"
	"sync"
	"testing"
)

// pcmBytes encodes samples as 16-bit little-endian PCM, the format the device delivers.
func pcmBytes(samples ...int16) []byte {
	b := make([]byte, 2*len(samples))
	for i, s := range samples {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(s))
	}
	return b
}

func TestPCM16LE(t *testing.T) {
	got := pcm16LE(nil, pcmBytes(-32768, -1, 0, 16384, 32767))
	want := []float32{-1, -1.0 / 32768, 0, 0.5, 32767.0 / 32768}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	// Explicit little-endian byte order: 0x4000 is 16384.
	if got := pcm16LE(nil, []byte{0x00, 0x40}); !reflect.DeepEqual(got, []float32{0.5}) {
		t.Fatalf("little-endian decode: got %v", got)
	}
	// A trailing odd byte is ignored.
	if got := pcm16LE(nil, []byte{0x00, 0x40, 0x7f}); len(got) != 1 {
		t.Fatalf("odd trailing byte: got %d samples", len(got))
	}
}

func TestRecorderPipeline(t *testing.T) {
	r := &Recorder{acc: newAccumulator()}
	block := pcmBytes(make([]int16, bufferSamples)...)
	for i := 0; i < 50; i++ { // 1 s of silence fills the pre-roll
		r.onPCM(block)
	}
	if lvl := r.Level(); lvl != 0 {
		t.Fatalf("level after silence = %v", lvl)
	}

	r.Begin()
	loud := make([]int16, bufferSamples)
	for i := range loud {
		loud[i] = 16384 // 0.5, about -6 dB
	}
	for i := 0; i < 10; i++ { // 200 ms of speech
		r.onPCM(pcmBytes(loud...))
	}
	if lvl := r.Level(); lvl < 0.5 {
		t.Fatalf("level during speech = %v, want > 0.5", lvl)
	}
	out := r.End()
	if want := prerollSamples + 10*bufferSamples; len(out) != want {
		t.Fatalf("utterance length = %d, want %d", len(out), want)
	}
	if out[len(out)-1] != 0.5 || out[0] != 0 {
		t.Fatalf("utterance ends are wrong: first %v last %v", out[0], out[len(out)-1])
	}
}

type fakeDevice struct {
	mu     sync.Mutex
	closes int
}

func (f *fakeDevice) close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closes++
	return nil
}

func TestCloseIsIdempotent(t *testing.T) {
	dev := &fakeDevice{}
	r := &Recorder{acc: newAccumulator(), dev: dev}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if dev.closes != 1 {
		t.Fatalf("device closed %d times", dev.closes)
	}
}

func TestConcurrentLevelAndUtterance(t *testing.T) {
	r := &Recorder{acc: newAccumulator()}
	block := pcmBytes(make([]int16, bufferSamples)...)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { // device goroutine
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			r.onPCM(block)
		}
	}()
	go func() { // hotkey and overlay goroutine
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			_ = r.Level()
			r.Begin()
			_ = r.End()
		}
	}()
	wg.Wait()
}
