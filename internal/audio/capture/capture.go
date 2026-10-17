// Package capture records the default microphone as 16 kHz mono float32.
//
// The microphone opens in Open and keeps running. The device goroutine feeds a rolling 300 ms
// pre-roll; Begin starts an utterance seeded with that pre-roll, and End returns it.
package capture

import (
	"sync"
)

// device is the platform microphone. close stops it and returns once onPCM will not be called again.
type device interface {
	close() error
}

// Recorder is an open microphone. It is safe for concurrent use by Begin, End, Level and Close.
type Recorder struct {
	acc     *accumulator
	level   meter
	scratch []float32 // converted samples; touched only by the device goroutine
	dev     device

	closeOnce sync.Once
	closeErr  error
}

// Open starts the default microphone at 16 kHz mono and keeps it running until Close.
// On platforms other than Windows it returns an error.
func Open() (*Recorder, error) {
	r := &Recorder{acc: newAccumulator()}
	dev, err := openDevice(r.onPCM)
	if err != nil {
		return nil, err
	}
	r.dev = dev
	return r, nil
}

// Begin marks the start of an utterance, including the pre-roll. Calling it again restarts the utterance.
func (r *Recorder) Begin() {
	r.acc.begin()
}

// End returns all samples since Begin (16 kHz mono, -1..1) and stops collecting them. The caller owns
// the returned slice. Samples past five minutes are dropped. End returns nil if Begin was not called.
func (r *Recorder) End() []float32 {
	return r.acc.end()
}

// Level is the recent RMS level, 0..1, for the overlay meter.
func (r *Recorder) Level() float32 {
	return r.level.get()
}

// Close stops the microphone. It is safe to call more than once.
func (r *Recorder) Close() error {
	r.closeOnce.Do(func() {
		if r.dev != nil {
			r.closeErr = r.dev.close()
		}
	})
	return r.closeErr
}

// onPCM receives raw 16-bit little-endian mono PCM from the device goroutine. raw is only valid for
// the duration of the call.
func (r *Recorder) onPCM(raw []byte) {
	r.scratch = pcm16LE(r.scratch[:0], raw)
	r.acc.feed(r.scratch)
	r.level.update(r.scratch)
}

// pcm16LE appends the little-endian 16-bit PCM samples in src to dst, scaled to [-1, 1).
// A trailing odd byte is ignored.
func pcm16LE(dst []float32, src []byte) []float32 {
	for i := 0; i+1 < len(src); i += 2 {
		v := int16(uint16(src[i]) | uint16(src[i+1])<<8)
		dst = append(dst, float32(v)/32768)
	}
	return dst
}
