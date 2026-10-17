package capture

import "sync"

const (
	sampleRate = 16000

	// prerollSamples is the audio kept before a key press so the first word is not clipped.
	prerollSamples = sampleRate * 300 / 1000

	// bufferSamples is one 20 ms device buffer; the Windows device queues several of them.
	bufferSamples = sampleRate * 20 / 1000

	// maxUtteranceSamples caps one utterance, pre-roll included, at five minutes.
	maxUtteranceSamples = sampleRate * 5 * 60
)

// ring keeps the most recent len(buf) samples in arrival order. The zero value is not usable; use newRing.
type ring struct {
	buf  []float32
	next int  // slot that the next sample overwrites
	full bool // true once the buffer has wrapped at least once
}

func newRing(n int) ring {
	return ring{buf: make([]float32, n)}
}

// push appends s, overwriting the oldest samples when the buffer is full.
func (r *ring) push(s []float32) {
	if len(s) >= len(r.buf) {
		copy(r.buf, s[len(s)-len(r.buf):])
		r.next = 0
		r.full = true
		return
	}
	n := copy(r.buf[r.next:], s)
	if n < len(s) {
		copy(r.buf, s[n:])
	}
	r.next += len(s)
	if r.next >= len(r.buf) {
		r.next -= len(r.buf)
		r.full = true
	}
}

// appendTo appends the buffered samples to dst, oldest first, and returns the extended slice.
func (r *ring) appendTo(dst []float32) []float32 {
	if !r.full {
		return append(dst, r.buf[:r.next]...)
	}
	dst = append(dst, r.buf[r.next:]...)
	return append(dst, r.buf[:r.next]...)
}

// accumulator holds the rolling pre-roll and the utterance being recorded. feed is called from the
// capture goroutine; begin, end and feed may run concurrently.
type accumulator struct {
	mu        sync.Mutex
	pre       ring
	recording bool
	utt       []float32
}

func newAccumulator() *accumulator {
	return &accumulator{pre: newRing(prerollSamples)}
}

// feed adds samples to the pre-roll and, while an utterance is open, to the utterance. Samples beyond
// the five-minute cap are dropped.
func (a *accumulator) feed(s []float32) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pre.push(s)
	if !a.recording {
		return
	}
	room := maxUtteranceSamples - len(a.utt)
	if room <= 0 {
		return
	}
	if len(s) > room {
		s = s[:room]
	}
	a.utt = append(a.utt, s...)
}

// begin starts an utterance seeded with the current pre-roll. If one is already open it is discarded.
func (a *accumulator) begin() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.utt = a.pre.appendTo(nil)
	a.recording = true
}

// end closes the open utterance and returns it. The returned slice is owned by the caller: the
// accumulator drops its reference, so later feeds cannot change it. It returns nil if nothing was open.
func (a *accumulator) end() []float32 {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.recording {
		return nil
	}
	out := a.utt
	a.utt = nil
	a.recording = false
	return out
}
