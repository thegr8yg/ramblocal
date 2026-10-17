package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
)

// fakeRecorder returns fixed samples from End and counts calls.
type fakeRecorder struct {
	mu                     sync.Mutex
	samples                []float32
	begins, ends, levelCnt int
}

func (r *fakeRecorder) Begin() { r.mu.Lock(); r.begins++; r.mu.Unlock() }

func (r *fakeRecorder) End() []float32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ends++
	return r.samples
}

func (r *fakeRecorder) Level() float32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.levelCnt++
	return 0.5
}

func (r *fakeRecorder) counts() (begins, ends int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.begins, r.ends
}

type ovlEvent struct {
	state   State
	level   float32
	isLevel bool
}

// fakeOverlay records every SetState and SetLevel call in order.
type fakeOverlay struct {
	mu     sync.Mutex
	events []ovlEvent
	quit   chan struct{}
}

func (o *fakeOverlay) SetState(s State) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.events = append(o.events, ovlEvent{state: s})
}

func (o *fakeOverlay) SetLevel(l float32) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.events = append(o.events, ovlEvent{level: l, isLevel: true})
}

func (o *fakeOverlay) Quit() <-chan struct{} { return o.quit }

func (o *fakeOverlay) snapshot() []ovlEvent {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]ovlEvent(nil), o.events...)
}

// lastState is StateIdle until the first SetState, which matches the overlay's starting state.
func (o *fakeOverlay) lastState() State {
	last := StateIdle
	for _, e := range o.snapshot() {
		if !e.isLevel {
			last = e.state
		}
	}
	return last
}

// stateSeq returns the sequence of states with consecutive repeats removed.
func (o *fakeOverlay) stateSeq() []State {
	var out []State
	for _, e := range o.snapshot() {
		if e.isLevel {
			continue
		}
		if len(out) == 0 || out[len(out)-1] != e.state {
			out = append(out, e.state)
		}
	}
	return out
}

func (o *fakeOverlay) levelCount() int {
	n := 0
	for _, e := range o.snapshot() {
		if e.isLevel {
			n++
		}
	}
	return n
}

type fakeInjector struct {
	mu    sync.Mutex
	err   error
	texts []string
}

func (f *fakeInjector) Paste(text string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return f.err
	}
	f.texts = append(f.texts, text)
	return nil
}

func (f *fakeInjector) pasted() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.texts...)
}

// gateEngine blocks in Transcribe until gate is closed or the context ends, and signals entered when it starts.
type gateEngine struct {
	gate    chan struct{}
	entered chan struct{}
	text    string
}

func (g *gateEngine) Transcribe(ctx context.Context, pcm []float32, opts asr.Options) (asr.Result, error) {
	select {
	case g.entered <- struct{}{}:
	default:
	}
	select {
	case <-g.gate:
		return asr.Result{Text: g.text}, nil
	case <-ctx.Done():
		return asr.Result{}, ctx.Err()
	}
}

// syncBuffer is a bytes.Buffer safe for use from the session goroutine and the test.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

type harness struct {
	t    *testing.T
	s    *session
	rec  *fakeRecorder
	ovl  *fakeOverlay
	inj  *fakeInjector
	keys chan KeyEvent
	log  *syncBuffer

	cancel context.CancelFunc
	errc   chan error
}

func newHarness(t *testing.T, eng Engine, samples []float32, quit chan struct{}) *harness {
	t.Helper()
	h := &harness{
		t:    t,
		rec:  &fakeRecorder{samples: samples},
		ovl:  &fakeOverlay{quit: quit},
		inj:  &fakeInjector{},
		keys: make(chan KeyEvent),
		log:  &syncBuffer{},
	}
	h.s = &session{
		recorder:      h.rec,
		overlay:       h.ovl,
		injector:      h.inj,
		engine:        eng,
		keys:          h.keys,
		log:           h.log,
		minUtterance:  minUtterance,
		levelInterval: time.Millisecond,
		failFlash:     20 * time.Millisecond,
	}
	return h
}

func (h *harness) start() {
	ctx, cancel := context.WithCancel(context.Background())
	h.cancel = cancel
	h.errc = make(chan error, 1)
	go func() { h.errc <- h.s.run(ctx) }()
	h.t.Cleanup(cancel)
}

// wait returns run's result, failing the test if run does not return promptly.
func (h *harness) wait() error {
	h.t.Helper()
	select {
	case err := <-h.errc:
		return err
	case <-time.After(2 * time.Second):
		h.t.Fatal("session did not shut down")
		return nil
	}
}

func (h *harness) stop() error {
	h.cancel()
	return h.wait()
}

func (h *harness) press()   { h.keys <- KeyDown }
func (h *harness) release() { h.keys <- KeyUp }

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

func seconds(n float64) []float32 {
	return make([]float32, int(n*asr.SampleRate))
}

func equalStates(a, b []State) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestSessionTapShorterThanThresholdIgnored(t *testing.T) {
	eng := &asr.FakeEngine{}
	h := newHarness(t, eng, seconds(0.1), nil)
	h.start()

	h.press()
	h.release()
	waitFor(t, "tap to end", func() bool {
		_, ends := h.rec.counts()
		return ends == 1 && h.ovl.lastState() == StateIdle
	})

	if got := h.inj.pasted(); len(got) != 0 {
		t.Fatalf("pasted %q for an accidental tap", got)
	}
	if n := eng.Calls(); n != 0 {
		t.Fatalf("engine ran %d times for an accidental tap", n)
	}
	if want := []State{StateListening, StateIdle}; !equalStates(h.ovl.stateSeq(), want) {
		t.Fatalf("states %v, want %v", h.ovl.stateSeq(), want)
	}
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionUtterancePastedWithTrailingSpace(t *testing.T) {
	eng := &asr.FakeEngine{Fn: func([]float32, asr.Options) (asr.Result, error) {
		return asr.Result{Text: "  hello world \n"}, nil
	}}
	h := newHarness(t, eng, seconds(1), nil)
	h.start()

	h.press()
	h.release()
	waitFor(t, "paste and idle", func() bool {
		return len(h.inj.pasted()) == 1 && h.ovl.lastState() == StateIdle
	})

	if got := h.inj.pasted(); len(got) != 1 || got[0] != "hello world " {
		t.Fatalf("pasted %q, want [\"hello world \"]", got)
	}
	if want := []State{StateListening, StateWorking, StateIdle}; !equalStates(h.ovl.stateSeq(), want) {
		t.Fatalf("states %v, want %v", h.ovl.stateSeq(), want)
	}
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(h.log.String(), "release-to-paste=") {
		t.Fatalf("no latency line in log:\n%s", h.log.String())
	}
}

func TestSessionEmptyTranscriptNotPasted(t *testing.T) {
	h := newHarness(t, &asr.FakeEngine{}, seconds(1), nil) // nil Fn: empty text
	h.start()

	h.press()
	h.release()
	waitFor(t, "idle", func() bool {
		_, ends := h.rec.counts()
		return ends == 1 && h.ovl.lastState() == StateIdle
	})
	if got := h.inj.pasted(); len(got) != 0 {
		t.Fatalf("pasted %q for silence", got)
	}
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionEngineErrorShowsFailedAndDoesNotPaste(t *testing.T) {
	eng := &asr.FakeEngine{Fn: func([]float32, asr.Options) (asr.Result, error) {
		return asr.Result{}, errors.New("boom")
	}}
	h := newHarness(t, eng, seconds(1), nil)
	h.start()

	h.press()
	h.release()
	waitFor(t, "failed flash to end", func() bool {
		return h.ovl.lastState() == StateIdle && equalStates(h.ovl.stateSeq(), []State{StateListening, StateWorking, StateFailed, StateIdle})
	})

	if got := h.inj.pasted(); len(got) != 0 {
		t.Fatalf("pasted %q after engine error", got)
	}
	if !strings.Contains(h.log.String(), "boom") {
		t.Fatalf("engine error not logged:\n%s", h.log.String())
	}
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionPasteErrorShowsFailed(t *testing.T) {
	eng := &asr.FakeEngine{Fn: func([]float32, asr.Options) (asr.Result, error) {
		return asr.Result{Text: "hi"}, nil
	}}
	h := newHarness(t, eng, seconds(1), nil)
	h.inj.err = errors.New("clipboard busy")
	h.start()

	h.press()
	h.release()
	waitFor(t, "failed flash to end", func() bool {
		return h.ovl.lastState() == StateIdle && equalStates(h.ovl.stateSeq(), []State{StateListening, StateWorking, StateFailed, StateIdle})
	})
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionKeyDownWhileWorkingIgnored(t *testing.T) {
	eng := &gateEngine{gate: make(chan struct{}), entered: make(chan struct{}, 1), text: "ok"}
	h := newHarness(t, eng, seconds(1), nil)
	h.start()

	h.press()
	h.release()
	select {
	case <-eng.entered:
	case <-time.After(2 * time.Second):
		t.Fatal("engine never started")
	}

	// The session loop handles these sequentially, so once the sends return the events are processed.
	h.press()
	h.release()
	close(eng.gate)

	waitFor(t, "paste and idle", func() bool {
		return len(h.inj.pasted()) == 1 && h.ovl.lastState() == StateIdle
	})
	if begins, ends := h.rec.counts(); begins != 1 || ends != 1 {
		t.Fatalf("recorder begin=%d end=%d, want 1 and 1 (ignored key events must not record)", begins, ends)
	}
	if got := h.inj.pasted(); len(got) != 1 || got[0] != "ok " {
		t.Fatalf("pasted %q, want [\"ok \"]", got)
	}
	if !strings.Contains(h.log.String(), "ignored key down") {
		t.Fatalf("ignored key down not logged:\n%s", h.log.String())
	}
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
}

// checkLevelsOnlyWhileListening fails if any SetLevel call happens when the last SetState was not Listening.
func checkLevelsOnlyWhileListening(t *testing.T, events []ovlEvent) {
	t.Helper()
	cur := StateIdle
	for _, e := range events {
		if e.isLevel {
			if cur != StateListening {
				t.Errorf("SetLevel(%v) while overlay state is %v", e.level, cur)
			}
			continue
		}
		cur = e.state
	}
}

func TestSessionLevelPumpRunsOnlyWhileListening(t *testing.T) {
	h := newHarness(t, &asr.FakeEngine{}, seconds(1), nil)
	h.start()

	h.press()
	waitFor(t, "level updates while listening", func() bool { return h.ovl.levelCount() >= 3 })
	h.release()
	waitFor(t, "release processed", func() bool {
		_, ends := h.rec.counts()
		return ends == 1 && h.ovl.lastState() == StateIdle
	})

	n := h.ovl.levelCount()
	time.Sleep(30 * time.Millisecond)
	if got := h.ovl.levelCount(); got != n {
		t.Fatalf("level updates continued after release: %d -> %d", n, got)
	}
	checkLevelsOnlyWhileListening(t, h.ovl.snapshot())
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionShutdownWhileListeningStopsLevelPump(t *testing.T) {
	h := newHarness(t, &asr.FakeEngine{}, seconds(1), nil)
	h.start()

	h.press()
	waitFor(t, "level updates", func() bool { return h.ovl.levelCount() >= 1 })
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
	n := h.ovl.levelCount()
	time.Sleep(20 * time.Millisecond)
	if got := h.ovl.levelCount(); got != n {
		t.Fatalf("SetLevel called after run returned: %d -> %d", n, got)
	}
	checkLevelsOnlyWhileListening(t, h.ovl.snapshot())
}

func TestSessionShutdownMidTranscriptionDoesNotPaste(t *testing.T) {
	eng := &gateEngine{gate: make(chan struct{}), entered: make(chan struct{}, 1), text: "late"}
	h := newHarness(t, eng, seconds(1), nil)
	h.start()

	h.press()
	h.release()
	select {
	case <-eng.entered:
	case <-time.After(2 * time.Second):
		t.Fatal("engine never started")
	}
	if err := h.stop(); err != nil {
		t.Fatal(err)
	}
	if got := h.inj.pasted(); len(got) != 0 {
		t.Fatalf("pasted %q after shutdown", got)
	}
}

func TestSessionQuitFromOverlayShutsDown(t *testing.T) {
	quit := make(chan struct{})
	h := newHarness(t, &asr.FakeEngine{}, seconds(1), quit)
	h.start()

	close(quit)
	if err := h.wait(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionClosedKeyChannelShutsDown(t *testing.T) {
	h := newHarness(t, &asr.FakeEngine{}, seconds(1), nil)
	h.start()

	close(h.keys)
	if err := h.wait(); err != nil {
		t.Fatal(err)
	}
}

func TestParseDictateFlags(t *testing.T) {
	o, err := parseDictateFlags(nil, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if o.key != "f9" || !o.swallow || !o.overlay {
		t.Fatalf("defaults = %+v, want key f9, swallow and overlay on", o)
	}

	o, err = parseDictateFlags([]string{"--key", "rctrl", "--no-swallow", "--no-overlay"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if o.key != "rctrl" || o.swallow || o.overlay {
		t.Fatalf("flags = %+v, want key rctrl, no swallow, no overlay", o)
	}

	if _, err := parseDictateFlags([]string{"extra"}, io.Discard); err == nil {
		t.Fatal("positional argument accepted")
	}
}
