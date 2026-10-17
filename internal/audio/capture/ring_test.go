package capture

import (
	"math/rand"
	"reflect"
	"sync"
	"testing"
)

// seq returns n samples counting up from start.
func seq(start, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(start + i)
	}
	return s
}

func TestRingKeepsLastN(t *testing.T) {
	r := newRing(4)
	r.push(seq(0, 3))
	if got := r.appendTo(nil); !reflect.DeepEqual(got, []float32{0, 1, 2}) {
		t.Fatalf("partial fill: got %v", got)
	}
	r.push(seq(3, 3)) // wraps
	if got := r.appendTo(nil); !reflect.DeepEqual(got, []float32{2, 3, 4, 5}) {
		t.Fatalf("after wrap: got %v", got)
	}
}

func TestRingChunkLargerThanBuffer(t *testing.T) {
	r := newRing(4)
	r.push(seq(0, 10))
	if got := r.appendTo(nil); !reflect.DeepEqual(got, []float32{6, 7, 8, 9}) {
		t.Fatalf("got %v", got)
	}
	r.push(seq(20, 1))
	if got := r.appendTo(nil); !reflect.DeepEqual(got, []float32{7, 8, 9, 20}) {
		t.Fatalf("after small push: got %v", got)
	}
}

func TestRingMatchesReference(t *testing.T) {
	const n = 37
	rng := rand.New(rand.NewSource(1))
	r := newRing(n)
	var all []float32
	for i := 0; i < 500; i++ {
		chunk := seq(len(all), 1+rng.Intn(60))
		all = append(all, chunk...)
		r.push(chunk)
		want := all
		if len(want) > n {
			want = want[len(want)-n:]
		}
		if got := r.appendTo(nil); !reflect.DeepEqual(got, append([]float32(nil), want...)) {
			t.Fatalf("step %d: got %v want %v", i, got, want)
		}
	}
}

func TestPrerollIs300ms(t *testing.T) {
	if prerollSamples != 4800 {
		t.Fatalf("prerollSamples = %d, want 4800 (300 ms at 16 kHz)", prerollSamples)
	}
	a := newAccumulator()
	for i := 0; i < 50; i++ { // 1 s of audio in 320-sample blocks
		a.feed(seq(i*320, 320))
	}
	a.begin()
	got := a.end()
	if len(got) != prerollSamples {
		t.Fatalf("pre-roll length = %d, want %d", len(got), prerollSamples)
	}
	if got[0] != 16000-4800 || got[len(got)-1] != 15999 {
		t.Fatalf("pre-roll spans [%v, %v], want the last 300 ms", got[0], got[len(got)-1])
	}
}

func TestBeginEndCollectsPrerollAndAudio(t *testing.T) {
	a := newAccumulator()
	a.feed(seq(0, 8000))
	a.begin() // pre-roll is the last 4800 samples: 3200..7999
	a.feed(seq(8000, 500))
	a.feed(seq(8500, 300))
	got := a.end()
	want := append(seq(3200, prerollSamples), seq(8000, 800)...)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %d samples, want %d (pre-roll + utterance)", len(got), len(want))
	}
}

func TestPrerollShorterThanWindowAtStart(t *testing.T) {
	a := newAccumulator()
	a.feed(seq(0, 100))
	a.begin()
	if got := a.end(); !reflect.DeepEqual(got, seq(0, 100)) {
		t.Fatalf("got %d samples, want the 100 available", len(got))
	}
}

func TestSamplesOutsideUtteranceAreNotCollected(t *testing.T) {
	a := newAccumulator()
	a.feed(seq(0, 100)) // before Begin: only pre-roll, not utterance
	a.begin()
	a.feed(seq(100, 10))
	if got := a.end(); len(got) != 110 {
		t.Fatalf("got %d samples, want 110", len(got))
	}
	a.feed(seq(200, 1000)) // after End: must not be collected
	if got := a.end(); got != nil {
		t.Fatalf("end without begin returned %d samples", len(got))
	}
}

func TestEndWithoutBeginReturnsNil(t *testing.T) {
	a := newAccumulator()
	a.feed(seq(0, 10))
	if got := a.end(); got != nil {
		t.Fatalf("got %v, want nil", got)
	}
}

func TestBeginAgainRestartsUtterance(t *testing.T) {
	a := newAccumulator()
	a.feed(seq(0, 5000))
	a.begin()
	a.feed(seq(5000, 1000))
	a.begin()
	a.feed(seq(6000, 10))
	got := a.end()
	// The new pre-roll is the last 4800 samples before the second Begin.
	want := append(seq(1200, 4800), seq(6000, 10)...)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("restarted utterance: got %d samples, want %d", len(got), len(want))
	}
}

func TestUtteranceCappedAtFiveMinutes(t *testing.T) {
	if maxUtteranceSamples != 4800000 {
		t.Fatalf("maxUtteranceSamples = %d", maxUtteranceSamples)
	}
	a := newAccumulator()
	a.begin()
	block := seq(0, 320)
	for i := 0; i < (maxUtteranceSamples+10000)/320+1; i++ {
		a.feed(block)
	}
	if got := a.end(); len(got) != maxUtteranceSamples {
		t.Fatalf("length = %d, want cap %d", len(got), maxUtteranceSamples)
	}
}

func TestEndResultIsNotAliased(t *testing.T) {
	a := newAccumulator()
	a.begin()
	a.feed(seq(0, 100))
	out := a.end()
	snapshot := append([]float32(nil), out...)

	a.begin()
	a.feed(seq(900, 100))
	a.feed(seq(1000, 100))
	if !reflect.DeepEqual(out, snapshot) {
		t.Fatal("returned utterance changed after a later utterance was recorded")
	}
}

func TestConcurrentFeedBeginEnd(t *testing.T) {
	a := newAccumulator()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 2000; i++ {
			a.feed(seq(i*320, 320))
		}
	}()
	for i := 0; i < 300; i++ {
		a.begin()
		for j := 0; j < 3; j++ {
			_ = a.end()
			a.begin()
		}
		if out := a.end(); len(out) > maxUtteranceSamples {
			t.Fatalf("utterance over cap: %d", len(out))
		}
	}
	wg.Wait()
}
