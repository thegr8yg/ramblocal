package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/ortengine"
)

const (
	defaultDictateKey = "f9"
	minUtterance      = 250 * time.Millisecond // shorter holds are accidental taps
	levelInterval     = 33 * time.Millisecond  // about 30 Hz meter updates while listening
	failFlash         = 1500 * time.Millisecond
)

const dictateUsage = `usage: ramblocal dictate [flags]

Hold the push-to-talk key, speak, release. The text is pasted into the focused window.

The key must be one that Windows can see. Fn is usually invisible to Windows (the keyboard
handles it in firmware), so it often does not work as a key name. Run "ramblocal keys" to see
which name your keyboard reports for each key. Key names: f1..f24, rctrl, capslock, fn, vk:0xNN.

flags:
`

// Recorder is the microphone side of a session. *capture.Recorder satisfies it.
type Recorder interface {
	Begin()
	End() []float32
	Level() float32
}

// State is what the overlay shows.
type State int

const (
	StateIdle State = iota
	StateListening
	StateWorking
	StateFailed
)

// Overlay is the status widget. SetState and SetLevel are only called from the session loop.
type Overlay interface {
	SetState(State)
	SetLevel(float32) // 0..1, shown only while listening
	Quit() <-chan struct{}
}

// Injector types text into the focused window.
type Injector interface {
	Paste(text string) error
}

// Engine transcribes a complete clip. asr.Engine satisfies it.
type Engine interface {
	Transcribe(ctx context.Context, pcm []float32, opts asr.Options) (asr.Result, error)
}

// KeyEvent is a push-to-talk key transition, independent of the hotkey package.
type KeyEvent int

const (
	KeyDown KeyEvent = iota + 1
	KeyUp
)

// devices is what dictate holds open while it runs. close releases them in reverse order of opening.
type devices struct {
	rec     Recorder
	ovl     Overlay
	inj     Injector
	keys    <-chan KeyEvent
	closers []func()
}

func (d *devices) close() {
	for i := len(d.closers) - 1; i >= 0; i-- {
		d.closers[i]()
	}
	d.closers = nil
}

// nopOverlay is used with --no-overlay.
type nopOverlay struct{}

func (nopOverlay) SetState(State)        {}
func (nopOverlay) SetLevel(float32)      {}
func (nopOverlay) Quit() <-chan struct{} { return nil }

type dictateOptions struct {
	key, model, ort, ep string
	swallow, overlay    bool
}

func parseDictateFlags(args []string, stderr io.Writer) (dictateOptions, error) {
	var o dictateOptions
	var noSwallow, noOverlay bool
	fs := flag.NewFlagSet("dictate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&o.key, "key", defaultDictateKey, "push-to-talk key name (f9, rctrl, capslock, vk:0xNN; run the keys command to see names)")
	fs.StringVar(&o.model, "model", "", "model directory (default: RAMBLOCAL_MODEL_DIR, then models/parakeet-tdt-0.6b-v2)")
	fs.StringVar(&o.ort, "ort", "", "ONNX Runtime library path (default: RAMBLOCAL_ORT_LIB, then next to the executable)")
	fs.StringVar(&o.ep, "ep", "", "execution provider: cuda, dml or cpu (default: try cuda, dml, cpu in order)")
	fs.BoolVar(&noSwallow, "no-swallow", false, "also pass the push-to-talk key to the focused application")
	fs.BoolVar(&noOverlay, "no-overlay", false, "do not show the status widget")
	fs.Usage = func() {
		fmt.Fprint(stderr, dictateUsage)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return o, err
	}
	if fs.NArg() != 0 {
		return o, fmt.Errorf("dictate takes no arguments (got %q)", fs.Args())
	}
	o.swallow = !noSwallow
	o.overlay = !noOverlay
	return o, nil
}

func runDictate(args []string, stderr io.Writer) error {
	opts, err := parseDictateFlags(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	if err != nil {
		return err
	}

	modelDir, err := findModelDir(opts.model)
	if err != nil {
		return err
	}
	lib, err := findORTLib(opts.ort)
	if err != nil {
		return err
	}
	cfg := ortengine.Config{ModelDir: modelDir, ORTLib: lib}
	if opts.ep != "" {
		cfg.EPs = []string{opts.ep}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	t0 := time.Now()
	eng, err := newEngine(cfg)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer eng.Close()
	fmt.Fprintf(stderr, "ramblocal: engine %s, execution provider %q, cold start %s\n",
		eng.Name(), activeEP(eng), time.Since(t0).Round(time.Millisecond))

	dev, err := openDevices(opts, stderr)
	if err != nil {
		return err
	}
	defer dev.close()

	fmt.Fprintf(stderr, "ramblocal: ready. Hold %s to dictate, release to paste. Ctrl+C quits.\n", opts.key)
	s := &session{
		recorder:      dev.rec,
		overlay:       dev.ovl,
		injector:      dev.inj,
		engine:        eng,
		keys:          dev.keys,
		log:           stderr,
		minUtterance:  minUtterance,
		levelInterval: levelInterval,
		failFlash:     failFlash,
	}
	return s.run(ctx)
}

// session runs the push-to-talk state machine. Only run's goroutine touches the recorder and overlay;
// transcription and paste run on one worker goroutine with at most one job in flight.
type session struct {
	recorder      Recorder
	overlay       Overlay
	injector      Injector
	engine        Engine
	keys          <-chan KeyEvent
	log           io.Writer
	minUtterance  time.Duration
	levelInterval time.Duration
	failFlash     time.Duration
}

type job struct {
	pcm        []float32
	releasedAt time.Time
	audioSec   float64
}

type jobResult struct {
	audioSec   float64
	transcribe time.Duration
	paste      time.Duration
	toPaste    time.Duration // release to paste finished
	pasted     bool
	err        error
}

func (s *session) run(ctx context.Context) error {
	wctx, cancel := context.WithCancel(ctx)
	jobs := make(chan job, 1)
	done := make(chan jobResult, 1)
	go s.worker(wctx, jobs, done)
	defer func() {
		// Cancel any transcription in flight, stop the worker, and wait for it so nothing runs after return.
		cancel()
		close(jobs)
		for range done {
		}
	}()

	quit := s.overlay.Quit()
	var (
		listening, busy, failed bool
		shown                   = StateIdle
		ticker                  *time.Ticker
		levelC                  <-chan time.Time
		flashC                  <-chan time.Time
	)
	refresh := func() {
		want := StateIdle
		switch {
		case listening:
			want = StateListening
		case busy:
			want = StateWorking
		case failed:
			want = StateFailed
		}
		if want != shown {
			shown = want
			s.overlay.SetState(want)
		}
	}
	stopLevel := func() {
		if ticker != nil {
			ticker.Stop()
			ticker = nil
		}
		levelC = nil
	}
	defer stopLevel()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-quit:
			return nil
		case ev, ok := <-s.keys:
			if !ok {
				return nil
			}
			switch ev {
			case KeyDown:
				switch {
				case listening:
					s.logf("ignored key down: already listening")
				case busy:
					s.logf("ignored key down: still transcribing the previous utterance")
				default:
					failed, flashC = false, nil
					s.recorder.Begin()
					listening = true
					ticker = time.NewTicker(s.levelInterval)
					levelC = ticker.C
				}
			case KeyUp:
				if !listening {
					break
				}
				stopLevel()
				listening = false
				releasedAt := time.Now()
				pcm := s.recorder.End()
				sec := float64(len(pcm)) / asr.SampleRate
				if sec < s.minUtterance.Seconds() {
					s.logf("ignored accidental tap: %.2fs of audio, need %.2fs", sec, s.minUtterance.Seconds())
					break
				}
				busy = true
				jobs <- job{pcm: pcm, releasedAt: releasedAt, audioSec: sec}
			}
			refresh()
		case <-levelC:
			s.overlay.SetLevel(s.recorder.Level())
		case r := <-done:
			busy = false
			s.report(r)
			if r.err != nil {
				failed = true
				flashC = time.After(s.failFlash)
			}
			refresh()
		case <-flashC:
			flashC = nil
			failed = false
			refresh()
		}
	}
}

func (s *session) worker(ctx context.Context, jobs <-chan job, done chan<- jobResult) {
	defer close(done)
	for j := range jobs {
		done <- s.process(ctx, j)
	}
}

// process transcribes one utterance and pastes the text followed by a space, so the next dictation does not glue on.
func (s *session) process(ctx context.Context, j job) jobResult {
	r := jobResult{audioSec: j.audioSec}
	t0 := time.Now()
	res, err := s.engine.Transcribe(ctx, j.pcm, asr.Options{})
	r.transcribe = time.Since(t0)
	if err != nil {
		r.err = fmt.Errorf("transcribe: %w", err)
		return r
	}
	text := strings.TrimSpace(res.Text)
	if text == "" {
		return r
	}
	if err := ctx.Err(); err != nil {
		r.err = err
		return r
	}
	p0 := time.Now()
	if err := s.injector.Paste(text + " "); err != nil {
		r.err = fmt.Errorf("paste: %w", err)
		return r
	}
	r.paste = time.Since(p0)
	r.toPaste = time.Since(j.releasedAt)
	r.pasted = true
	return r
}

func (s *session) report(r jobResult) {
	switch {
	case r.err != nil:
		s.logf("utterance failed: audio=%.2fs transcribe=%dms: %v", r.audioSec, r.transcribe.Milliseconds(), r.err)
	case !r.pasted:
		s.logf("utterance audio=%.2fs transcribe=%dms: no speech recognised, nothing pasted",
			r.audioSec, r.transcribe.Milliseconds())
	default:
		s.logf("utterance audio=%.2fs transcribe=%dms paste=%dms release-to-paste=%dms",
			r.audioSec, r.transcribe.Milliseconds(), r.paste.Milliseconds(), r.toPaste.Milliseconds())
	}
}

func (s *session) logf(format string, args ...any) {
	fmt.Fprintf(s.log, "ramblocal: "+format+"\n", args...)
}
