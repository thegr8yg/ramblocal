package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/ortengine"
	"github.com/thegr8yg/ramblocal/internal/audio"
)

// newEngine opens the ASR backend. Tests replace it with an asr.FakeEngine so they need no model.
var newEngine = func(cfg ortengine.Config) (asr.Engine, error) {
	e, err := ortengine.New(cfg)
	if err != nil {
		return nil, err
	}
	return e, nil
}

type transcriptJSON struct {
	Text     string     `json:"text"`
	AudioSec float64    `json:"audio_sec"`
	EP       string     `json:"ep"`
	Words    []wordJSON `json:"words"`
}

type wordJSON struct {
	Text  string  `json:"text"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

func runTranscribe(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("transcribe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "model directory (default: RAMBLOCAL_MODEL_DIR, then models/parakeet-tdt-0.6b-v2)")
	ortLib := fs.String("ort", "", "ONNX Runtime library path (default: RAMBLOCAL_ORT_LIB, then next to the executable)")
	ep := fs.String("ep", "", "execution provider: cuda, dml or cpu (default: try cuda, dml, cpu in order)")
	asJSON := fs.Bool("json", false, "print JSON with text, audio length, execution provider and word timestamps")
	timings := fs.Bool("timings", false, "print per-stage timings to stderr")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: ramblocal transcribe [flags] file.wav (got %d arguments)", fs.NArg())
	}

	path := fs.Arg(0)
	pcm, rate, err := readWAVFile(path)
	if err != nil {
		return err
	}
	if rate != asr.SampleRate {
		return fmt.Errorf("%s: sample rate %d Hz, want %d Hz (convert to 16 kHz mono first)", path, rate, asr.SampleRate)
	}

	modelDir, err := findModelDir(*model)
	if err != nil {
		return err
	}
	lib, err := findORTLib(*ortLib)
	if err != nil {
		return err
	}
	cfg := ortengine.Config{ModelDir: modelDir, ORTLib: lib}
	if *ep != "" {
		cfg.EPs = []string{*ep}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	t0 := time.Now()
	e, err := newEngine(cfg)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer e.Close()
	loadTime := time.Since(t0)

	epName := activeEP(e)
	fmt.Fprintf(stderr, "ramblocal: engine %s, execution provider %q\n", e.Name(), epName)

	res, err := e.Transcribe(ctx, pcm, asr.Options{})
	if err != nil {
		return fmt.Errorf("transcribe %s: %w", path, err)
	}

	if *asJSON {
		if err := writeTranscriptJSON(stdout, epName, res); err != nil {
			return err
		}
	} else {
		fmt.Fprintln(stdout, res.Text)
	}
	if *timings {
		t := res.Timings
		fmt.Fprintf(stderr, "timings: load %s, mel %s, encoder %s, decoder %s, detok %s (audio %.2fs)\n",
			loadTime.Round(time.Millisecond), t.Mel.Round(time.Microsecond), t.Encoder.Round(time.Microsecond),
			t.Decoder.Round(time.Microsecond), t.Detok.Round(time.Microsecond), res.AudioSec)
	}
	return nil
}

// activeEP returns the execution provider an engine reports, or "" if it does not report one.
func activeEP(e asr.Engine) string {
	if p, ok := e.(interface{ EP() string }); ok {
		return p.EP()
	}
	return ""
}

func readWAVFile(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	pcm, rate, err := audio.ReadWAV(f)
	if err != nil {
		return nil, 0, fmt.Errorf("%s: %w", path, err)
	}
	return pcm, rate, nil
}

func writeTranscriptJSON(w io.Writer, ep string, res asr.Result) error {
	out := transcriptJSON{
		Text:     res.Text,
		AudioSec: res.AudioSec,
		EP:       ep,
		Words:    make([]wordJSON, 0, len(res.Words)),
	}
	for _, wd := range res.Words {
		out.Words = append(out.Words, wordJSON{Text: wd.Text, Start: wd.Start, End: wd.End})
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}
