// Command rbbench measures WER, RTF and latency.
//
//	rbbench run   -manifest m.jsonl -model DIR -ort onnxruntime.dll -out report.json [-limit N] [-ep cuda]
//	rbbench score -manifest m.jsonl -hyps h.jsonl -engine NAME -out report.json
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/ortengine"
	"github.com/thegr8yg/ramblocal/internal/audio"
	"github.com/thegr8yg/ramblocal/internal/bench"
)

func main() {
	if len(os.Args) < 2 {
		fail(errors.New("usage: rbbench run|score [flags]"))
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = cmdRun(os.Args[2:])
	case "score":
		err = cmdScore(os.Args[2:])
	default:
		err = fmt.Errorf("unknown subcommand %q", os.Args[1])
	}
	if err != nil {
		fail(err)
	}
}

// required returns an error naming the first empty flag value. kv is name, value pairs.
func required(cmd string, kv ...string) error {
	for i := 0; i < len(kv); i += 2 {
		if kv[i+1] == "" {
			return fmt.Errorf("%s: -%s is required", cmd, kv[i])
		}
	}
	return nil
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	manifest := fs.String("manifest", "", "items JSONL")
	model := fs.String("model", os.Getenv("RAMBLOCAL_MODEL_DIR"), "model directory")
	ortLib := fs.String("ort", os.Getenv("RAMBLOCAL_ORT_LIB"), "onnxruntime.dll path")
	ep := fs.String("ep", "", "force one execution provider (cuda, dml, cpu)")
	out := fs.String("out", "", "report JSON path")
	limit := fs.Int("limit", 0, "only the first N items")
	warmup := fs.Int("warmup", 3, "untimed warm-up runs")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := required("run", "manifest", *manifest, "model", *model, "ort", *ortLib, "out", *out); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	release, err := bench.AcquireGPULock(filepath.Join(os.TempDir(), "ramblocal-gpu.lock"), 30*time.Minute)
	if err != nil {
		return err
	}
	defer release()

	items, err := loadItems(*manifest, *limit)
	if err != nil {
		return err
	}
	_, before, _ := bench.QueryGPU()
	cfg := ortengine.Config{ModelDir: *model, ORTLib: *ortLib}
	if *ep != "" {
		cfg.EPs = []string{*ep}
	}
	t0 := time.Now()
	e, err := ortengine.New(cfg)
	if err != nil {
		return err
	}
	defer e.Close()
	cold := time.Since(t0)

	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		return err
	}
	hypsPath := strings.TrimSuffix(*out, ".json") + ".hyps.jsonl"
	hf, err := os.Create(hypsPath)
	if err != nil {
		return err
	}
	defer hf.Close()

	base := filepath.Dir(*manifest)
	load := func(p string) ([]float32, error) { return readAudio(base, p) }
	r, err := bench.Run(ctx, e, items, load, *warmup, hf)
	if err != nil {
		return err
	}
	gpu, after, _ := bench.QueryGPU()
	r.GPU, r.VRAMMB, r.ColdStartMS = gpu, after-before, float64(cold.Microseconds())/1000
	return write(*out, r)
}

func cmdScore(args []string) error {
	fs := flag.NewFlagSet("score", flag.ContinueOnError)
	manifest := fs.String("manifest", "", "items JSONL")
	hyps := fs.String("hyps", "", "hypotheses JSONL")
	engine := fs.String("engine", "", "engine name for the report")
	out := fs.String("out", "", "report JSON path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := required("score", "manifest", *manifest, "hyps", *hyps, "engine", *engine, "out", *out); err != nil {
		return err
	}

	items, err := loadItems(*manifest, 0)
	if err != nil {
		return err
	}
	// Durations come from the WAVs themselves so AudioSec and RTF match what run would report.
	base := filepath.Dir(*manifest)
	durations := make(map[string]float64, len(items))
	for _, it := range items {
		pcm, err := readAudio(base, it.Audio)
		if err != nil {
			return err
		}
		durations[it.Audio] = float64(len(pcm)) / asr.SampleRate
	}

	f, err := os.Open(*hyps)
	if err != nil {
		return err
	}
	defer f.Close()
	h, err := bench.LoadHyps(f)
	if err != nil {
		return err
	}
	r, err := bench.Score(*engine, items, h, durations)
	if err != nil {
		return err
	}
	return write(*out, r)
}

// readAudio loads a 16 kHz mono WAV, resolving relative paths against the manifest directory.
func readAudio(base, p string) ([]float32, error) {
	if !filepath.IsAbs(p) {
		p = filepath.Join(base, p)
	}
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	pcm, rate, err := audio.ReadWAV(f)
	if err == nil && rate != asr.SampleRate {
		err = fmt.Errorf("%s: sample rate %d, want %d (run bench/prep-librispeech.ps1)", p, rate, asr.SampleRate)
	}
	return pcm, err
}

func loadItems(path string, limit int) ([]bench.Item, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	items, err := bench.LoadItems(f)
	if err != nil {
		return nil, err
	}
	if limit > 0 && limit < len(items) {
		items = items[:limit]
	}
	return items, nil
}

func write(path string, r bench.Report) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(b))
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "rbbench:", err)
	os.Exit(1)
}
