// Command rbbench measures WER, RTF and latency.
//
//	rbbench run   -manifest m.jsonl -model DIR -ort onnxruntime.dll -out report.json [-limit N] [-ep cuda]
//	rbbench score -manifest m.jsonl -hyps h.jsonl -engine NAME -out report.json
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr/ortengine"
	"github.com/thegr8yg/ramblocal/internal/audio"
	"github.com/thegr8yg/ramblocal/internal/bench"
)

func main() {
	if len(os.Args) < 2 {
		fail(fmt.Errorf("usage: rbbench run|score [flags]"))
	}
	switch os.Args[1] {
	case "run":
		cmdRun(os.Args[2:])
	case "score":
		cmdScore(os.Args[2:])
	default:
		fail(fmt.Errorf("unknown subcommand %q", os.Args[1]))
	}
}

func cmdRun(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	manifest := fs.String("manifest", "", "items JSONL")
	model := fs.String("model", os.Getenv("RAMBLOCAL_MODEL_DIR"), "model directory")
	ortLib := fs.String("ort", os.Getenv("RAMBLOCAL_ORT_LIB"), "onnxruntime.dll path")
	ep := fs.String("ep", "", "force one execution provider (cuda, dml, cpu)")
	out := fs.String("out", "", "report JSON path")
	limit := fs.Int("limit", 0, "only the first N items")
	warmup := fs.Int("warmup", 3, "untimed warm-up runs")
	fs.Parse(args)
	if *out == "" {
		fail(fmt.Errorf("-out is required"))
	}

	release, err := bench.AcquireGPULock(filepath.Join(os.TempDir(), "ramblocal-gpu.lock"), 30*time.Minute)
	must(err)
	defer release()

	items := loadItems(*manifest, *limit)
	_, before, _ := bench.QueryGPU()
	cfg := ortengine.Config{ModelDir: *model, ORTLib: *ortLib}
	if *ep != "" {
		cfg.EPs = []string{*ep}
	}
	t0 := time.Now()
	e, err := ortengine.New(cfg)
	must(err)
	defer e.Close()
	cold := time.Since(t0)

	hypsPath := strings.TrimSuffix(*out, ".json") + ".hyps.jsonl"
	hf, err := os.Create(hypsPath)
	must(err)
	defer hf.Close()
	base := filepath.Dir(*manifest)
	load := func(p string) ([]float32, error) {
		if !filepath.IsAbs(p) {
			p = filepath.Join(base, p)
		}
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		pcm, rate, err := audio.ReadWAV(f)
		if err == nil && rate != 16000 {
			err = fmt.Errorf("%s: sample rate %d, want 16000 (run bench/prep-librispeech.ps1)", p, rate)
		}
		return pcm, err
	}
	r, err := bench.Run(context.Background(), e, items, load, *warmup, hf)
	must(err)
	gpu, after, _ := bench.QueryGPU()
	r.GPU, r.VRAMMB, r.ColdStartMS = gpu, after-before, float64(cold.Microseconds())/1000
	write(*out, r)
}

func cmdScore(args []string) {
	fs := flag.NewFlagSet("score", flag.ExitOnError)
	manifest := fs.String("manifest", "", "items JSONL")
	hyps := fs.String("hyps", "", "hypotheses JSONL")
	engine := fs.String("engine", "", "engine name for the report")
	out := fs.String("out", "", "report JSON path")
	fs.Parse(args)

	items := loadItems(*manifest, 0)
	f, err := os.Open(*hyps)
	must(err)
	defer f.Close()
	h, err := bench.LoadHyps(f)
	must(err)
	r, err := bench.Score(*engine, items, h)
	must(err)
	write(*out, r)
}

func loadItems(path string, limit int) []bench.Item {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	items, err := bench.LoadItems(f)
	must(err)
	if limit > 0 && limit < len(items) {
		items = items[:limit]
	}
	return items
}

func write(path string, r bench.Report) {
	b, _ := json.MarshalIndent(r, "", "  ")
	fmt.Println(string(b))
	if path != "" {
		must(os.MkdirAll(filepath.Dir(path), 0o755))
		must(os.WriteFile(path, b, 0o644))
	}
}

func must(err error) {
	if err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "rbbench:", err)
	os.Exit(1)
}
