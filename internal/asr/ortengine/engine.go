package ortengine

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
)

// MinSamples: clips shorter than 100 ms return an empty Result without running the model.
const MinSamples = asr.SampleRate / 10

const maxSymbolsPerFrame = 10 // must match tools/export greedy config

type Config struct {
	ModelDir string
	ORTLib   string
	EPs      []string // tried in order; default cuda, dml, cpu
}

type Engine struct {
	mu  sync.Mutex // one inference at a time; the scheduler (Phase 2) orders callers
	m   *asr.Manifest
	tok *asr.Tokenizer
	g   *graphs
	ep  string
}

// New opens the model on the first execution provider that loads and survives a warm-up run.
func New(cfg Config) (*Engine, error) {
	if err := InitRuntime(cfg.ORTLib); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}
	m, err := asr.LoadManifest(filepath.Join(cfg.ModelDir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	f, err := os.Open(filepath.Join(cfg.ModelDir, "vocab.txt"))
	if err != nil {
		return nil, err
	}
	tok, err := asr.LoadVocab(f)
	f.Close()
	if err != nil {
		return nil, err
	}
	if tok.Size() != m.VocabSize {
		return nil, fmt.Errorf("vocab.txt has %d pieces, manifest vocab_size %d", tok.Size(), m.VocabSize)
	}
	eps := cfg.EPs
	if len(eps) == 0 {
		eps = []string{"cuda", "dml", "cpu"}
	}
	var errs []error
	for _, ep := range eps {
		g, err := openGraphs(cfg.ModelDir, m, ep)
		if err == nil {
			e := &Engine{m: m, tok: tok, g: g, ep: ep}
			if _, err = e.Transcribe(context.Background(), make([]float32, asr.SampleRate), asr.Options{}); err == nil {
				return e, nil
			}
			g.close()
		}
		errs = append(errs, fmt.Errorf("%s: %w", ep, err))
	}
	return nil, fmt.Errorf("no execution provider worked: %w", errors.Join(errs...))
}

func (e *Engine) Name() string { return "ort-" + e.ep }
func (e *Engine) EP() string   { return e.ep }

func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.g != nil {
		e.g.close()
		e.g = nil
	}
	return nil
}

func (e *Engine) Transcribe(ctx context.Context, pcm []float32, _ asr.Options) (asr.Result, error) {
	res := asr.Result{AudioSec: float64(len(pcm)) / asr.SampleRate}
	if err := ctx.Err(); err != nil {
		return res, err
	}
	if len(pcm) < MinSamples {
		return res, nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return res, err
	}
	if e.g == nil {
		return res, errors.New("engine closed")
	}

	t0 := time.Now()
	feats, frames, err := e.g.preprocess(pcm)
	if err != nil {
		return res, fmt.Errorf("preprocess: %w", err)
	}
	t1 := time.Now()
	enc, outFrames, err := e.g.encode(feats, frames)
	if err != nil {
		return res, fmt.Errorf("encode: %w", err)
	}
	t2 := time.Now()
	hyp, err := asr.GreedyTDT(ctx, e.g.joint(enc, outFrames), outFrames, e.m.BlankID, e.m.Durations, maxSymbolsPerFrame)
	if err != nil {
		return res, fmt.Errorf("decode: %w", err)
	}
	t3 := time.Now()
	text, err := e.tok.Decode(hyp.Tokens)
	if err != nil {
		return res, fmt.Errorf("detokenize: %w", err)
	}
	words, err := e.tok.Words(hyp.Tokens, hyp.Frames, e.m.FrameSeconds())
	if err != nil {
		return res, fmt.Errorf("detokenize: %w", err)
	}
	t4 := time.Now()

	res.Text, res.Tokens, res.Words = text, hyp.Tokens, words
	res.Timings = asr.StageTimings{Mel: t1.Sub(t0), Encoder: t2.Sub(t1), Decoder: t3.Sub(t2), Detok: t4.Sub(t3)}
	return res, nil
}
