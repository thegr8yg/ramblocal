package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/eval"
)

type Report struct {
	Engine      string         `json:"engine"`
	GPU         string         `json:"gpu"`
	Date        string         `json:"date"`
	Items       int            `json:"items"`
	AudioSec    float64        `json:"audio_sec"`
	WallSec     float64        `json:"wall_sec"`
	RTF         float64        `json:"rtf"`
	WER         float64        `json:"wer"`
	Latency     map[string]Pct `json:"latency_ms"`
	VRAMMB      int            `json:"vram_mb"`
	ColdStartMS float64        `json:"cold_start_ms"`
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// Run transcribes every item (after `warmup` untimed runs of the first item) and writes one HypLine per item to hyps.
func Run(ctx context.Context, e asr.Engine, items []Item, load func(string) ([]float32, error), warmup int, hyps io.Writer) (Report, error) {
	r := Report{Engine: e.Name(), Date: time.Now().Format("2006-01-02"), Latency: map[string]Pct{}}
	if len(items) == 0 {
		return r, fmt.Errorf("no items")
	}
	if warmup > 0 {
		pcm, err := load(items[0].Audio)
		if err != nil {
			return r, err
		}
		for i := 0; i < warmup; i++ {
			if _, err := e.Transcribe(ctx, pcm, asr.Options{}); err != nil {
				return r, fmt.Errorf("warm-up: %w", err)
			}
		}
	}
	stages := map[string][]float64{}
	var corpus eval.Corpus
	enc := json.NewEncoder(hyps)
	var wall time.Duration
	for _, it := range items {
		pcm, err := load(it.Audio)
		if err != nil {
			return r, fmt.Errorf("%s: %w", it.Audio, err)
		}
		start := time.Now()
		res, err := e.Transcribe(ctx, pcm, asr.Options{})
		total := time.Since(start)
		if err != nil {
			return r, fmt.Errorf("%s: %w", it.Audio, err)
		}
		wall += total
		r.AudioSec += float64(len(pcm)) / asr.SampleRate
		corpus.Add(it.Text, res.Text)
		stages["total"] = append(stages["total"], ms(total))
		stages["mel"] = append(stages["mel"], ms(res.Timings.Mel))
		stages["encoder"] = append(stages["encoder"], ms(res.Timings.Encoder))
		stages["decoder"] = append(stages["decoder"], ms(res.Timings.Decoder))
		stages["detok"] = append(stages["detok"], ms(res.Timings.Detok))
		if err := enc.Encode(HypLine{Audio: it.Audio, Hyp: res.Text, Seconds: total.Seconds()}); err != nil {
			return r, err
		}
	}
	for k, v := range stages {
		r.Latency[k] = Percentiles(v)
	}
	r.Items = len(items)
	r.WallSec = wall.Seconds()
	r.WER = corpus.WER()
	if r.AudioSec > 0 {
		r.RTF = r.WallSec / r.AudioSec
	}
	return r, nil
}

// Score builds a report from hypotheses produced elsewhere (NeMo reference, Whisper baseline).
func Score(engine string, items []Item, hyps []HypLine) (Report, error) {
	r := Report{Engine: engine, Date: time.Now().Format("2006-01-02"), Latency: map[string]Pct{}}
	byAudio := make(map[string]HypLine, len(hyps))
	for _, h := range hyps {
		byAudio[h.Audio] = h
	}
	var corpus eval.Corpus
	var totals []float64
	for _, it := range items {
		h, ok := byAudio[it.Audio]
		if !ok {
			return r, fmt.Errorf("no hypothesis for %s", it.Audio)
		}
		corpus.Add(it.Text, h.Hyp)
		if h.Seconds > 0 {
			totals = append(totals, h.Seconds*1000)
			r.WallSec += h.Seconds
		}
	}
	r.Items = len(items)
	r.WER = corpus.WER()
	if len(totals) > 0 {
		r.Latency["total"] = Percentiles(totals)
	}
	return r, nil
}
