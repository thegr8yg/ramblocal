package bench

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
)

func TestRunWithFakeEngine(t *testing.T) {
	texts := map[int]string{16000: "the cat sat", 32000: "on a mat"}
	e := &asr.FakeEngine{Fn: func(pcm []float32, _ asr.Options) (asr.Result, error) {
		return asr.Result{Text: texts[len(pcm)], AudioSec: float64(len(pcm)) / 16000,
			Timings: asr.StageTimings{Encoder: time.Millisecond}}, nil
	}}
	items := []Item{{Audio: "1s", Text: "The cat sat."}, {Audio: "2s", Text: "on the mat"}}
	load := func(p string) ([]float32, error) {
		if p == "1s" {
			return make([]float32, 16000), nil
		}
		return make([]float32, 32000), nil
	}
	var hyps bytes.Buffer
	r, err := Run(context.Background(), e, items, load, 1, &hyps)
	if err != nil {
		t.Fatal(err)
	}
	if r.Items != 2 || r.AudioSec != 3 || r.WER < 16.66 || r.WER > 16.67 {
		t.Fatalf("report %+v", r)
	}
	if r.Latency["encoder"].P50 != 1 {
		t.Fatalf("encoder p50 = %v ms", r.Latency["encoder"].P50)
	}
	if e.Calls() != 3 { // 1 warm-up + 2 items
		t.Fatalf("calls = %d", e.Calls())
	}
	if strings.Count(hyps.String(), "\n") != 2 {
		t.Fatalf("hyps:\n%s", hyps.String())
	}
}

func TestScore(t *testing.T) {
	items := []Item{{Audio: "a", Text: "one two"}, {Audio: "b", Text: "three"}}
	durs := map[string]float64{"a": 2, "b": 1}
	r, err := Score("whisper", items, []HypLine{{Audio: "a", Hyp: "one two", Seconds: 0.1}, {Audio: "b", Hyp: "tree", Seconds: 0.3}}, durs)
	if err != nil || r.WER < 33.33 || r.WER > 33.34 || r.Latency["total"].Max != 300 {
		t.Fatalf("%+v %v", r, err)
	}
	if r.AudioSec != 3 || r.WallSec < 0.399 || r.WallSec > 0.401 || r.RTF < 0.133 || r.RTF > 0.134 {
		t.Fatalf("parity fields: %+v", r)
	}
	if _, err := Score("x", items, []HypLine{{Audio: "a", Hyp: "one two"}}, durs); err == nil {
		t.Fatal("missing hypothesis: expected error")
	}
}

func TestScorePartialTiming(t *testing.T) {
	items := []Item{{Audio: "a", Text: "one two"}, {Audio: "b", Text: "three"}}
	durs := map[string]float64{"a": 2, "b": 1}
	r, err := Score("whisper", items, []HypLine{{Audio: "a", Hyp: "one two", Seconds: 0.1}, {Audio: "b", Hyp: "tree"}}, durs)
	if err != nil {
		t.Fatal(err)
	}
	if r.WallSec != 0 || r.RTF != 0 || r.AudioSec != 3 {
		t.Fatalf("partial timing should not report wall/rtf: %+v", r)
	}
	if _, ok := r.Latency["total"]; ok {
		t.Fatal("partial timing should not report latency")
	}
}

func TestScoreDuplicateHyp(t *testing.T) {
	items := []Item{{Audio: "a", Text: "one"}}
	durs := map[string]float64{"a": 1}
	hyps := []HypLine{{Audio: "a", Hyp: "one"}, {Audio: "a", Hyp: "two"}}
	if _, err := Score("x", items, hyps, durs); err == nil {
		t.Fatal("duplicate hypothesis: expected error")
	}
}

func TestScoreUnknownHyp(t *testing.T) {
	items := []Item{{Audio: "a", Text: "one"}}
	durs := map[string]float64{"a": 1}
	hyps := []HypLine{{Audio: "a", Hyp: "one"}, {Audio: "zzz", Hyp: "stray"}}
	if _, err := Score("x", items, hyps, durs); err == nil {
		t.Fatal("hypothesis not in manifest: expected error")
	}
}
