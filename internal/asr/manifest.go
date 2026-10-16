package asr

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

// Graph describes one exported ONNX graph. Input/output order is fixed by tools/export.
type Graph struct {
	File    string            `json:"file"`
	Inputs  []string          `json:"inputs"`
	Outputs []string          `json:"outputs"`
	DTypes  map[string]string `json:"dtypes"`
}

// Frontend describes the NeMo log-mel frontend computed in Go. Filterbank and Window name
// little-endian float32 files in the model directory. Optional: old manifests omit it and
// use the preprocessor graph instead.
type Frontend struct {
	NFFT       int     `json:"n_fft"`
	WinLength  int     `json:"win_length"`
	HopLength  int     `json:"hop_length"`
	Preemph    float64 `json:"preemph"`
	LogGuard   float64 `json:"log_guard"`
	NormEps    float64 `json:"norm_eps"`
	Filterbank string  `json:"filterbank"`
	Window     string  `json:"window"`
}

// Manifest is written by tools/export/export_parakeet.py; it is the only source of model facts.
type Manifest struct {
	Name         string    `json:"name"`
	SampleRate   int       `json:"sample_rate"`
	NumFeatures  int       `json:"num_features"`
	HopSeconds   float64   `json:"hop_seconds"`
	Subsampling  int       `json:"subsampling"`
	EncoderDim   int       `json:"encoder_dim"`
	VocabSize    int       `json:"vocab_size"`
	BlankID      int       `json:"blank_id"`
	Durations    []int     `json:"tdt_durations"`
	PredLayers   int       `json:"pred_layers"`
	PredHidden   int       `json:"pred_hidden"`
	Preprocessor Graph     `json:"preprocessor"`
	Encoder      Graph     `json:"encoder"`
	DecoderJoint Graph     `json:"decoder_joint"`
	Frontend     *Frontend `json:"frontend"`
}

func LoadManifest(path string) (*Manifest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("manifest %s: %w", path, err)
	}
	if err := m.validate(); err != nil {
		return nil, fmt.Errorf("manifest %s: %w", path, err)
	}
	return &m, nil
}

// FrameSeconds is the duration of one encoder output frame.
func (m *Manifest) FrameSeconds() float64 {
	return float64(m.Subsampling) * m.HopSeconds
}

func (m *Manifest) validate() error {
	var errs []error
	check := func(ok bool, msg string) {
		if !ok {
			errs = append(errs, errors.New(msg))
		}
	}
	check(m.SampleRate == SampleRate, "sample_rate must be 16000")
	check(m.NumFeatures > 0, "num_features must be > 0")
	check(m.HopSeconds > 0, "hop_seconds must be > 0")
	check(m.Subsampling > 0, "subsampling must be > 0")
	check(m.EncoderDim > 0, "encoder_dim must be > 0")
	check(m.VocabSize > 0, "vocab_size must be > 0")
	check(m.BlankID == m.VocabSize, "blank_id must equal vocab_size (blank is the last logit)")
	check(len(m.Durations) > 0, "tdt_durations must not be empty")
	check(m.PredLayers > 0 && m.PredHidden > 0, "pred_layers and pred_hidden must be > 0")
	check(m.Preprocessor.File != "" && len(m.Preprocessor.Inputs) == 2 && len(m.Preprocessor.Outputs) == 2, "preprocessor needs file, 2 inputs, 2 outputs")
	check(m.Encoder.File != "" && len(m.Encoder.Inputs) == 2 && len(m.Encoder.Outputs) == 2, "encoder needs file, 2 inputs, 2 outputs")
	check(m.DecoderJoint.File != "" && len(m.DecoderJoint.Inputs) == 5 && len(m.DecoderJoint.Outputs) == 4, "decoder_joint needs file, 5 inputs, 4 outputs")
	if f := m.Frontend; f != nil {
		check(f.NFFT >= 2 && f.NFFT&(f.NFFT-1) == 0, "frontend.n_fft must be a power of two >= 2")
		check(f.WinLength > 0 && f.WinLength <= f.NFFT, "frontend.win_length must be in (0, n_fft]")
		check(f.HopLength > 0, "frontend.hop_length must be > 0")
		check(f.LogGuard > 0, "frontend.log_guard must be > 0")
		check(f.NormEps >= 0, "frontend.norm_eps must be >= 0")
		check(f.Filterbank != "", "frontend.filterbank must not be empty")
		check(f.Window != "", "frontend.window must not be empty")
	}
	return errors.Join(errs...)
}
