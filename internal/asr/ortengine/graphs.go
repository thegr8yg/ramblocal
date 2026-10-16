package ortengine

import (
	"fmt"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/thegr8yg/ramblocal/internal/asr"
)

type graph struct {
	s    *ort.DynamicAdvancedSession
	spec asr.Graph
}

func openGraph(dir string, spec asr.Graph, ep string) (*graph, error) {
	opts, err := sessionOptions(ep)
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	s, err := ort.NewDynamicAdvancedSession(filepath.Join(dir, spec.File), spec.Inputs, spec.Outputs, opts)
	if err != nil {
		return nil, fmt.Errorf("open %s on %s: %w", spec.File, ep, err)
	}
	return &graph{s: s, spec: spec}, nil
}

// run executes the graph; the caller must destroyAll the returned outputs.
func (g *graph) run(inputs []ort.Value) ([]ort.Value, error) {
	outs := make([]ort.Value, len(g.spec.Outputs))
	if err := g.s.Run(inputs, outs); err != nil {
		destroyAll(outs)
		return nil, fmt.Errorf("run %s: %w", g.spec.File, err)
	}
	return outs, nil
}

func (g *graph) inDType(i int) string { return g.spec.DTypes[g.spec.Inputs[i]] }

func destroyAll(vs []ort.Value) {
	for _, v := range vs {
		if v != nil {
			v.Destroy()
		}
	}
}

func intTensor(dtype string, shape ort.Shape, vals ...int64) (ort.Value, error) {
	switch dtype {
	case "int64":
		t, err := ort.NewTensor(shape, vals)
		if err != nil {
			return nil, err
		}
		return t, nil
	case "int32":
		v := make([]int32, len(vals))
		for i, x := range vals {
			v[i] = int32(x)
		}
		t, err := ort.NewTensor(shape, v)
		if err != nil {
			return nil, err
		}
		return t, nil
	}
	return nil, fmt.Errorf("unsupported integer dtype %q", dtype)
}

func floatTensor(shape ort.Shape, data []float32) (ort.Value, error) {
	t, err := ort.NewTensor(shape, data)
	if err != nil {
		return nil, err
	}
	return t, nil
}

// floats copies a float32 output so it survives Destroy.
func floats(v ort.Value) ([]float32, ort.Shape, error) {
	t, ok := v.(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("expected float32 tensor, got %T", v)
	}
	return append([]float32(nil), t.GetData()...), t.GetShape(), nil
}

func firstInt(v ort.Value) (int, error) {
	switch t := v.(type) {
	case *ort.Tensor[int64]:
		return int(t.GetData()[0]), nil
	case *ort.Tensor[int32]:
		return int(t.GetData()[0]), nil
	}
	return 0, fmt.Errorf("expected int tensor, got %T", v)
}

type graphs struct {
	m            *asr.Manifest
	pre, enc, dj *graph
}

func openGraphs(dir string, m *asr.Manifest, ep string) (*graphs, error) {
	g := &graphs{m: m}
	var err error
	if g.pre, err = openGraph(dir, m.Preprocessor, ep); err != nil {
		return nil, err
	}
	if g.enc, err = openGraph(dir, m.Encoder, ep); err != nil {
		g.close()
		return nil, err
	}
	if g.dj, err = openGraph(dir, m.DecoderJoint, ep); err != nil {
		g.close()
		return nil, err
	}
	return g, nil
}

func (g *graphs) close() {
	for _, x := range []*graph{g.pre, g.enc, g.dj} {
		if x != nil {
			x.s.Destroy()
		}
	}
}

// preprocess returns log-mel features laid out [F, frames] and the valid frame count.
func (g *graphs) preprocess(pcm []float32) ([]float32, int, error) {
	w, err := floatTensor(ort.NewShape(1, int64(len(pcm))), pcm)
	if err != nil {
		return nil, 0, err
	}
	l, err := intTensor(g.pre.inDType(1), ort.NewShape(1), int64(len(pcm)))
	if err != nil {
		w.Destroy()
		return nil, 0, err
	}
	ins := []ort.Value{w, l}
	defer destroyAll(ins)
	outs, err := g.pre.run(ins)
	if err != nil {
		return nil, 0, err
	}
	defer destroyAll(outs)
	data, shape, err := floats(outs[0])
	if err != nil {
		return nil, 0, err
	}
	valid, err := firstInt(outs[1])
	if err != nil {
		return nil, 0, err
	}
	trimmed, err := trimFrames(data, int(shape[1]), int(shape[2]), valid)
	if err != nil {
		return nil, 0, err
	}
	return trimmed, valid, nil
}

// encode runs the encoder on [F, frames] features; returns [D, outFrames].
func (g *graphs) encode(feats []float32, frames int) ([]float32, int, error) {
	f := g.m.NumFeatures
	x, err := floatTensor(ort.NewShape(1, int64(f), int64(frames)), feats)
	if err != nil {
		return nil, 0, err
	}
	l, err := intTensor(g.enc.inDType(1), ort.NewShape(1), int64(frames))
	if err != nil {
		x.Destroy()
		return nil, 0, err
	}
	ins := []ort.Value{x, l}
	defer destroyAll(ins)
	outs, err := g.enc.run(ins)
	if err != nil {
		return nil, 0, err
	}
	defer destroyAll(outs)
	data, shape, err := floats(outs[0])
	if err != nil {
		return nil, 0, err
	}
	if int(shape[1]) != g.m.EncoderDim {
		return nil, 0, fmt.Errorf("encoder output dim %d, manifest says %d", shape[1], g.m.EncoderDim)
	}
	valid, err := firstInt(outs[1])
	if err != nil {
		return nil, 0, err
	}
	trimmed, err := trimFrames(data, int(shape[1]), int(shape[2]), valid)
	if err != nil {
		return nil, 0, err
	}
	return trimmed, valid, nil
}

// trimFrames keeps the first `valid` columns of a [rows, cols] row-major matrix.
func trimFrames(data []float32, rows, cols, valid int) ([]float32, error) {
	if valid < 0 || valid > cols {
		return nil, fmt.Errorf("valid frames %d out of range [0, %d]", valid, cols)
	}
	if valid == cols {
		return data, nil
	}
	out := make([]float32, rows*valid)
	for r := 0; r < rows; r++ {
		copy(out[r*valid:(r+1)*valid], data[r*cols:r*cols+valid])
	}
	return out, nil
}

type lstmState struct{ h, c []float32 }

type ortJoint struct {
	g      *graphs
	enc    []float32 // [D, frames]
	frames int
}

func (g *graphs) joint(enc []float32, frames int) asr.Joint {
	return &ortJoint{g: g, enc: enc, frames: frames}
}

func (j *ortJoint) InitialState() asr.DecoderState {
	n := j.g.m.PredLayers * j.g.m.PredHidden
	return lstmState{h: make([]float32, n), c: make([]float32, n)}
}

func (j *ortJoint) Step(t, last int, st asr.DecoderState) ([]float32, []float32, asr.DecoderState, error) {
	m, dj := j.g.m, j.g.dj
	if t < 0 || t >= j.frames {
		return nil, nil, nil, fmt.Errorf("joint frame %d out of range [0, %d)", t, j.frames)
	}
	s := st.(lstmState)
	frame := make([]float32, m.EncoderDim)
	for d := range frame {
		frame[d] = j.enc[d*j.frames+t]
	}
	stateShape := ort.NewShape(int64(m.PredLayers), 1, int64(m.PredHidden))
	var ins []ort.Value
	defer func() { destroyAll(ins) }()
	add := func(v ort.Value, err error) error {
		if err != nil {
			return err
		}
		ins = append(ins, v)
		return nil
	}
	if err := add(floatTensor(ort.NewShape(1, int64(m.EncoderDim), 1), frame)); err != nil {
		return nil, nil, nil, err
	}
	if err := add(intTensor(dj.inDType(1), ort.NewShape(1, 1), int64(last))); err != nil {
		return nil, nil, nil, err
	}
	if err := add(intTensor(dj.inDType(2), ort.NewShape(1), 1)); err != nil {
		return nil, nil, nil, err
	}
	if err := add(floatTensor(stateShape, s.h)); err != nil {
		return nil, nil, nil, err
	}
	if err := add(floatTensor(stateShape, s.c)); err != nil {
		return nil, nil, nil, err
	}
	outs, err := dj.run(ins)
	if err != nil {
		return nil, nil, nil, err
	}
	defer destroyAll(outs)
	logits, _, err := floats(outs[0])
	if err != nil {
		return nil, nil, nil, err
	}
	h, _, err := floats(outs[2])
	if err != nil {
		return nil, nil, nil, err
	}
	c, _, err := floats(outs[3])
	if err != nil {
		return nil, nil, nil, err
	}
	nTok := m.BlankID + 1
	if len(logits) != nTok+len(m.Durations) {
		return nil, nil, nil, fmt.Errorf("joint logits %d, want %d", len(logits), nTok+len(m.Durations))
	}
	return logits[:nTok], logits[nTok:], lstmState{h: h, c: c}, nil
}
