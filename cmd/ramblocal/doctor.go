package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/thegr8yg/ramblocal/internal/asr"
	"github.com/thegr8yg/ramblocal/internal/asr/ortengine"
	"github.com/thegr8yg/ramblocal/internal/bench"
)

// doctorEngine is the part of ortengine.Engine that doctor uses. Tests replace newDoctorEngine with a fake.
type doctorEngine interface {
	asr.Engine
	EP() string
}

// gpuInfo is what doctor reports about the first GPU.
type gpuInfo struct {
	Name, Driver    string
	TotalMB, UsedMB int
}

// Hooks replaced by tests so doctor runs without a model, ONNX Runtime or GPU.
var (
	newDoctorEngine = func(cfg ortengine.Config) (doctorEngine, error) {
		e, err := ortengine.New(cfg)
		if err != nil {
			return nil, err
		}
		return e, nil
	}
	queryGPU = queryGPUInfo
)

// doctorReport prints one line per check and counts the FAIL lines.
type doctorReport struct {
	out   io.Writer
	fails int
}

func (r *doctorReport) line(status, check, detail string) {
	if status == "FAIL" {
		r.fails++
	}
	fmt.Fprintf(r.out, "%-5s %-12s %s\n", status, check, detail)
}

func runDoctor(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelFlag := fs.String("model", "", "model directory (default: $RAMBLOCAL_MODEL_DIR, then models/parakeet-tdt-0.6b-v2)")
	ortFlag := fs.String("ort", "", "ONNX Runtime shared library (default: $RAMBLOCAL_ORT_LIB, then next to the executable)")
	epFlag := fs.String("ep", "", "execution providers to try, comma-separated (default cuda,dml,cpu)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	r := &doctorReport{out: stdout}
	modelDir, modelOK := checkModel(r, *modelFlag)
	ortLib, ortOK := checkORT(r, *ortFlag)
	gpuBefore, gpuOK := checkGPU(r)

	if !modelOK || !ortOK {
		r.line("skip", "engine", "model or ONNX Runtime check failed")
	} else {
		eps := splitEPs(*epFlag)
		cudaWanted := hasEP(eps, "cuda") || (len(eps) == 0 && gpuOK)
		checkEngine(r, ortengine.Config{ModelDir: modelDir, ORTLib: ortLib, EPs: eps}, cudaWanted, gpuOK, gpuBefore)
	}

	if r.fails > 0 {
		return fmt.Errorf("%d check(s) failed", r.fails)
	}
	return nil
}

// checkModel finds the model directory and verifies the manifest and the files it names.
func checkModel(r *doctorReport, flagVal string) (string, bool) {
	dir, err := findModelDir(flagVal)
	if err != nil {
		r.line("FAIL", "model", err.Error())
		return "", false
	}
	r.line("ok", "model", dir)
	m, err := asr.LoadManifest(filepath.Join(dir, "manifest.json"))
	if err != nil {
		r.line("FAIL", "manifest", err.Error())
		return dir, false
	}
	r.line("ok", "manifest", m.Name)
	ok := true
	files := modelFiles(m)
	for _, f := range files {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			r.line("FAIL", "model file", "missing "+f)
			ok = false
		}
	}
	if ok {
		r.line("ok", "model files", fmt.Sprintf("%d present", len(files)))
	}
	return dir, ok
}

// modelFiles lists what the engine opens besides manifest.json: the vocabulary, the encoder and
// decoder/joint graphs, and either the ONNX preprocessor or, when the manifest has a frontend
// section, the filterbank and window files that replace it.
func modelFiles(m *asr.Manifest) []string {
	files := []string{"vocab.txt", m.Encoder.File, m.DecoderJoint.File}
	if m.Frontend == nil {
		return append(files, m.Preprocessor.File)
	}
	return append(files, m.Frontend.Filterbank, m.Frontend.Window)
}

func checkORT(r *doctorReport, flagVal string) (string, bool) {
	lib, err := findORTLib(flagVal)
	if err != nil {
		r.line("FAIL", "onnxruntime", err.Error())
		return "", false
	}
	r.line("ok", "onnxruntime", lib)
	return lib, true
}

// checkGPU reports the GPU. A failure is only a warning because the CPU path still works.
func checkGPU(r *doctorReport) (gpuInfo, bool) {
	g, err := queryGPU()
	if err != nil {
		r.line("warn", "gpu", fmt.Sprintf("not available (%v); CPU still works", err))
		return gpuInfo{}, false
	}
	driver := g.Driver
	if driver == "" {
		driver = "unknown"
	}
	vram := fmt.Sprintf("%d MB used", g.UsedMB)
	if g.TotalMB > 0 {
		vram += fmt.Sprintf(" of %d MB", g.TotalMB)
	}
	r.line("ok", "gpu", fmt.Sprintf("%s, driver %s, VRAM %s", g.Name, driver, vram))
	return g, true
}

// checkEngine loads the engine, reports the execution provider and VRAM cost, then runs a
// one-second silence self-test.
func checkEngine(r *doctorReport, cfg ortengine.Config, cudaWanted, gpuOK bool, gpuBefore gpuInfo) {
	t0 := time.Now()
	eng, err := newDoctorEngine(cfg)
	cold := time.Since(t0)
	if err != nil {
		r.line("FAIL", "engine", err.Error())
		return
	}
	defer eng.Close()

	ep := eng.EP()
	r.line("ok", "engine", fmt.Sprintf("ep %s, cold start %s", ep, cold.Round(time.Millisecond)))
	if gpuOK {
		if after, err := queryGPU(); err != nil {
			r.line("warn", "vram", "query after load failed: "+err.Error())
		} else {
			r.line("ok", "vram", fmt.Sprintf("%d MB used after load (%+d MB since before load)", after.UsedMB, after.UsedMB-gpuBefore.UsedMB))
		}
	}
	if cudaWanted && ep != "cuda" {
		r.line("warn", "ep", fmt.Sprintf("cuda was expected but the engine fell back to %s (the cuda provider did not load the model)", ep))
	}

	t1 := time.Now()
	res, err := eng.Transcribe(context.Background(), make([]float32, asr.SampleRate), asr.Options{})
	if err != nil {
		r.line("FAIL", "self-test", err.Error())
		return
	}
	r.line("ok", "self-test", fmt.Sprintf("1 s of silence in %s (text %q)", time.Since(t1).Round(time.Millisecond), res.Text))
}

// queryGPUInfo takes the name and used VRAM from bench.QueryGPU. That call does not report the
// driver or total VRAM, so nvidia-smi is queried once more for them; if that fails they are left out.
func queryGPUInfo() (gpuInfo, error) {
	name, used, err := bench.QueryGPU()
	if err != nil {
		return gpuInfo{}, err
	}
	g := gpuInfo{Name: name, UsedMB: used}
	out, err := exec.Command("nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader,nounits").Output()
	if err == nil {
		parts := strings.Split(strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)[0], ",")
		if len(parts) == 2 {
			g.Driver = strings.TrimSpace(parts[0])
			g.TotalMB, _ = strconv.Atoi(strings.TrimSpace(parts[1]))
		}
	}
	return g, nil
}

// splitEPs parses the --ep list; an empty result means the engine default order.
func splitEPs(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func hasEP(eps []string, want string) bool {
	for _, e := range eps {
		if e == want {
			return true
		}
	}
	return false
}
