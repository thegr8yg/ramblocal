package capture

import (
	"math"
	"math/rand"
	"testing"
)

func approx(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func constBlock(v float32, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = v
	}
	return s
}

func TestRMSDB(t *testing.T) {
	cases := []struct {
		name  string
		block []float32
		want  float32
	}{
		{"empty", nil, silenceDB},
		{"silence", constBlock(0, 320), silenceDB},
		{"full scale constant", constBlock(1, 320), 0},
		{"half scale constant", constBlock(0.5, 320), -6.0206},
		{"tenth scale constant", constBlock(0.1, 320), -20},
	}
	for _, c := range cases {
		if got := rmsDB(c.block); !approx(got, c.want, 0.01) {
			t.Errorf("%s: rmsDB = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestDBToUnit(t *testing.T) {
	cases := []struct{ db, want float32 }{
		{-120, 0},
		{-90, 0},
		{-60, 0},
		{-30, 0.5},
		{-15, 0.75},
		{0, 1},
		{6, 1},
	}
	for _, c := range cases {
		if got := dbToUnit(c.db); !approx(got, c.want, 1e-6) {
			t.Errorf("dbToUnit(%v) = %v, want %v", c.db, got, c.want)
		}
	}
}

func TestMeterAttackFasterThanRelease(t *testing.T) {
	var m meter
	m.update(constBlock(1, 320))
	up := m.get()
	if !approx(up, meterAttack, 1e-6) {
		t.Fatalf("after one loud block: %v, want %v", up, meterAttack)
	}
	m.update(nil) // empty blocks are ignored
	if m.get() != up {
		t.Fatal("empty block changed the level")
	}
	m.update(make([]float32, 320)) // silence
	down := m.get()
	if want := up * (1 - meterRelease); !approx(down, want, 1e-6) {
		t.Fatalf("after one silent block: %v, want %v", down, want)
	}
	if fall, rise := up-down, up; fall >= rise { // release must be slower than attack
		t.Fatalf("falling step %v is not smaller than rising step %v", fall, rise)
	}
}

func TestMeterStaysInRangeAndSettles(t *testing.T) {
	var m meter
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 1000; i++ {
		block := make([]float32, 320)
		amp := rng.Float32() * 1.5 // includes over-range input
		for j := range block {
			block[j] = (rng.Float32()*2 - 1) * amp
		}
		m.update(block)
		if v := m.get(); v < 0 || v > 1 {
			t.Fatalf("level out of range: %v", v)
		}
	}
	for i := 0; i < 200; i++ {
		m.update(constBlock(0.5, 320)) // about -6 dB, expect about 0.9
	}
	if v := m.get(); !approx(v, 0.9, 0.01) {
		t.Fatalf("settled level = %v, want about 0.9", v)
	}
}
