package bench

import "testing"

func TestPercentiles(t *testing.T) {
	var ms []float64
	for i := 1; i <= 100; i++ {
		ms = append(ms, float64(i))
	}
	p := Percentiles(ms)
	if p.P50 != 50 || p.P95 != 95 || p.Max != 100 {
		t.Fatalf("%+v", p)
	}
	if p := Percentiles([]float64{7}); p.P50 != 7 || p.P95 != 7 {
		t.Fatalf("single: %+v", p)
	}
	if p := Percentiles(nil); p != (Pct{}) {
		t.Fatalf("empty: %+v", p)
	}
}
