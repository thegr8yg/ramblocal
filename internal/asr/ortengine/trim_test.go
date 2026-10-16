package ortengine

import (
	"reflect"
	"testing"
)

func TestTrimFrames(t *testing.T) {
	// 2 rows x 4 cols, row-major.
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	cases := []struct {
		name    string
		valid   int
		want    []float32
		wantErr bool
	}{
		{name: "valid less than cols trims each row", valid: 2, want: []float32{1, 2, 5, 6}},
		{name: "valid zero keeps no frames", valid: 0, want: []float32{}},
		{name: "valid equal to cols returns data unchanged", valid: 4, want: data},
		{name: "valid greater than cols is an error", valid: 5, wantErr: true},
		{name: "negative valid is an error", valid: -1, wantErr: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := trimFrames(append([]float32(nil), data...), 2, 4, c.valid)
			if c.wantErr {
				if err == nil {
					t.Fatalf("trimFrames(valid=%d): expected error, got %v", c.valid, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("trimFrames(valid=%d): unexpected error %v", c.valid, err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Fatalf("trimFrames(valid=%d) = %v, want %v", c.valid, got, c.want)
			}
		})
	}
}
