package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// buildWAV assembles a RIFF/WAVE file from a fmt body, optional extra chunks, and raw data.
func buildWAV(fmtBody []byte, extra [][2][]byte, data []byte) []byte {
	var body bytes.Buffer
	body.WriteString("WAVE")
	chunk := func(id string, b []byte) {
		body.WriteString(id)
		binary.Write(&body, binary.LittleEndian, uint32(len(b)))
		body.Write(b)
		if len(b)%2 == 1 {
			body.WriteByte(0)
		}
	}
	chunk("fmt ", fmtBody)
	for _, e := range extra {
		chunk(string(e[0]), e[1])
	}
	chunk("data", data)
	var out bytes.Buffer
	out.WriteString("RIFF")
	binary.Write(&out, binary.LittleEndian, uint32(body.Len()))
	out.Write(body.Bytes())
	return out.Bytes()
}

func fmtChunk(format, channels uint16, rate uint32, bits uint16) []byte {
	b := make([]byte, 16)
	binary.LittleEndian.PutUint16(b[0:], format)
	binary.LittleEndian.PutUint16(b[2:], channels)
	binary.LittleEndian.PutUint32(b[4:], rate)
	blockAlign := channels * bits / 8
	binary.LittleEndian.PutUint32(b[8:], rate*uint32(blockAlign))
	binary.LittleEndian.PutUint16(b[12:], blockAlign)
	binary.LittleEndian.PutUint16(b[14:], bits)
	return b
}

func near(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-3 }

func TestRoundTrip16(t *testing.T) {
	in := []float32{0, 0.5, -0.5, 0.25}
	var buf bytes.Buffer
	if err := WriteWAV16(&buf, in, 16000); err != nil {
		t.Fatal(err)
	}
	out, rate, err := ReadWAV(&buf)
	if err != nil || rate != 16000 || len(out) != len(in) {
		t.Fatalf("rate=%d len=%d err=%v", rate, len(out), err)
	}
	for i := range in {
		if !near(in[i], out[i]) {
			t.Fatalf("sample %d: %v != %v", i, out[i], in[i])
		}
	}
}

func TestStereoDownmix(t *testing.T) {
	data := make([]byte, 4)
	l, r := int16(16384), int16(-16384) // L = 0.5, R = -0.5 (variables: constant uint16(negative) does not compile)
	binary.LittleEndian.PutUint16(data[0:], uint16(l))
	binary.LittleEndian.PutUint16(data[2:], uint16(r))
	out, _, err := ReadWAV(bytes.NewReader(buildWAV(fmtChunk(1, 2, 16000, 16), nil, data)))
	if err != nil || len(out) != 1 || !near(out[0], 0) {
		t.Fatalf("out=%v err=%v", out, err)
	}
}

func TestFloat32AndExtensibleAnd24Bit(t *testing.T) {
	f := make([]byte, 4)
	binary.LittleEndian.PutUint32(f, math.Float32bits(0.75))
	out, _, err := ReadWAV(bytes.NewReader(buildWAV(fmtChunk(3, 1, 16000, 32), nil, f)))
	if err != nil || !near(out[0], 0.75) {
		t.Fatalf("float: %v %v", out, err)
	}

	ext := append(fmtChunk(0xFFFE, 1, 16000, 16), make([]byte, 24)...) // cbSize..SubFormat
	binary.LittleEndian.PutUint16(ext[16:], 22)                        // cbSize
	binary.LittleEndian.PutUint16(ext[24:], 1)                         // SubFormat starts with PCM tag
	d := make([]byte, 2)
	min16 := int16(-32768)
	binary.LittleEndian.PutUint16(d, uint16(min16))
	out, _, err = ReadWAV(bytes.NewReader(buildWAV(ext, nil, d)))
	if err != nil || !near(out[0], -1) {
		t.Fatalf("extensible: %v %v", out, err)
	}

	d24 := []byte{0x00, 0x00, 0x40} // 0x400000 = 0.5 * 2^23
	out, _, err = ReadWAV(bytes.NewReader(buildWAV(fmtChunk(1, 1, 16000, 24), nil, d24)))
	if err != nil || !near(out[0], 0.5) {
		t.Fatalf("24-bit: %v %v", out, err)
	}
}

func TestOddChunkBeforeData(t *testing.T) {
	d := make([]byte, 2)
	binary.LittleEndian.PutUint16(d, uint16(int16(16384)))
	extra := [][2][]byte{{[]byte("LIST"), []byte("abc")}} // odd size, padded
	out, _, err := ReadWAV(bytes.NewReader(buildWAV(fmtChunk(1, 1, 16000, 16), extra, d)))
	if err != nil || len(out) != 1 || !near(out[0], 0.5) {
		t.Fatalf("out=%v err=%v", out, err)
	}
}

func TestErrors(t *testing.T) {
	if _, _, err := ReadWAV(bytes.NewReader([]byte("not a wav file at all"))); !errors.Is(err, ErrNotWAV) {
		t.Fatalf("garbage: %v", err)
	}
	_, _, err := ReadWAV(bytes.NewReader(buildWAV(fmtChunk(1, 1, 16000, 8), nil, []byte{1})))
	if !errors.Is(err, ErrUnsupportedFormat) {
		t.Fatalf("8-bit: %v", err)
	}
}

// riffRaw builds a RIFF/WAVE file from pre-encoded chunks, with no implicit fmt or data chunk.
func riffRaw(chunks ...[]byte) []byte {
	body := append([]byte("WAVE"), bytes.Join(chunks, nil)...)
	var out bytes.Buffer
	out.WriteString("RIFF")
	binary.Write(&out, binary.LittleEndian, uint32(len(body)))
	out.Write(body)
	return out.Bytes()
}

// chunkRaw encodes one RIFF chunk, padded to an even length.
func chunkRaw(id string, b []byte) []byte {
	var out bytes.Buffer
	out.WriteString(id)
	binary.Write(&out, binary.LittleEndian, uint32(len(b)))
	out.Write(b)
	if len(b)%2 == 1 {
		out.WriteByte(0)
	}
	return out.Bytes()
}

func TestMalformedInput(t *testing.T) {
	pcm16 := func(vals ...int16) []byte {
		b := make([]byte, 2*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
		}
		return b
	}

	// Truncated data chunk: copy of a valid file with the data size field inflated.
	truncData := pcm16(16384, -16384)
	truncated := buildWAV(fmtChunk(1, 1, 16000, 16), nil, truncData)
	binary.LittleEndian.PutUint32(truncated[len(truncated)-len(truncData)-4:], 1000)

	// Extensible fmt chunk, 18 bytes: long enough for the base header, too short for SubFormat.
	shortExt := append(fmtChunk(0xFFFE, 1, 16000, 16), 0, 0) // 16 + cbSize = 18 bytes

	// 32-bit PCM value 0x40000000 = 0.5 * 2^31.
	pcm32 := make([]byte, 4)
	binary.LittleEndian.PutUint32(pcm32, 0x40000000)

	cases := []struct {
		name    string
		in      []byte
		wantErr error
		want    []float32
	}{
		{
			name: "truncated data chunk decodes available samples",
			in:   truncated,
			want: []float32{0.5, -0.5},
		},
		{
			name:    "fmt chunk shorter than 16 bytes",
			in:      buildWAV(make([]byte, 14), nil, pcm16(16384)),
			wantErr: ErrNotWAV,
		},
		{
			name:    "extensible fmt chunk shorter than 26 bytes",
			in:      buildWAV(shortExt, nil, pcm16(16384)),
			wantErr: ErrNotWAV,
		},
		{
			name:    "zero channels",
			in:      buildWAV(fmtChunk(1, 0, 16000, 16), nil, pcm16(16384)),
			wantErr: ErrUnsupportedFormat,
		},
		{
			name:    "fmt chunk without data chunk",
			in:      riffRaw(chunkRaw("fmt ", fmtChunk(1, 1, 16000, 16))),
			wantErr: ErrNotWAV,
		},
		{
			name:    "data chunk without fmt chunk",
			in:      riffRaw(chunkRaw("data", pcm16(16384))),
			wantErr: ErrNotWAV,
		},
		{
			name: "32-bit PCM 0x40000000 decodes to 0.5",
			in:   buildWAV(fmtChunk(1, 1, 16000, 32), nil, pcm32),
			want: []float32{0.5},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, rate, err := ReadWAV(bytes.NewReader(tc.in))
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err=%v, want wrapping %v", err, tc.wantErr)
				}
				return
			}
			if err != nil || rate != 16000 {
				t.Fatalf("rate=%d err=%v", rate, err)
			}
			if len(out) != len(tc.want) {
				t.Fatalf("got %d samples %v, want %d", len(out), out, len(tc.want))
			}
			for i := range tc.want {
				if !near(out[i], tc.want[i]) {
					t.Fatalf("sample %d: %v != %v", i, out[i], tc.want[i])
				}
			}
		})
	}
}
