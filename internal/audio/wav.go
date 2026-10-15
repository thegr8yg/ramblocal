// Package audio handles PCM audio I/O.
package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

var (
	ErrNotWAV            = errors.New("not a RIFF/WAVE file")
	ErrUnsupportedFormat = errors.New("unsupported WAV sample format")
)

const (
	fmtPCM        = 1
	fmtFloat      = 3
	fmtExtensible = 0xFFFE
)

// ReadWAV decodes PCM 16/24/32-bit integer or 32-bit float WAV, including WAVE_FORMAT_EXTENSIBLE.
// Channels are averaged to mono. The caller checks the sample rate.
func ReadWAV(r io.Reader) ([]float32, int, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, 0, err
	}
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, ErrNotWAV
	}
	le := binary.LittleEndian
	var format, channels, bits uint16
	var rate uint32
	var pcm []byte
	haveFmt, haveData := false, false
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := int(le.Uint32(data[off+4 : off+8]))
		body := off + 8
		if size > len(data)-body {
			size = len(data) - body // tolerate a truncated final chunk
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, 0, fmt.Errorf("%w: fmt chunk too small", ErrNotWAV)
			}
			format = le.Uint16(data[body:])
			channels = le.Uint16(data[body+2:])
			rate = le.Uint32(data[body+4:])
			bits = le.Uint16(data[body+14:])
			if format == fmtExtensible {
				if size < 26 {
					return nil, 0, fmt.Errorf("%w: extensible fmt chunk too small", ErrNotWAV)
				}
				format = le.Uint16(data[body+24:]) // first two bytes of SubFormat GUID
			}
			haveFmt = true
		case "data":
			pcm = data[body : body+size]
			haveData = true
		}
		off = body + size + size%2
	}
	if !haveFmt || !haveData {
		return nil, 0, fmt.Errorf("%w: missing fmt or data chunk", ErrNotWAV)
	}
	if channels == 0 {
		return nil, 0, fmt.Errorf("%w: zero channels", ErrUnsupportedFormat)
	}

	var sample func(b []byte) float32
	switch {
	case format == fmtPCM && bits == 16:
		sample = func(b []byte) float32 { return float32(int16(le.Uint16(b))) / 32768 }
	case format == fmtPCM && bits == 24:
		sample = func(b []byte) float32 {
			v := int32(b[0]) | int32(b[1])<<8 | int32(int8(b[2]))<<16
			return float32(v) / 8388608
		}
	case format == fmtPCM && bits == 32:
		sample = func(b []byte) float32 { return float32(int32(le.Uint32(b))) / 2147483648 }
	case format == fmtFloat && bits == 32:
		sample = func(b []byte) float32 { return math.Float32frombits(le.Uint32(b)) }
	default:
		return nil, 0, fmt.Errorf("%w: format %d, %d bits", ErrUnsupportedFormat, format, bits)
	}

	bps := int(bits) / 8
	frame := bps * int(channels)
	n := len(pcm) / frame
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for c := 0; c < int(channels); c++ {
			o := i*frame + c*bps
			sum += sample(pcm[o : o+bps])
		}
		out[i] = sum / float32(channels)
	}
	return out, int(rate), nil
}

// WriteWAV16 writes mono 16-bit PCM, clamping samples to [-1, 1].
func WriteWAV16(w io.Writer, pcm []float32, sampleRate int) error {
	var b bytes.Buffer
	le := binary.LittleEndian
	dataLen := uint32(len(pcm) * 2)
	b.WriteString("RIFF")
	binary.Write(&b, le, 36+dataLen)
	b.WriteString("WAVEfmt ")
	binary.Write(&b, le, uint32(16))
	binary.Write(&b, le, uint16(fmtPCM))
	binary.Write(&b, le, uint16(1))
	binary.Write(&b, le, uint32(sampleRate))
	binary.Write(&b, le, uint32(sampleRate*2))
	binary.Write(&b, le, uint16(2))
	binary.Write(&b, le, uint16(16))
	b.WriteString("data")
	binary.Write(&b, le, dataLen)
	for _, s := range pcm {
		s = max(-1, min(1, s))
		binary.Write(&b, le, int16(math.Round(float64(s)*32767)))
	}
	_, err := w.Write(b.Bytes())
	return err
}
