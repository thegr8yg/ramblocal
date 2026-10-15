// Package bench measures accuracy and speed of ASR engines.
package bench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

type Item struct {
	Audio string `json:"audio"`
	Text  string `json:"text"`
}

type HypLine struct {
	Audio   string  `json:"audio"`
	Hyp     string  `json:"hyp"`
	Seconds float64 `json:"seconds,omitempty"`
}

func LoadItems(r io.Reader) ([]Item, error)   { return loadJSONL[Item](r) }
func LoadHyps(r io.Reader) ([]HypLine, error) { return loadJSONL[HypLine](r) }

func loadJSONL[T any](r io.Reader) ([]T, error) {
	var out []T
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var v T
		if err := json.Unmarshal([]byte(line), &v); err != nil {
			return nil, fmt.Errorf("line %d: %w", n, err)
		}
		out = append(out, v)
	}
	return out, sc.Err()
}
