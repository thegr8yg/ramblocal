package asr

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
)

const wordStart = "▁" // U+2581, SentencePiece word-boundary marker

// Tokenizer maps SentencePiece token ids to text.
type Tokenizer struct {
	pieces []string
}

// LoadVocab reads one piece per line; the line index is the token id.
func LoadVocab(r io.Reader) (*Tokenizer, error) {
	var pieces []string
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		pieces = append(pieces, strings.TrimRight(sc.Text(), "\r"))
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(pieces) == 0 {
		return nil, errors.New("vocab is empty")
	}
	return &Tokenizer{pieces: pieces}, nil
}

func (t *Tokenizer) Size() int { return len(t.pieces) }

func (t *Tokenizer) piece(id int) (string, error) {
	if id < 0 || id >= len(t.pieces) {
		return "", fmt.Errorf("token id %d out of range [0,%d)", id, len(t.pieces))
	}
	return t.pieces[id], nil
}

func (t *Tokenizer) Decode(ids []int) (string, error) {
	var b strings.Builder
	for _, id := range ids {
		p, err := t.piece(id)
		if err != nil {
			return "", err
		}
		b.WriteString(p)
	}
	return strings.TrimSpace(strings.ReplaceAll(b.String(), wordStart, " ")), nil
}

// Words groups tokens into words. frames[i] is the encoder frame where ids[i] was emitted.
func (t *Tokenizer) Words(ids, frames []int, frameSec float64) ([]Word, error) {
	if len(ids) != len(frames) {
		return nil, fmt.Errorf("ids (%d) and frames (%d) differ in length", len(ids), len(frames))
	}
	var words []Word
	for i, id := range ids {
		p, err := t.piece(id)
		if err != nil {
			return nil, err
		}
		start := float64(frames[i]) * frameSec
		end := float64(frames[i]+1) * frameSec
		if strings.HasPrefix(p, wordStart) || len(words) == 0 {
			words = append(words, Word{Text: strings.TrimPrefix(p, wordStart), Start: start, End: end})
			continue
		}
		w := &words[len(words)-1]
		if w.Text == "" {
			w.Start = start // a bare "▁" piece opened this word; it starts with this token
		}
		w.Text += p
		w.End = end
	}
	out := words[:0]
	for _, w := range words {
		if w.Text != "" {
			out = append(out, w)
		}
	}
	return out, nil
}
