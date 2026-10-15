package asr

import (
	"strings"
	"testing"
)

const testVocab = "<unk>\n▁the\n▁cat\ns\n.\n▁\na\n"

func loadTestVocab(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := LoadVocab(strings.NewReader(testVocab))
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestDecode(t *testing.T) {
	tok := loadTestVocab(t)
	if tok.Size() != 7 {
		t.Fatalf("Size = %d", tok.Size())
	}
	cases := []struct {
		ids  []int
		want string
	}{
		{[]int{1, 2, 3, 4}, "the cats."},
		{[]int{5, 6}, "a"},
		{nil, ""},
	}
	for _, c := range cases {
		got, err := tok.Decode(c.ids)
		if err != nil || got != c.want {
			t.Errorf("Decode(%v) = %q, %v; want %q", c.ids, got, err, c.want)
		}
	}
	if _, err := tok.Decode([]int{99}); err == nil {
		t.Error("out-of-range id: expected error")
	}
}

func TestWords(t *testing.T) {
	tok := loadTestVocab(t)
	words, err := tok.Words([]int{1, 2, 3, 4, 5, 6}, []int{0, 3, 4, 4, 9, 10}, 0.08)
	if err != nil {
		t.Fatal(err)
	}
	// A bare "▁" piece (frame 9) opens a word; the word starts at its first visible piece (frame 10).
	want := []Word{
		{"the", 0, 0.08},
		{"cats.", 0.24, 0.40},
		{"a", 0.80, 0.88},
	}
	if len(words) != len(want) {
		t.Fatalf("got %+v", words)
	}
	for i := range want {
		w, g := want[i], words[i]
		if g.Text != w.Text || !approx(g.Start, w.Start) || !approx(g.End, w.End) {
			t.Errorf("word %d = %+v, want %+v", i, g, w)
		}
	}
	if _, err := tok.Words([]int{1}, nil, 0.08); err == nil {
		t.Error("length mismatch: expected error")
	}
}

func TestLoadVocabEmpty(t *testing.T) {
	if _, err := LoadVocab(strings.NewReader("")); err == nil {
		t.Fatal("expected error")
	}
}

func approx(a, b float64) bool { d := a - b; return d < 1e-9 && d > -1e-9 }
