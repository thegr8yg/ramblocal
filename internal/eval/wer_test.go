package eval

import "testing"

func TestEditDistance(t *testing.T) {
	cases := []struct {
		ref, hyp []string
		want     int
	}{
		{[]string{"a", "b", "c"}, []string{"a", "b", "c"}, 0},
		{[]string{"a", "b", "c"}, []string{"a", "x", "c"}, 1},
		{[]string{"a", "b", "c"}, nil, 3},
		{nil, []string{"a"}, 1},
		{[]string{"a", "b"}, []string{"a", "b", "c"}, 1},
		{[]string{"a", "b", "c", "d"}, []string{"b", "c", "d", "e"}, 2},
	}
	for _, c := range cases {
		if got := EditDistance(c.ref, c.hyp); got != c.want {
			t.Errorf("EditDistance(%v, %v) = %d, want %d", c.ref, c.hyp, got, c.want)
		}
	}
}

func TestCorpusWER(t *testing.T) {
	var c Corpus
	c.Add("The cat sat.", "the cat sat") // 0 errors, 3 words
	c.Add("on the mat", "on a mat")      // 1 error, 3 words
	if c.Errors != 1 || c.RefWords != 6 || c.Utterances != 2 {
		t.Fatalf("corpus = %+v", c)
	}
	if w := c.WER(); w < 16.66 || w > 16.67 {
		t.Fatalf("WER = %v, want 16.67", w)
	}
	var empty Corpus
	if empty.WER() != 0 {
		t.Fatal("empty corpus WER must be 0")
	}
}
