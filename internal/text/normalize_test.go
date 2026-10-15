package text

import "testing"

func TestNormalizeEN(t *testing.T) {
	cases := map[string]string{
		"Hello, World!":         "hello world",
		"DON'T stop":            "dont stop",
		"don’t stop":            "dont stop",
		"well-known":            "well known",
		"  a   b ":              "a b",
		"":                      "",
		"Café au lait.":         "café au lait",
		"He said: \"20 ducks\"": "he said 20 ducks",
	}
	for in, want := range cases {
		if got := NormalizeEN(in); got != want {
			t.Errorf("NormalizeEN(%q) = %q, want %q", in, got, want)
		}
	}
}
