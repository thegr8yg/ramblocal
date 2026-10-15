// Package text holds text processing shared by evaluation and the dictation pipeline.
package text

import (
	"strings"
	"unicode"
)

// NormalizeEN lowercases, removes apostrophes, turns other punctuation into spaces, and collapses whitespace.
func NormalizeEN(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
		case r == '\'' || r == '’':
			// "don't" -> "dont"
		default:
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}
