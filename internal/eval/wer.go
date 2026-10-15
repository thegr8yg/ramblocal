// Package eval computes accuracy metrics.
package eval

import (
	"strings"

	"github.com/thegr8yg/ramblocal/internal/text"
)

type Corpus struct {
	Errors, RefWords, Utterances int
}

func (c *Corpus) Add(ref, hyp string) {
	r := strings.Fields(text.NormalizeEN(ref))
	h := strings.Fields(text.NormalizeEN(hyp))
	c.Errors += EditDistance(r, h)
	c.RefWords += len(r)
	c.Utterances++
}

// WER returns the corpus word error rate in percent.
func (c *Corpus) WER() float64 {
	if c.RefWords == 0 {
		return 0
	}
	return 100 * float64(c.Errors) / float64(c.RefWords)
}

// EditDistance is the word-level Levenshtein distance (substitutions + deletions + insertions).
func EditDistance(ref, hyp []string) int {
	prev := make([]int, len(hyp)+1)
	cur := make([]int, len(hyp)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ref); i++ {
		cur[0] = i
		for j := 1; j <= len(hyp); j++ {
			cost := 1
			if ref[i-1] == hyp[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev, cur = cur, prev
	}
	return prev[len(hyp)]
}
