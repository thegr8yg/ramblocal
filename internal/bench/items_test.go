package bench

import (
	"strings"
	"testing"
)

func TestLoadItems(t *testing.T) {
	items, err := LoadItems(strings.NewReader("{\"audio\":\"a.wav\",\"text\":\"hi\"}\n\n{\"audio\":\"b.wav\",\"text\":\"yo\"}\n"))
	if err != nil || len(items) != 2 || items[1].Audio != "b.wav" {
		t.Fatalf("%+v %v", items, err)
	}
	if _, err := LoadItems(strings.NewReader("{\"audio\":\"a.wav\"}\nnot json\n")); err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("want line-numbered error, got %v", err)
	}
}
