package inject

import (
	"errors"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode/utf16"
	"unsafe"
)

var (
	errBusy  = errors.New("clipboard busy")
	errRead  = errors.New("read failed")
	errWrite = errors.New("write failed")
	errSend  = errors.New("send failed")
)

// fakeAPI simulates the clipboard and SendInput. It records every call in log so tests can
// check ordering.
type fakeAPI struct {
	clip    []uint16
	hasText bool
	open    bool

	openFails  int // OpenClipboard calls that fail before one succeeds
	attempts   int
	opens      int
	closes     int
	reads      int
	readFailAt int   // 1-based ReadText call that fails with errRead (0 = never)
	writeErr   error // returned by every WriteText

	// sendFn scripts SendInput results; call is 1-based. nil accepts every event.
	sendFn func(call int, in []input) (int, error)
	// onSend runs when SendInput is called, to simulate the target app or the user.
	onSend func(f *fakeAPI)

	sends      [][]input
	openAtSend bool
	textAtSend string
	sleeps     []time.Duration
	log        []string
}

func newFake(text string) *fakeAPI {
	return &fakeAPI{clip: encodeUTF16(text), hasText: true}
}

func (f *fakeAPI) OpenClipboard() error {
	f.attempts++
	f.log = append(f.log, "open")
	if f.openFails > 0 {
		f.openFails--
		return errBusy
	}
	f.open = true
	f.opens++
	return nil
}

func (f *fakeAPI) CloseClipboard() error {
	f.log = append(f.log, "close")
	if !f.open {
		return errors.New("close without open clipboard")
	}
	f.open = false
	f.closes++
	return nil
}

func (f *fakeAPI) ReadText() ([]uint16, bool, error) {
	f.reads++
	f.log = append(f.log, "read")
	if !f.open {
		return nil, false, errors.New("read without open clipboard")
	}
	if f.reads == f.readFailAt {
		return nil, false, errRead
	}
	if !f.hasText {
		return nil, false, nil
	}
	return append([]uint16(nil), f.clip...), true, nil
}

func (f *fakeAPI) WriteText(units []uint16) error {
	f.log = append(f.log, "write")
	if !f.open {
		return errors.New("write without open clipboard")
	}
	if f.writeErr != nil {
		return f.writeErr
	}
	f.clip = append([]uint16(nil), units...)
	f.hasText = true
	return nil
}

func (f *fakeAPI) SendInput(in []input) (int, error) {
	f.log = append(f.log, "send")
	f.sends = append(f.sends, append([]input(nil), in...))
	f.openAtSend = f.open
	f.textAtSend = f.clipText()
	if f.onSend != nil {
		f.onSend(f)
	}
	if f.sendFn != nil {
		return f.sendFn(len(f.sends), in)
	}
	return len(in), nil
}

func (f *fakeAPI) Sleep(d time.Duration) {
	f.log = append(f.log, "sleep")
	f.sleeps = append(f.sleeps, d)
}

func (f *fakeAPI) clipText() string {
	if !f.hasText {
		return "<no text>"
	}
	return string(utf16.Decode(f.clip))
}

func u(s string) []uint16 { return utf16.Encode([]rune(s)) }

func TestPasteRestoresPreviousText(t *testing.T) {
	f := newFake("old")
	var seen string
	f.onSend = func(f *fakeAPI) { seen = f.clipText() }

	if err := pasteWith(f, "new text"); err != nil {
		t.Fatalf("pasteWith: %v", err)
	}
	if seen != "new text" {
		t.Errorf("clipboard at Ctrl+V = %q, want %q", seen, "new text")
	}
	if f.openAtSend {
		t.Error("clipboard was still open when Ctrl+V was sent")
	}
	if !reflect.DeepEqual(f.sends[0], ctrlVInputs()) {
		t.Errorf("sent %v, want Ctrl+V sequence", f.sends[0])
	}
	if got := f.clipText(); got != "old" {
		t.Errorf("clipboard after paste = %q, want restored %q", got, "old")
	}
	want := []string{"open", "read", "write", "close", "send", "sleep", "open", "read", "write", "close"}
	if !reflect.DeepEqual(f.log, want) {
		t.Errorf("call order = %v, want %v", f.log, want)
	}
	if !reflect.DeepEqual(f.sleeps, []time.Duration{restoreDelay}) {
		t.Errorf("sleeps = %v, want [%v]", f.sleeps, restoreDelay)
	}
	if f.opens != f.closes {
		t.Errorf("opens=%d closes=%d", f.opens, f.closes)
	}
}

func TestPasteDoesNotClobberNewerUserCopy(t *testing.T) {
	f := newFake("old")
	f.onSend = func(f *fakeAPI) { f.clip = u("user copy") }

	if err := pasteWith(f, "new"); err != nil {
		t.Fatalf("pasteWith: %v", err)
	}
	if got := f.clipText(); got != "user copy" {
		t.Errorf("clipboard = %q, want the user's copy left alone", got)
	}
	if f.log[len(f.log)-1] != "close" {
		t.Errorf("restore should only read, got log %v", f.log)
	}
	for _, op := range f.log[len(f.log)-4:] {
		if op == "write" {
			t.Errorf("restore wrote to the clipboard although it had changed: %v", f.log)
		}
	}
}

func TestPasteWithNonTextClipboardSkipsRestore(t *testing.T) {
	f := &fakeAPI{} // holds no CF_UNICODETEXT, e.g. an image
	if err := pasteWith(f, "new"); err != nil {
		t.Fatalf("pasteWith: %v", err)
	}
	if len(f.sends) != 1 || len(f.sends[0]) != 4 {
		t.Fatalf("expected a single Ctrl+V send, got %d sends", len(f.sends))
	}
	want := []string{"open", "read", "write", "close", "send", "sleep"}
	if !reflect.DeepEqual(f.log, want) {
		t.Errorf("call order = %v, want %v", f.log, want)
	}
	if got := f.clipText(); got != "new" {
		t.Errorf("clipboard = %q, want pasted text", got)
	}
}

func TestPasteEmptyIsNoop(t *testing.T) {
	f := newFake("old")
	if err := pasteWith(f, ""); err != nil {
		t.Fatalf("pasteWith(\"\"): %v", err)
	}
	if len(f.log) != 0 {
		t.Errorf("empty text touched the OS: %v", f.log)
	}
}

func TestOpenClipboardRetries(t *testing.T) {
	f := newFake("old")
	f.openFails = 3
	if err := pasteWith(f, "new"); err != nil {
		t.Fatalf("pasteWith: %v", err)
	}
	// 3 busy failures + 1 success for the paste, then 1 more open for the restore.
	if f.attempts != 3+1+1 {
		t.Errorf("open attempts = %d, want 5", f.attempts)
	}
	wantSleeps := []time.Duration{openRetryDelay, openRetryDelay, openRetryDelay, restoreDelay}
	if !reflect.DeepEqual(f.sleeps, wantSleeps) {
		t.Errorf("sleeps = %v, want %v", f.sleeps, wantSleeps)
	}
	if len(f.sends) != 1 || len(f.sends[0]) != 4 {
		t.Errorf("expected Ctrl+V, not typing, after retries succeeded")
	}
}

func TestOpenClipboardGivesUpAndTypes(t *testing.T) {
	f := newFake("old")
	f.openFails = 1000
	if err := pasteWith(f, "hé"); err != nil {
		t.Fatalf("pasteWith should fall back to typing, got %v", err)
	}
	if f.attempts != openRetries+1 {
		t.Errorf("open attempts = %d, want %d", f.attempts, openRetries+1)
	}
	if len(f.sends) != 1 || !reflect.DeepEqual(f.sends[0], unicodeInputs(u("hé"))) {
		t.Fatalf("expected the typing fallback, got sends %v", f.sends)
	}
	if got := f.clipText(); got != "old" {
		t.Errorf("clipboard changed although it never opened: %q", got)
	}
}

func TestSendFailureRestoresThenTypes(t *testing.T) {
	f := newFake("old")
	f.sendFn = func(call int, in []input) (int, error) {
		if call == 1 {
			return 0, errSend
		}
		return len(in), nil
	}
	if err := pasteWith(f, "new"); err != nil {
		t.Fatalf("pasteWith: %v", err)
	}
	if got := f.clipText(); got != "old" {
		t.Errorf("clipboard = %q, want restored %q", got, "old")
	}
	want := []string{"open", "read", "write", "close", "send", "open", "read", "write", "close", "send"}
	if !reflect.DeepEqual(f.log, want) {
		t.Errorf("call order = %v, want %v", f.log, want)
	}
	if !reflect.DeepEqual(f.sends[1], unicodeInputs(u("new"))) {
		t.Error("fallback did not type the text")
	}
}

func TestPartialSendCountFallsBack(t *testing.T) {
	f := newFake("old")
	f.sendFn = func(call int, in []input) (int, error) {
		if call == 1 {
			return 2, nil // only Ctrl and V down were accepted
		}
		return len(in), nil
	}
	if err := pasteWith(f, "new"); err != nil {
		t.Fatalf("pasteWith: %v", err)
	}
	if len(f.sends) != 2 || len(f.sends[1]) != 2*len(u("new")) {
		t.Errorf("expected typing fallback after partial send, got %d sends", len(f.sends))
	}
}

func TestWriteFailureClosesAndTypes(t *testing.T) {
	f := newFake("old")
	f.writeErr = errWrite
	if err := pasteWith(f, "new"); err != nil {
		t.Fatalf("pasteWith: %v", err)
	}
	if f.opens != f.closes {
		t.Errorf("clipboard left open: opens=%d closes=%d", f.opens, f.closes)
	}
	if len(f.sends) != 1 || !reflect.DeepEqual(f.sends[0], unicodeInputs(u("new"))) {
		t.Errorf("expected typing fallback, got sends %v", f.sends)
	}
}

func TestPasteReportsFailureOfBothPaths(t *testing.T) {
	f := newFake("old")
	f.writeErr = errWrite
	f.sendFn = func(int, []input) (int, error) { return 0, errSend }
	err := pasteWith(f, "new")
	if !errors.Is(err, errWrite) || !errors.Is(err, errSend) {
		t.Fatalf("err = %v, want both the clipboard and typing failures", err)
	}
	if !strings.Contains(err.Error(), "paste:") {
		t.Errorf("error %q does not name the failing step", err)
	}
}

func TestRestoreFailureIsReported(t *testing.T) {
	f := newFake("old")
	f.readFailAt = 2 // the restore read fails; the save read succeeds
	err := pasteWith(f, "new")
	if !errors.Is(err, errRead) {
		t.Fatalf("err = %v, want errRead", err)
	}
	if !strings.Contains(err.Error(), "restore clipboard") {
		t.Errorf("error %q does not name the restore step", err)
	}
	if len(f.sends) != 1 {
		t.Errorf("paste was retried by typing after Ctrl+V succeeded: %d sends", len(f.sends))
	}
}

func TestTypeUnicodeSendsOneEventPairPerUnit(t *testing.T) {
	f := &fakeAPI{}
	if err := typeWith(f, encodeUTF16("a😀")); err != nil {
		t.Fatalf("typeWith: %v", err)
	}
	if len(f.sends) != 1 || len(f.sends[0]) != 6 {
		t.Fatalf("want 6 events (3 units down+up), got %v", f.sends)
	}
	// U+1F600 is the surrogate pair D83D DE00.
	wantUnits := []uint16{0x61, 0xD83D, 0xDE00}
	for i, unit := range wantUnits {
		down, up := f.sends[0][2*i], f.sends[0][2*i+1]
		for _, in := range []input{down, up} {
			if in.inputType != inputKeyboard || in.ki.wVk != 0 || in.ki.wScan != unit {
				t.Errorf("event for unit %#x = %+v", unit, in.ki)
			}
			if in.ki.dwFlags&keyeventfUnicode == 0 {
				t.Errorf("unit %#x missing KEYEVENTF_UNICODE", unit)
			}
		}
		if down.ki.dwFlags&keyeventfKeyUp != 0 || up.ki.dwFlags&keyeventfKeyUp == 0 {
			t.Errorf("unit %#x: want key down then key up", unit)
		}
	}
}

func TestTypeUnicodeEmptyAndErrors(t *testing.T) {
	f := &fakeAPI{}
	if err := typeWith(f, nil); err != nil || len(f.log) != 0 {
		t.Fatalf("empty typing: err=%v log=%v", err, f.log)
	}
	f.sendFn = func(int, []input) (int, error) { return 1, nil }
	if err := typeWith(f, u("ab")); err == nil {
		t.Error("expected error when SendInput accepts fewer events")
	}
}

func TestCtrlVInputs(t *testing.T) {
	want := []struct {
		vk    uint16
		flags uint32
	}{
		{vkControl, 0},
		{vkV, 0},
		{vkV, keyeventfKeyUp},
		{vkControl, keyeventfKeyUp},
	}
	got := ctrlVInputs()
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i].inputType != inputKeyboard || got[i].ki.wVk != w.vk || got[i].ki.dwFlags != w.flags {
			t.Errorf("event %d = %+v, want vk=%#x flags=%#x", i, got[i].ki, w.vk, w.flags)
		}
	}
}

func TestEncodeUTF16(t *testing.T) {
	tests := []struct {
		in   string
		want []uint16
	}{
		{"", []uint16{}},
		{"abc", []uint16{0x61, 0x62, 0x63}},
		{"a€😀", []uint16{0x61, 0x20AC, 0xD83D, 0xDE00}},
		{"a\x00b", []uint16{0x61, 0x62}},
		{"\xff", []uint16{0xFFFD}},
	}
	for _, tt := range tests {
		got := encodeUTF16(tt.in)
		if len(got) != len(tt.want) || (len(got) > 0 && !reflect.DeepEqual(got, tt.want)) {
			t.Errorf("encodeUTF16(%q) = %x, want %x", tt.in, got, tt.want)
		}
	}
}

func TestInputLayout(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("the INPUT layout is defined for 64-bit Windows")
	}
	if got := unsafe.Sizeof(input{}); got != 40 {
		t.Errorf("sizeof(INPUT) = %d, want 40", got)
	}
	if got := unsafe.Offsetof(input{}.ki); got != 8 {
		t.Errorf("offsetof(INPUT.ki) = %d, want 8", got)
	}
	if got := unsafe.Sizeof(keybdInput{}); got != 24 {
		t.Errorf("sizeof(KEYBDINPUT) = %d, want 24", got)
	}
}

func TestUnsupportedPlatformReturnsError(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("exercises the non-Windows stub")
	}
	if err := Paste("x"); err == nil {
		t.Error("Paste on a non-Windows OS should fail")
	}
	if err := TypeUnicode("x"); err == nil {
		t.Error("TypeUnicode on a non-Windows OS should fail")
	}
	if err := Paste(""); err != nil {
		t.Errorf("Paste(\"\") = %v, want nil", err)
	}
	if err := TypeUnicode(""); err != nil {
		t.Errorf("TypeUnicode(\"\") = %v, want nil", err)
	}
}
