package hotkey

import (
	"context"
	"fmt"
	"io"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

const (
	whKeyboardLL  = 13
	wmQuit        = 0x0012
	wmUser        = 0x0400
	pmNoRemove    = 0x0000
	llkhfInjected = 0x10
	llkhfUp       = 0x80
)

var (
	user32                  = syscall.NewLazyDLL("user32.dll")
	kernel32                = syscall.NewLazyDLL("kernel32.dll")
	procSetWindowsHookExW   = user32.NewProc("SetWindowsHookExW")
	procUnhookWindowsHookEx = user32.NewProc("UnhookWindowsHookEx")
	procCallNextHookEx      = user32.NewProc("CallNextHookEx")
	procGetMessageW         = user32.NewProc("GetMessageW")
	procPeekMessageW        = user32.NewProc("PeekMessageW")
	procPostThreadMessageW  = user32.NewProc("PostThreadMessageW")
	procGetCurrentThreadID  = kernel32.NewProc("GetCurrentThreadId")
	procGetModuleHandleW    = kernel32.NewProc("GetModuleHandleW")
)

// kbdllHookStruct mirrors KBDLLHOOKSTRUCT.
type kbdllHookStruct struct {
	VKCode    uint32
	ScanCode  uint32
	Flags     uint32
	Time      uint32
	ExtraInfo uintptr
}

// msg mirrors MSG (48 bytes on amd64).
type msg struct {
	HWnd    uintptr
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	PtX     int32
	PtY     int32
}

type kbHook struct {
	handle uintptr
	sink   func(rawKey) bool
}

// hooks maps the hook thread's ID to its hook. The callback runs on that thread, so it finds its hook by thread ID.
var hooks sync.Map

// hookCallback is created once: syscall.NewCallback slots are never freed.
var hookCallback = syscall.NewCallback(lowLevelKeyboardProc)

// kbdFromLParam reinterprets the lParam the system passed to the hook as a KBDLLHOOKSTRUCT pointer. It goes through
// the address of the parameter so vet does not flag a direct uintptr-to-unsafe.Pointer conversion. The struct lives
// in system-owned memory for the duration of the callback only.
func kbdFromLParam(lParam uintptr) *kbdllHookStruct {
	return (*kbdllHookStruct)(*(*unsafe.Pointer)(unsafe.Pointer(&lParam)))
}

// lowLevelKeyboardProc is the WH_KEYBOARD_LL hook procedure. It runs on the hook thread and must not block.
func lowLevelKeyboardProc(nCode, wParam, lParam uintptr) uintptr {
	if int32(nCode) == 0 && lParam != 0 {
		if v, ok := hooks.Load(currentThreadID()); ok {
			h := v.(*kbHook)
			kb := kbdFromLParam(lParam)
			k := rawKey{
				VK:       kb.VKCode,
				Scan:     kb.ScanCode,
				Flags:    kb.Flags,
				Down:     kb.Flags&llkhfUp == 0,
				Injected: kb.Flags&llkhfInjected != 0,
			}
			if h.sink(k) {
				return 1 // swallow: the focused application never sees this transition
			}
		}
	}
	r, _, _ := procCallNextHookEx.Call(0, nCode, wParam, lParam)
	return r
}

func currentThreadID() uint32 {
	r, _, _ := procGetCurrentThreadID.Call()
	return uint32(r)
}

// installHook installs a WH_KEYBOARD_LL hook on a new locked OS thread that pumps messages. sink is called on that
// thread for every key transition; it returns true to swallow the transition. The returned stop removes the hook,
// ends the thread and waits for it to exit. stop is safe to call more than once.
func installHook(sink func(rawKey) bool) (func(), error) {
	type ready struct {
		tid uint32
		err error
	}
	readyCh := make(chan ready, 1)
	done := make(chan struct{})

	go func() {
		defer close(done)
		// Deliberately not unlocked: the thread ends with the goroutine, which removes the hook thread cleanly.
		runtime.LockOSThread()

		// Heap-allocated so the message buffer cannot move if the goroutine stack grows inside a callback.
		m := new(msg)
		// Creating the thread's message queue now lets PostThreadMessage reach it later.
		procPeekMessageW.Call(uintptr(unsafe.Pointer(m)), 0, wmUser, wmUser, pmNoRemove)

		tid := currentThreadID()
		hmod, _, _ := procGetModuleHandleW.Call(0)
		h, _, err := procSetWindowsHookExW.Call(whKeyboardLL, hookCallback, hmod, 0)
		if h == 0 {
			readyCh <- ready{err: fmt.Errorf("hotkey: SetWindowsHookExW failed: %v", err)}
			return
		}
		hooks.Store(tid, &kbHook{handle: h, sink: sink})
		readyCh <- ready{tid: tid}

		for {
			r, _, _ := procGetMessageW.Call(uintptr(unsafe.Pointer(m)), 0, 0, 0)
			if int32(r) <= 0 { // 0: WM_QUIT from stop; -1: error. Either way, stop pumping.
				break
			}
		}
		hooks.Delete(tid)
		procUnhookWindowsHookEx.Call(h)
	}()

	r := <-readyCh
	if r.err != nil {
		<-done
		return nil, r.err
	}
	var once sync.Once
	stop := func() {
		once.Do(func() {
			// If the thread already exited, PostThreadMessage fails and done is already closed.
			procPostThreadMessageW.Call(uintptr(r.tid), wmQuit, 0, 0)
			<-done
		})
	}
	return stop, nil
}

// Probe installs a non-swallowing hook and writes one line per key transition to w until ctx is done. It is a
// diagnostic for finding the vk code of a key (for example whether the Fn key is visible at all).
func Probe(ctx context.Context, w io.Writer) error {
	ch := make(chan rawKey, 64)
	stop, err := installHook(func(k rawKey) bool {
		select {
		case ch <- k:
		default:
		}
		return false
	})
	if err != nil {
		return err
	}
	defer stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case k := <-ch:
			if _, err := fmt.Fprintln(w, formatKey(k)); err != nil {
				return err
			}
		}
	}
}
