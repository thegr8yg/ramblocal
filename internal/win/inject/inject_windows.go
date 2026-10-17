//go:build windows

package inject

import (
	"fmt"
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

var (
	user32   = syscall.NewLazyDLL("user32.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")

	procOpenClipboard              = user32.NewProc("OpenClipboard")
	procCloseClipboard             = user32.NewProc("CloseClipboard")
	procEmptyClipboard             = user32.NewProc("EmptyClipboard")
	procIsClipboardFormatAvailable = user32.NewProc("IsClipboardFormatAvailable")
	procGetClipboardData           = user32.NewProc("GetClipboardData")
	procSetClipboardData           = user32.NewProc("SetClipboardData")
	procSendInput                  = user32.NewProc("SendInput")

	procGlobalAlloc  = kernel32.NewProc("GlobalAlloc")
	procGlobalFree   = kernel32.NewProc("GlobalFree")
	procGlobalLock   = kernel32.NewProc("GlobalLock")
	procGlobalUnlock = kernel32.NewProc("GlobalUnlock")
	procGlobalSize   = kernel32.NewProc("GlobalSize")
)

const gmemMoveable = 0x0002 // GMEM_MOVEABLE

// platform returns the real Win32 implementation. The input struct layout is only correct
// for 64-bit Windows, so 32-bit builds refuse to run.
func platform() (osAPI, error) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		return nil, fmt.Errorf("inject: 64-bit Windows build required")
	}
	return winAPI{}, nil
}

type winAPI struct{}

func (winAPI) OpenClipboard() error {
	r, _, err := procOpenClipboard.Call(0)
	if r == 0 {
		return callErr("OpenClipboard", err)
	}
	return nil
}

func (winAPI) CloseClipboard() error {
	r, _, err := procCloseClipboard.Call()
	if r == 0 {
		return callErr("CloseClipboard", err)
	}
	return nil
}

func (winAPI) ReadText() ([]uint16, bool, error) {
	if r, _, _ := procIsClipboardFormatAvailable.Call(cfUnicodeText); r == 0 {
		return nil, false, nil
	}
	h, _, err := procGetClipboardData.Call(cfUnicodeText)
	if h == 0 {
		return nil, false, callErr("GetClipboardData", err)
	}
	// The clipboard owns the handle: read it, never free it.
	size, _, _ := procGlobalSize.Call(h)
	p, _, err := procGlobalLock.Call(h)
	if p == 0 {
		return nil, false, callErr("GlobalLock", err)
	}
	defer procGlobalUnlock.Call(h)

	src := unsafe.Slice((*uint16)(addrToPtr(p)), size/2)
	n := 0
	for n < len(src) && src[n] != 0 {
		n++
	}
	return append([]uint16(nil), src[:n]...), true, nil
}

func (winAPI) WriteText(units []uint16) error {
	if r, _, err := procEmptyClipboard.Call(); r == 0 {
		return callErr("EmptyClipboard", err)
	}
	h, _, err := procGlobalAlloc.Call(gmemMoveable, uintptr(len(units)+1)*2)
	if h == 0 {
		return callErr("GlobalAlloc", err)
	}
	p, _, err := procGlobalLock.Call(h)
	if p == 0 {
		procGlobalFree.Call(h)
		return callErr("GlobalLock", err)
	}
	dst := unsafe.Slice((*uint16)(addrToPtr(p)), len(units)+1)
	copy(dst, units)
	dst[len(units)] = 0
	procGlobalUnlock.Call(h)

	if r, _, err := procSetClipboardData.Call(cfUnicodeText, h); r == 0 {
		// The clipboard did not take ownership, so the block is still ours to free.
		procGlobalFree.Call(h)
		return callErr("SetClipboardData", err)
	}
	return nil
}

func (winAPI) SendInput(in []input) (int, error) {
	if len(in) == 0 {
		return 0, nil
	}
	r, _, err := procSendInput.Call(
		uintptr(len(in)),
		uintptr(unsafe.Pointer(&in[0])),
		unsafe.Sizeof(in[0]),
	)
	runtime.KeepAlive(in)
	if r == 0 {
		return 0, callErr("SendInput", err)
	}
	return int(r), nil
}

func (winAPI) Sleep(d time.Duration) { time.Sleep(d) }

func callErr(name string, err error) error {
	if err == nil {
		return fmt.Errorf("%s failed", name)
	}
	return fmt.Errorf("%s: %w", name, err)
}

// addrToPtr turns an address returned by GlobalLock into a pointer. The memory belongs to
// the OS, not the Go heap, so the GC neither tracks nor moves it. Reinterpreting the uintptr
// keeps go vet's unsafeptr check from flagging the conversion.
func addrToPtr(addr uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&addr))
}
