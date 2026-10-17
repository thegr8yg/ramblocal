//go:build windows

package overlay

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

var (
	user32   = syscall.NewLazyDLL("user32.dll")
	gdi32    = syscall.NewLazyDLL("gdi32.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")

	pRegisterClassExW      = user32.NewProc("RegisterClassExW")
	pCreateWindowExW       = user32.NewProc("CreateWindowExW")
	pDestroyWindow         = user32.NewProc("DestroyWindow")
	pDefWindowProcW        = user32.NewProc("DefWindowProcW")
	pGetMessageW           = user32.NewProc("GetMessageW")
	pTranslateMessage      = user32.NewProc("TranslateMessage")
	pDispatchMessageW      = user32.NewProc("DispatchMessageW")
	pPostMessageW          = user32.NewProc("PostMessageW")
	pPostQuitMessage       = user32.NewProc("PostQuitMessage")
	pShowWindow            = user32.NewProc("ShowWindow")
	pUpdateLayeredWindow   = user32.NewProc("UpdateLayeredWindow")
	pGetDC                 = user32.NewProc("GetDC")
	pReleaseDC             = user32.NewProc("ReleaseDC")
	pSetCapture            = user32.NewProc("SetCapture")
	pReleaseCapture        = user32.NewProc("ReleaseCapture")
	pSetTimer              = user32.NewProc("SetTimer")
	pKillTimer             = user32.NewProc("KillTimer")
	pCreatePopupMenu       = user32.NewProc("CreatePopupMenu")
	pAppendMenuW           = user32.NewProc("AppendMenuW")
	pTrackPopupMenu        = user32.NewProc("TrackPopupMenu")
	pDestroyMenu           = user32.NewProc("DestroyMenu")
	pGetCursorPos          = user32.NewProc("GetCursorPos")
	pSetForegroundWindow   = user32.NewProc("SetForegroundWindow")
	pSystemParametersInfoW = user32.NewProc("SystemParametersInfoW")
	pLoadCursorW           = user32.NewProc("LoadCursorW")

	pCreateCompatibleDC = gdi32.NewProc("CreateCompatibleDC")
	pDeleteDC           = gdi32.NewProc("DeleteDC")
	pCreateDIBSection   = gdi32.NewProc("CreateDIBSection")
	pSelectObject       = gdi32.NewProc("SelectObject")
	pDeleteObject       = gdi32.NewProc("DeleteObject")

	pGetModuleHandleW = kernel32.NewProc("GetModuleHandleW")
)

const (
	wsPopup      = 0x80000000
	wsExTopmost  = 0x00000008
	wsExToolWin  = 0x00000080
	wsExNoActiv  = 0x08000000
	wsExLayered  = 0x00080000
	swShowNoAct  = 4
	maNoActivate = 3
	ulwAlpha     = 0x2
	acSrcOver    = 0x00
	acSrcAlpha   = 0x01
	dibRGBColors = 0
	biRGB        = 0
	mfString     = 0x0
	tpmRightBtn  = 0x0002
	tpmReturnCmd = 0x0100
	spiGetWork   = 0x0030
	idcArrow     = 32512

	wmDestroy       = 0x0002
	wmClose         = 0x0010
	wmTimer         = 0x0113
	wmMouseMove     = 0x0200
	wmLButtonDown   = 0x0201
	wmLButtonUp     = 0x0202
	wmRButtonUp     = 0x0205
	wmMouseActivate = 0x0021
	wmCaptureChange = 0x0215
	wmNull          = 0x0000
	wmWake          = 0x8000 + 1 // WM_APP+1: state or level changed, repaint

	timerAnim    = 1
	timerFail    = 2
	animPeriodMs = 30
	failHoldMs   = 1500
	idQuit       = 1

	transitionDur = 120 * time.Millisecond
	bottomMargin  = 24 // px between the pill and the bottom of the work area
	className     = "ramblocal.overlay"
)

var (
	classNameW, _ = syscall.UTF16PtrFromString(className)
	titleW, _     = syscall.UTF16PtrFromString("ramblocal")
	menuLabelW, _ = syscall.UTF16PtrFromString("Quit ramblocal")

	regOnce sync.Once
	regErr  error

	// overlays maps HWND (uintptr) to *Overlay for the window procedure.
	overlays sync.Map
)

type point struct{ X, Y int32 }
type rect struct{ Left, Top, Right, Bottom int32 }
type size struct{ CX, CY int32 }

type msg struct {
	HWnd    uintptr
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	Pt      point
}

type wndClassEx struct {
	Size       uint32
	Style      uint32
	WndProc    uintptr
	ClsExtra   int32
	WndExtra   int32
	Instance   uintptr
	Icon       uintptr
	Cursor     uintptr
	Background uintptr
	MenuName   *uint16
	ClassName  *uint16
	IconSm     uintptr
}

type bitmapInfoHeader struct {
	Size          uint32
	Width         int32
	Height        int32
	Planes        uint16
	BitCount      uint16
	Compression   uint32
	SizeImage     uint32
	XPelsPerMeter int32
	YPelsPerMeter int32
	ClrUsed       uint32
	ClrImportant  uint32
}

type bitmapInfo struct{ Header bitmapInfoHeader }

type blendFunction struct {
	BlendOp             uint8
	BlendFlags          uint8
	SourceConstantAlpha uint8
	AlphaFormat         uint8
}

// call invokes a user32/gdi32 procedure and returns its first result.
func call(p *syscall.LazyProc, args ...uintptr) uintptr {
	r, _, _ := p.Call(args...)
	return r
}

// callErr is call that also reports the Win32 error when the result is zero.
func callErr(p *syscall.LazyProc, args ...uintptr) (uintptr, error) {
	r, _, err := p.Call(args...)
	if r == 0 {
		return 0, err
	}
	return r, nil
}

// platform is the window-thread state. Only hwnd is read from other goroutines (atomically).
type platform struct {
	hwnd atomic.Uintptr
	done chan struct{} // closed when the window thread has finished

	// Everything below is touched only on the window thread.
	epoch time.Time

	screenDC uintptr
	memDC    uintptr
	bmp      uintptr
	stock    uintptr // the bitmap that was in memDC before ours; restored before deletion
	bits     []uint32
	bw, bh   int

	cur State
	tr  transition

	animOn bool
	failOn bool

	anchorX, anchorBottom int // centre-x and bottom edge of the pill; kept fixed while it resizes
	dragging              bool
	dragFrom              point
	dragAnchorX           int
	dragAnchorBottom      int
}

// transition animates the pill from one size to another.
type transition struct {
	fromW, fromH int
	toW, toH     int
	start        time.Time
	active       bool
}

// start creates the window thread and waits until the window exists.
func (o *Overlay) start() error {
	o.pl.done = make(chan struct{})
	ready := make(chan error, 1)
	go o.run(ready)
	return <-ready
}

// run is the window thread. It never unlocks the OS thread: the thread ends with the goroutine.
func (o *Overlay) run(ready chan<- error) {
	runtime.LockOSThread()
	defer close(o.pl.done)

	if err := o.createWindow(); err != nil {
		ready <- err
		return
	}
	o.paint(time.Now())
	call(pShowWindow, o.pl.hwnd.Load(), swShowNoAct)
	ready <- nil

	var m msg
	for {
		r := call(pGetMessageW, uintptr(unsafe.Pointer(&m)), 0, 0, 0)
		if int32(r) <= 0 { // 0 = WM_QUIT, -1 = error
			break
		}
		call(pTranslateMessage, uintptr(unsafe.Pointer(&m)))
		call(pDispatchMessageW, uintptr(unsafe.Pointer(&m)))
	}
	o.pl.releaseGDI()
}

func registerClass() {
	inst := call(pGetModuleHandleW, 0)
	wc := wndClassEx{
		Size:      uint32(unsafe.Sizeof(wndClassEx{})),
		WndProc:   syscall.NewCallback(wndProc),
		Instance:  inst,
		Cursor:    call(pLoadCursorW, 0, idcArrow),
		ClassName: classNameW,
	}
	if _, err := callErr(pRegisterClassExW, uintptr(unsafe.Pointer(&wc))); err != nil {
		regErr = fmt.Errorf("overlay: RegisterClassExW: %w", err)
	}
}

func (o *Overlay) createWindow() error {
	regOnce.Do(registerClass)
	if regErr != nil {
		return regErr
	}
	p := &o.pl
	p.epoch = time.Now()
	p.cur = Idle
	p.anchorX, p.anchorBottom = defaultAnchor()
	w, h := restSize(Idle)
	p.tr = transition{fromW: w, fromH: h, toW: w, toH: h}

	hwnd, err := callErr(pCreateWindowExW,
		wsExTopmost|wsExToolWin|wsExNoActiv|wsExLayered,
		uintptr(unsafe.Pointer(classNameW)),
		uintptr(unsafe.Pointer(titleW)),
		wsPopup,
		uintptr(int32(p.anchorX-w/2)), uintptr(int32(p.anchorBottom-h)),
		uintptr(w), uintptr(h),
		0, 0, call(pGetModuleHandleW, 0), 0)
	if err != nil {
		return fmt.Errorf("overlay: CreateWindowExW: %w", err)
	}
	p.hwnd.Store(hwnd)
	overlays.Store(hwnd, o)

	p.screenDC = call(pGetDC, 0)
	p.memDC = call(pCreateCompatibleDC, p.screenDC)
	if p.screenDC == 0 || p.memDC == 0 {
		return errors.New("overlay: could not create device contexts")
	}
	return nil
}

// defaultAnchor is the bottom-centre of the primary monitor's work area (above the taskbar).
func defaultAnchor() (x, bottom int) {
	var rc rect
	if _, err := callErr(pSystemParametersInfoW, spiGetWork, 0, uintptr(unsafe.Pointer(&rc)), 0); err != nil || rc.Right <= rc.Left {
		rc = rect{0, 0, 1920, 1080}
	}
	return int(rc.Left+rc.Right) / 2, int(rc.Bottom) - bottomMargin
}

// destroy asks the window thread to tear down the window and waits for it to finish.
func (o *Overlay) destroy() {
	hwnd := o.pl.hwnd.Load()
	if hwnd == 0 {
		return
	}
	call(pPostMessageW, hwnd, wmClose, 0, 0)
	<-o.pl.done
}

// wake asks the window thread to re-read state and repaint. Safe from any goroutine.
func (o *Overlay) wake() {
	if hwnd := o.pl.hwnd.Load(); hwnd != 0 {
		call(pPostMessageW, hwnd, wmWake, 0, 0)
	}
}

// wndProc runs on the window thread.
func wndProc(hwnd, m, wParam, lParam uintptr) uintptr {
	v, ok := overlays.Load(hwnd)
	if !ok {
		return call(pDefWindowProcW, hwnd, m, wParam, lParam)
	}
	o := v.(*Overlay)
	switch m {
	case wmWake:
		o.sync(time.Now())
		return 0
	case wmTimer:
		now := time.Now()
		switch wParam {
		case timerAnim:
			o.paint(now)
			o.updateAnimTimer()
		case timerFail:
			call(pKillTimer, hwnd, timerFail)
			o.pl.failOn = false
			o.state.CompareAndSwap(int32(Failed), int32(Idle))
			o.sync(now)
		}
		return 0
	case wmMouseActivate:
		return maNoActivate
	case wmLButtonDown:
		o.beginDrag()
		return 0
	case wmMouseMove:
		o.drag(time.Now())
		return 0
	case wmLButtonUp:
		o.endDrag()
		return 0
	case wmCaptureChange:
		o.pl.dragging = false
		return 0
	case wmRButtonUp:
		o.showMenu(hwnd)
		return 0
	case wmClose:
		call(pDestroyWindow, hwnd)
		return 0
	case wmDestroy:
		overlays.Delete(hwnd)
		call(pPostQuitMessage, 0)
		return 0
	}
	return call(pDefWindowProcW, hwnd, m, wParam, lParam)
}

// sync applies the current state: starts a size transition when it changed, manages timers, repaints.
func (o *Overlay) sync(now time.Time) {
	p := &o.pl
	want := State(o.state.Load())
	if want != p.cur {
		p.cur = want
		w, h := restSize(want)
		fw, fh := p.frameSize(now)
		p.tr = transition{fromW: fw, fromH: fh, toW: w, toH: h, start: now, active: fw != w || fh != h}
	}
	hwnd := p.hwnd.Load()
	if want == Failed {
		// (Re)start the hold timer: a repeated Failed extends the flash.
		call(pSetTimer, hwnd, timerFail, failHoldMs, 0)
		p.failOn = true
	} else if p.failOn {
		call(pKillTimer, hwnd, timerFail)
		p.failOn = false
	}
	o.paint(now)
	o.updateAnimTimer()
}

func (o *Overlay) updateAnimTimer() {
	p := &o.pl
	hwnd := p.hwnd.Load()
	need := p.cur == Listening || p.cur == Working || p.tr.active
	switch {
	case need && !p.animOn:
		call(pSetTimer, hwnd, timerAnim, animPeriodMs, 0)
		p.animOn = true
	case !need && p.animOn:
		call(pKillTimer, hwnd, timerAnim)
		p.animOn = false
	}
}

// frameSize is the size to draw now, interpolating during a transition (ease-out cubic).
func (p *platform) frameSize(now time.Time) (int, int) {
	tr := &p.tr
	if !tr.active {
		return tr.toW, tr.toH
	}
	f := float64(now.Sub(tr.start)) / float64(transitionDur)
	if f >= 1 {
		tr.active = false
		return tr.toW, tr.toH
	}
	e := 1 - math.Pow(1-f, 3)
	lerp := func(a, b int) int {
		return max(1, int(math.Round(float64(a)+float64(b-a)*e)))
	}
	return lerp(tr.fromW, tr.toW), lerp(tr.fromH, tr.toH)
}

// paint renders the current frame and pushes it to the layered window.
func (o *Overlay) paint(now time.Time) {
	p := &o.pl
	hwnd := p.hwnd.Load()
	w, h := p.frameSize(now)
	if w != p.bw || h != p.bh {
		if err := p.resizeBitmap(w, h); err != nil {
			return
		}
	}
	level := math.Float32frombits(o.level.Load())
	px := render(p.cur, level, now.Sub(p.epoch).Seconds(), w, h)
	copy(p.bits, px)

	pos := point{X: int32(p.anchorX - w/2), Y: int32(p.anchorBottom - h)}
	sz := size{CX: int32(w), CY: int32(h)}
	src := point{}
	blend := blendFunction{BlendOp: acSrcOver, SourceConstantAlpha: 255, AlphaFormat: acSrcAlpha}
	call(pUpdateLayeredWindow, hwnd, p.screenDC,
		uintptr(unsafe.Pointer(&pos)), uintptr(unsafe.Pointer(&sz)),
		p.memDC, uintptr(unsafe.Pointer(&src)), 0,
		uintptr(unsafe.Pointer(&blend)), ulwAlpha)
}

// resizeBitmap replaces the 32-bit premultiplied DIB backing the window.
func (p *platform) resizeBitmap(w, h int) error {
	bi := bitmapInfo{Header: bitmapInfoHeader{
		Size:        uint32(unsafe.Sizeof(bitmapInfoHeader{})),
		Width:       int32(w),
		Height:      -int32(h), // top-down
		Planes:      1,
		BitCount:    32,
		Compression: biRGB,
	}}
	var bitsPtr unsafe.Pointer
	bmp, err := callErr(pCreateDIBSection, p.memDC, uintptr(unsafe.Pointer(&bi)), dibRGBColors,
		uintptr(unsafe.Pointer(&bitsPtr)), 0, 0)
	if err != nil {
		return fmt.Errorf("overlay: CreateDIBSection: %w", err)
	}
	old := call(pSelectObject, p.memDC, bmp)
	if p.stock == 0 {
		p.stock = old
	}
	if p.bmp != 0 {
		call(pDeleteObject, p.bmp)
	}
	p.bmp = bmp
	p.bits = unsafe.Slice((*uint32)(bitsPtr), w*h)
	p.bw, p.bh = w, h
	return nil
}

// releaseGDI frees the device contexts and bitmap. Window thread only, after the window is gone.
func (p *platform) releaseGDI() {
	if p.bmp != 0 {
		call(pSelectObject, p.memDC, p.stock)
		call(pDeleteObject, p.bmp)
		p.bmp = 0
	}
	if p.memDC != 0 {
		call(pDeleteDC, p.memDC)
		p.memDC = 0
	}
	if p.screenDC != 0 {
		call(pReleaseDC, 0, p.screenDC)
		p.screenDC = 0
	}
}

func cursorPos() point {
	var pt point
	call(pGetCursorPos, uintptr(unsafe.Pointer(&pt)))
	return pt
}

// Left-drag moves the pill. Manual mouse capture keeps the drag going when the pointer leaves the window.
func (o *Overlay) beginDrag() {
	p := &o.pl
	p.dragging = true
	p.dragFrom = cursorPos()
	p.dragAnchorX, p.dragAnchorBottom = p.anchorX, p.anchorBottom
	call(pSetCapture, p.hwnd.Load())
}

func (o *Overlay) drag(now time.Time) {
	p := &o.pl
	if !p.dragging {
		return
	}
	pt := cursorPos()
	p.anchorX = p.dragAnchorX + int(pt.X-p.dragFrom.X)
	p.anchorBottom = p.dragAnchorBottom + int(pt.Y-p.dragFrom.Y)
	o.paint(now)
}

func (o *Overlay) endDrag() {
	if !o.pl.dragging {
		return
	}
	o.pl.dragging = false
	call(pReleaseCapture)
}

// showMenu opens the right-click menu. Choosing Quit closes the Quit channel.
func (o *Overlay) showMenu(hwnd uintptr) {
	menu := call(pCreatePopupMenu)
	if menu == 0 {
		return
	}
	defer call(pDestroyMenu, menu)
	call(pAppendMenuW, menu, mfString, idQuit, uintptr(unsafe.Pointer(menuLabelW)))
	pt := cursorPos()
	call(pSetForegroundWindow, hwnd) // needed so the menu dismisses when focus moves elsewhere
	cmd := call(pTrackPopupMenu, menu, tpmRightBtn|tpmReturnCmd,
		uintptr(int32(pt.X)), uintptr(int32(pt.Y)), 0, hwnd, 0)
	call(pPostMessageW, hwnd, wmNull, 0, 0)
	if cmd == idQuit {
		o.signalQuit()
	}
}
