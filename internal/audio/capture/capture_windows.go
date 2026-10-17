package capture

import (
	"errors"
	"fmt"
	"sync"
	"syscall"
	"unsafe"
)

// This file talks to winmm.dll directly through syscall, so the build needs no cgo. Buffers are
// completed through a Win32 event instead of a waveIn callback, so no Go callback is created on the
// driver thread. A dedicated goroutine waits on the event, hands completed buffers to onPCM, and
// re-queues them.

var (
	winmm    = syscall.NewLazyDLL("winmm.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")

	procWaveInGetNumDevs      = winmm.NewProc("waveInGetNumDevs")
	procWaveInOpen            = winmm.NewProc("waveInOpen")
	procWaveInClose           = winmm.NewProc("waveInClose")
	procWaveInPrepareHeader   = winmm.NewProc("waveInPrepareHeader")
	procWaveInUnprepareHeader = winmm.NewProc("waveInUnprepareHeader")
	procWaveInAddBuffer       = winmm.NewProc("waveInAddBuffer")
	procWaveInStart           = winmm.NewProc("waveInStart")
	procWaveInReset           = winmm.NewProc("waveInReset")

	procCreateEventW        = kernel32.NewProc("CreateEventW")
	procWaitForSingleObject = kernel32.NewProc("WaitForSingleObject")
	procCloseHandle         = kernel32.NewProc("CloseHandle")
)

const (
	waveMapper    = ^uintptr(0) // WAVE_MAPPER: default recording device
	waveFormatPCM = 1
	callbackEvent = 0x00050000 // CALLBACK_EVENT
	whdrDone      = 0x00000001 // WHDR_DONE

	mmsysErrNoDriver  = 6
	mmsysErrAllocated = 4
	waverrBadFormat   = 32

	numBuffers  = 10
	bufferBytes = bufferSamples * 2 // 16-bit mono

	pollMs = 50 // how often the device goroutine checks for a stop request

	hdrSize = unsafe.Sizeof(waveHdr{})
)

// waveFormatEx mirrors WAVEFORMATEX. The 18-byte C layout is what winmm reads; Go's padding to 20
// bytes is harmless because the struct is only passed by pointer.
type waveFormatEx struct {
	FormatTag      uint16
	Channels       uint16
	SamplesPerSec  uint32
	AvgBytesPerSec uint32
	BlockAlign     uint16
	BitsPerSample  uint16
	Size           uint16
}

// waveHdr mirrors WAVEHDR. Pointers are stored as uintptr so the layout matches on 32- and 64-bit
// builds. The data they point to is kept alive by winDevice.data.
type waveHdr struct {
	Data          uintptr
	BufferLength  uint32
	BytesRecorded uint32
	User          uintptr
	Flags         uint32
	Loops         uint32
	Next          uintptr
	Reserved      uintptr
}

type winDevice struct {
	hwi   uintptr // HWAVEIN
	event uintptr // HANDLE, auto-reset, signalled by winmm when a buffer completes
	data  []byte
	hdrs  []waveHdr

	prepared int // headers prepared so far; release unprepares only these
	onPCM    func([]byte)

	stop     chan struct{}
	done     chan struct{}
	stopOnce sync.Once
	closeErr error
}

func openDevice(onPCM func([]byte)) (device, error) {
	if n, _, _ := procWaveInGetNumDevs.Call(); uint32(n) == 0 {
		return nil, errors.New("capture: no recording device found")
	}
	ev, _, err := procCreateEventW.Call(0, 0, 0, 0)
	if ev == 0 {
		return nil, fmt.Errorf("capture: CreateEventW: %v", err)
	}

	format := waveFormatEx{
		FormatTag:      waveFormatPCM,
		Channels:       1,
		SamplesPerSec:  sampleRate,
		AvgBytesPerSec: sampleRate * 2,
		BlockAlign:     2,
		BitsPerSample:  16,
		Size:           0,
	}
	d := &winDevice{
		event: ev,
		data:  make([]byte, numBuffers*bufferBytes),
		hdrs:  make([]waveHdr, numBuffers),
		onPCM: onPCM,
		stop:  make(chan struct{}),
		done:  make(chan struct{}),
	}

	// MME converts from the device's native format to the requested 16 kHz mono 16-bit PCM.
	r, _, _ := procWaveInOpen.Call(
		uintptr(unsafe.Pointer(&d.hwi)),
		waveMapper,
		uintptr(unsafe.Pointer(&format)),
		ev,
		0,
		callbackEvent,
	)
	if err := mmError("waveInOpen", r); err != nil {
		closeHandle(ev)
		return nil, err
	}

	fail := func(err error) (device, error) {
		d.release()
		return nil, err
	}

	for i := range d.hdrs {
		d.hdrs[i].Data = uintptr(unsafe.Pointer(&d.data[i*bufferBytes]))
		d.hdrs[i].BufferLength = bufferBytes
		r, _, _ := procWaveInPrepareHeader.Call(d.hwi, d.hdrPtr(i), hdrSize)
		if err := mmError("waveInPrepareHeader", r); err != nil {
			return fail(err)
		}
		d.prepared++
	}
	for i := range d.hdrs {
		r, _, _ := procWaveInAddBuffer.Call(d.hwi, d.hdrPtr(i), hdrSize)
		if err := mmError("waveInAddBuffer", r); err != nil {
			return fail(err)
		}
	}
	r, _, _ = procWaveInStart.Call(d.hwi)
	if err := mmError("waveInStart", r); err != nil {
		return fail(err)
	}

	go d.run()
	return d, nil
}

func (d *winDevice) hdrPtr(i int) uintptr {
	return uintptr(unsafe.Pointer(&d.hdrs[i]))
}

// run is the only goroutine that touches the queue after Open. Buffers complete in queue order, so
// the cursor walks the ring and never skips one.
func (d *winDevice) run() {
	defer close(d.done)
	cursor := 0
	for {
		select {
		case <-d.stop:
			d.closeErr = d.release()
			return
		default:
		}
		procWaitForSingleObject.Call(d.event, pollMs)

		for d.hdrs[cursor].Flags&whdrDone != 0 {
			h := &d.hdrs[cursor]
			if n := int(h.BytesRecorded); n > 0 {
				off := cursor * bufferBytes
				d.onPCM(d.data[off : off+n])
			}
			// Keep WHDR_PREPARED: winmm refuses to queue an unprepared header.
			h.Flags &^= whdrDone
			h.BytesRecorded = 0
			r, _, _ := procWaveInAddBuffer.Call(d.hwi, d.hdrPtr(cursor), hdrSize)
			if err := mmError("waveInAddBuffer", r); err != nil {
				// The device is gone or failed. Stop feeding; Close reports the error.
				d.closeErr = err
				d.release()
				return
			}
			cursor = (cursor + 1) % numBuffers
		}
	}
}

// release stops the device and frees it. It runs on the device goroutine, or from openDevice
// before that goroutine starts. waveInReset returns every queued buffer synchronously, so the
// headers can be unprepared and the device closed.
func (d *winDevice) release() error {
	var first error
	if d.hwi != 0 {
		procWaveInReset.Call(d.hwi)
		for i := 0; i < d.prepared; i++ {
			procWaveInUnprepareHeader.Call(d.hwi, d.hdrPtr(i), hdrSize)
		}
		d.prepared = 0
		r, _, _ := procWaveInClose.Call(d.hwi)
		first = mmError("waveInClose", r)
		d.hwi = 0
	}
	closeHandle(d.event)
	d.event = 0
	return first
}

func (d *winDevice) close() error {
	d.stopOnce.Do(func() { close(d.stop) })
	<-d.done
	return d.closeErr
}

func closeHandle(h uintptr) {
	if h != 0 {
		procCloseHandle.Call(h)
	}
}

// mmError converts an MMRESULT (a UINT returned in the low bits of r1) into an error.
func mmError(op string, r uintptr) error {
	code := uint32(r)
	switch code {
	case 0:
		return nil
	case mmsysErrNoDriver:
		return fmt.Errorf("capture: %s: no recording driver (MMSYSERR_NODRIVER)", op)
	case mmsysErrAllocated:
		return fmt.Errorf("capture: %s: the microphone is in use by another application", op)
	case waverrBadFormat:
		return fmt.Errorf("capture: %s: the device rejected 16 kHz mono 16-bit PCM (WAVERR_BADFORMAT)", op)
	}
	return fmt.Errorf("capture: %s failed with MMRESULT %d", op, code)
}
