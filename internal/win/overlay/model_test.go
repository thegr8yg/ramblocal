package overlay

import "testing"

func sumAlpha(px []uint32) (total uint64) {
	for _, p := range px {
		total += uint64(p >> 24)
	}
	return total
}

func TestRenderDimensions(t *testing.T) {
	for _, s := range []State{Idle, Listening, Working, Failed} {
		w, h := restSize(s)
		px := render(s, 0.3, 0.5, w, h)
		if len(px) != w*h {
			t.Errorf("%v: len=%d, want %d", s, len(px), w*h)
		}
		if sumAlpha(px) == 0 {
			t.Errorf("%v: rendered nothing", s)
		}
	}
	if got := render(Idle, 0, 0, 0, 10); got != nil {
		t.Errorf("zero width: got %d pixels, want nil", len(got))
	}
	if got := render(Idle, 0, 0, 10, -1); got != nil {
		t.Errorf("negative height: got %d pixels, want nil", len(got))
	}
}

func TestRenderIsPremultiplied(t *testing.T) {
	for _, s := range []State{Idle, Listening, Working, Failed} {
		w, h := restSize(s)
		for i, p := range render(s, 1, 0.7, w, h) {
			a, r, g, b := p>>24, (p>>16)&0xff, (p>>8)&0xff, p&0xff
			if r > a || g > a || b > a {
				t.Fatalf("%v pixel %d not premultiplied: a=%d r=%d g=%d b=%d", s, i, a, r, g, b)
			}
		}
	}
}

func TestRenderCornersAreTransparent(t *testing.T) {
	w, h := restSize(Listening)
	px := render(Listening, 0.5, 0, w, h)
	if px[0]>>24 != 0 || px[w-1]>>24 != 0 || px[(h-1)*w]>>24 != 0 {
		t.Errorf("pill corners should be outside the rounded rect")
	}
}

func TestLevelChangesPixels(t *testing.T) {
	w, h := restSize(Listening)
	quiet := render(Listening, 0, 0.4, w, h)
	loud := render(Listening, 1, 0.4, w, h)
	if equal(quiet, loud) {
		t.Fatal("level 0 and level 1 rendered identical frames")
	}
}

func TestIdleSmallerAndDimmerThanListening(t *testing.T) {
	iw, ih := restSize(Idle)
	lw, lh := restSize(Listening)
	if iw*ih >= lw*lh {
		t.Fatalf("idle %dx%d is not smaller than listening %dx%d", iw, ih, lw, lh)
	}
	idle := sumAlpha(render(Idle, 0, 0, iw, ih))
	listen := sumAlpha(render(Listening, 0.5, 0, lw, lh))
	if idle >= listen {
		t.Errorf("idle alpha sum %d is not below listening %d", idle, listen)
	}
}

func TestWorkingPulseMoves(t *testing.T) {
	w, h := restSize(Working)
	if equal(render(Working, 0, 0, w, h), render(Working, 0, 0.3, w, h)) {
		t.Fatal("working frames at different times are identical")
	}
}

func TestFailedIsRed(t *testing.T) {
	w, h := restSize(Failed)
	px := render(Failed, 0, 0, w, h)
	p := px[(h/2)*w+w/2-10] // inside the pill, clear of the centre dot
	a, r, g, b := p>>24, (p>>16)&0xff, (p>>8)&0xff, p&0xff
	if a == 0 || r <= g || r <= b {
		t.Errorf("failed pill is not red: a=%d r=%d g=%d b=%d", a, r, g, b)
	}
}

func equal(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
