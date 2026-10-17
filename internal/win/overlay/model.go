package overlay

import "math"

// Resting pill sizes in device pixels.
const (
	idleW, idleH     = 44, 12
	listenW, listenH = 120, 32
	workW, workH     = 64, 20
)

// restSize is the size a state settles at.
func restSize(s State) (w, h int) {
	switch s {
	case Listening:
		return listenW, listenH
	case Working, Failed:
		return workW, workH
	}
	return idleW, idleH
}

// color is straight (non-premultiplied) RGBA with components in 0..1.
type color struct{ r, g, b, a float64 }

const nBarsListen = 9

var (
	pillIdle   = color{0.09, 0.09, 0.11, 0.72}
	pillLive   = color{0.09, 0.09, 0.11, 0.88}
	pillFailed = color{0.86, 0.15, 0.15, 0.95}
	inkWhite   = color{1, 1, 1, 1}
	accentRed  = color{0.94, 0.27, 0.27, 1}
	idleDotInk = color{1, 1, 1, 0.35}
	failDotInk = color{1, 1, 1, 0.90}
)

// pixel is premultiplied RGBA with components in 0..1.
type pixel struct{ r, g, b, a float64 }

type canvas struct {
	w, h int
	px   []pixel
}

// sdf is a signed distance function: negative inside a shape, positive outside.
type sdf func(x, y float64) float64

// paint composites col over the whole canvas, using d for coverage with 1 px anti-aliasing.
func (c *canvas) paint(d sdf, col color) {
	for y := 0; y < c.h; y++ {
		for x := 0; x < c.w; x++ {
			cov := 0.5 - d(float64(x)+0.5, float64(y)+0.5)
			if cov <= 0 {
				continue
			}
			if cov > 1 {
				cov = 1
			}
			a := col.a * cov
			k := 1 - a
			p := &c.px[y*c.w+x]
			p.r = col.r*a + p.r*k
			p.g = col.g*a + p.g*k
			p.b = col.b*a + p.b*k
			p.a = a + p.a*k
		}
	}
}

func circle(cx, cy, r float64) sdf {
	return func(x, y float64) float64 { return math.Hypot(x-cx, y-cy) - r }
}

// roundBox is a rectangle centred on (cx, cy) with half-sizes hw, hh and corner radius r.
func roundBox(cx, cy, hw, hh, r float64) sdf {
	return func(x, y float64) float64 {
		qx := math.Abs(x-cx) - (hw - r)
		qy := math.Abs(y-cy) - (hh - r)
		ox := math.Max(qx, 0)
		oy := math.Max(qy, 0)
		return math.Hypot(ox, oy) + math.Min(math.Max(qx, qy), 0) - r
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func to8(v float64) uint32 { return uint32(math.Round(clamp01(v) * 255)) }

// pack converts a premultiplied pixel to 0xAARRGGBB, which is the 32-bit BGRA layout of a top-down DIB.
func pack(p pixel) uint32 {
	a := to8(p.a)
	r := min(to8(p.r), a)
	g := min(to8(p.g), a)
	b := min(to8(p.b), a)
	return a<<24 | r<<16 | g<<8 | b
}

// render draws the pill for the given state into a w*h buffer of premultiplied 0xAARRGGBB pixels, row 0 at the top.
// level is the microphone level (only used when Listening) and t is the animation time in seconds.
// It returns nil when w or h is not positive.
func render(state State, level float32, t float64, w, h int) []uint32 {
	if w <= 0 || h <= 0 {
		return nil
	}
	cv := &canvas{w: w, h: h, px: make([]pixel, w*h)}
	fw, fh := float64(w), float64(h)
	cx, cy := fw/2, fh/2
	hw, hh := fw/2, fh/2
	pill := roundBox(cx, cy, hw, hh, math.Min(hw, hh))

	switch state {
	case Listening:
		cv.paint(pill, pillLive)
		cv.paint(circle(fh/2, cy, fh*0.13), accentRed)

		left, right := fw*0.28, fw*0.88
		slot := (right - left) / float64(nBarsListen)
		const barW = 3.0
		minH, maxH := fh*0.2, fh*0.75
		lv := math.Sqrt(float64(clamp01(float64(level))))
		for i := 0; i < nBarsListen; i++ {
			env := 0.5 + 0.5*math.Sin(t*6+float64(i)*0.9)
			bh := minH + (maxH-minH)*lv*(0.3+0.7*env)
			bx := left + slot*(float64(i)+0.5)
			cv.paint(roundBox(bx, cy, barW/2, bh/2, barW/2), inkWhite)
		}

	case Working:
		cv.paint(pill, pillLive)
		spacing := fw * 0.11
		for i := 0; i < 3; i++ {
			p := 0.5 + 0.5*math.Sin(2*math.Pi*t/1.2-float64(i)*0.9)
			col := inkWhite
			col.a = 0.35 + 0.65*p
			cv.paint(circle(cx+(float64(i)-1)*spacing, cy, 2+1.2*p), col)
		}

	case Failed:
		cv.paint(pill, pillFailed)
		cv.paint(circle(cx, cy, 2.2), failDotInk)

	default:
		cv.paint(pill, pillIdle)
		cv.paint(circle(cx, cy, 1.8), idleDotInk)
	}

	out := make([]uint32, w*h)
	for i, p := range cv.px {
		out[i] = pack(p)
	}
	return out
}
