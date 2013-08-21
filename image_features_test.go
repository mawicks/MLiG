package ML

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"testing"
)

func aboutEqual(x, y float64) bool {
	const eps = 1.0e-14
	result := math.Abs(x-y) <= eps*math.Abs(x)
	return result
}

func testImageCentroid (t *testing.T, msg string, im GrayWithFeatures, cx, cy float64) {
	x, y := im.Centroid()
	if (!aboutEqual(cx, x)) {
		t.Errorf (fmt.Sprintf("%s: expected col centroid to be %g, got %g", msg, cx, x))
	}
	if (!aboutEqual(cy, y)) {
		t.Errorf (fmt.Sprintf("%s: expected row centroid to be %g, got %g", msg, cy, y))
	}
}

func testImageEdges (t *testing.T, msg string, im GrayWithFeatures, vertical, horizontal float64) {
	v, h := im.Edges()
	if (!aboutEqual(v, vertical)) {
		t.Errorf (fmt.Sprintf("%s: expected vertical count of %g, got %g", msg, vertical, v))
	}
	if (!aboutEqual(h, vertical)) {
		t.Errorf (fmt.Sprintf("%s: expected horizontal count of %g, got %g", msg, horizontal, v))
	}
}

func testImageMoments (t *testing.T, msg string, im GrayWithFeatures, rxx, ryy, rxy float64) {
	eRxx, eRyy, eRxy := im.Moments()
	if (!aboutEqual(eRxx, rxx)) {
		t.Errorf (fmt.Sprintf("%s: expected horizontal 2nd moment to be %g, got %g", msg, rxx, eRxx))
	}
	if (!aboutEqual(eRyy, ryy)) {
		t.Errorf (fmt.Sprintf("%s: expected vertical 2nd moment to be %g, got %g", msg, ryy, eRyy))
	}
	if (!aboutEqual(eRxy, rxy)) {
		t.Errorf (fmt.Sprintf("%s: expected moment coupling to be %g, got %g", msg, rxy, eRxy))
	}
}

func TestImageFeatures (t *testing.T) {
	//    0 1 2 3 4 5
        // 0: _ _ _ _ _ _
        // 1: _ X _ _ _ _
        // 2: _ _ X _ _ _
        // 3: _ _ _ X _ _
        // 4: _ _ X X _ _
        // 5: _ _ _ _ _ _

	img := image.NewGray(image.Rect(0, 0, 6, 6))
	img.Set(1,1,color.Gray{255})
	img.Set(2,2,color.Gray{255})
	img.Set(3,3,color.Gray{255})
	img.Set(4,2,color.Gray{255})
	img.Set(4,3,color.Gray{255})

	gf := GrayWithFeatures{img,0,0.0}
	testImageEdges(t, "Image edges", gf, 4.0/3.0, 4.0/3.0)
	testImageCentroid(t, "Image centroid", gf, 2.2, 2.8)
	testImageMoments(t, "Image moments", gf, math.Sqrt(0.56), math.Sqrt(1.36), 0.64/math.Sqrt(0.56*1.36))

	subimage := img.SubImage(image.Rect(1,1,3,3)).(*image.Gray)
	gf = GrayWithFeatures{subimage,0,0.0}
	testImageEdges(t, "Image edges", gf, 1.0, 1.0)
	testImageCentroid(t, "Image centroid", gf, 0.5, 0.5)
	testImageMoments(t, "Image moments", gf, 0.5, 0.5, 1.0)

	subimage = img.SubImage(image.Rect(3,3,4,4)).(*image.Gray)
	gf = GrayWithFeatures{subimage,0,0.0}
	testImageEdges(t, "Image edges", gf, 0.0, 0.0)
	testImageCentroid(t, "Image centroid", gf, 0.0, 0.0)
	testImageMoments(t, "Image moments", gf, 0.0, 0.0, 0.0)
}



