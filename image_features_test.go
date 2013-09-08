package ML

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
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
	if (!aboutEqual(h, horizontal)) {
		t.Errorf (fmt.Sprintf("%s: expected horizontal count of %g, got %g", msg, horizontal, h))
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
	//    0 1 2 3 4 5 6 7
        // 0: _ _ _ _ _ _ _ _
        // 1: _ X _ _ _ _ _ _
        // 2: _ _ X _ _ _ _ _
        // 3: _ _ _ X _ _ _ _
        // 4: _ _ X X _ _ _ _
        // 5: _ _ _ _ _ _ _ _

	img := image.NewGray(image.Rect(0, 0, 8, 6))
	for i:=0; i<6; i++ {
		for j:=0; j<8; j++ {
			img.Set(j,i,color.Gray{uint8(0)})
		}
	}
	img.Set(1,1,color.Gray{255})
	img.Set(2,2,color.Gray{255})
	img.Set(3,3,color.Gray{255})
	img.Set(2,4,color.Gray{255})
	img.Set(3,4,color.Gray{255})

	//    0 1 2 3 4 5 6 7
        // 0: _ _ _ _ _ _ _ _
        // 1: _ _ X X X _ _ _
        // 2: _ _ X _ X _ _ _
        // 3: _ _ _ X _ _ _ _
        // 4: _ _ X _ X _ _ _
        // 5: _ _ X _ _ X _ _
        // 6: _ _ X X X X _ _
        // 7: _ _ _ _ _ _ _ _
	img2 := image.NewGray(image.Rect(0, 0, 8, 8))
	for i:=0; i<8; i++ {
		for j:=0; j<8; j++ {
			img.Set(j,i,color.Gray{uint8(0)})
		}
	}
	img2.Set(2,1,color.Gray{1})
	img2.Set(3,1,color.Gray{1})
	img2.Set(4,1,color.Gray{1})
	img2.Set(2,2,color.Gray{1})
	img2.Set(4,2,color.Gray{1})
	img2.Set(3,3,color.Gray{1})
	img2.Set(2,4,color.Gray{1})
	img2.Set(4,4,color.Gray{1})
	img2.Set(2,5,color.Gray{1})
	img2.Set(5,5,color.Gray{1})
	img2.Set(2,6,color.Gray{1})
	img2.Set(3,6,color.Gray{1})
	img2.Set(4,6,color.Gray{1})
	img2.Set(5,6,color.Gray{1})

	fmt.Printf ("Full original image: %v\n", img2)
	hf := NewHierarchicalFeatures(img2)
	hf.Dump(os.Stdout)

	xbar,ybar := hf.Centroid(image.Rect(1,1,3,3))
	fmt.Printf ("hf.Centroid(image.Rect(1,1,3,3)) = %g,%g\n", xbar, ybar)

	xbar,ybar = hf.Centroid(image.Rect(0,0,8,6))
	fmt.Printf ("hf.Centroid(image.Rect(0,0,8,6)) = %g,%g\n", xbar, ybar)

	for s:=0; s<1000; s++ {
		hf.RandomFeature(int32(s))
	}

	gf := GrayWithFeatures{img,0,0.0,nil}
	testImageEdges(t, "Image edges", gf, 4.0/3.0, 1.0)
	testImageCentroid(t, "Image centroid", gf, 2.2, 2.8)
	testImageMoments(t, "Image moments", gf, math.Sqrt(0.56), math.Sqrt(1.36), 0.64/math.Sqrt(0.56*1.36))

	fmt.Printf ("random feature(%d)=%g\n", 1234, gf.RandomFeature(1234))
	fmt.Printf ("random feature(%d)=%g\n", 4321, gf.RandomFeature(4321))
	fmt.Printf ("random feature(%d)=%g\n", 7531, gf.RandomFeature(7531))

	subimage := img.SubImage(image.Rect(1,1,3,3)).(*image.Gray)
	fmt.Printf ("subimage: %v\n", subimage)

	gfsi := GrayWithFeatures{subimage,0,0.0,nil}
	testImageEdges(t, "Image edge (1st subimage)", gfsi, 1.0, 1.0)
	testImageCentroid(t, "Image centroid (1st subimage)", gfsi, 0.5, 0.5)
	testImageMoments(t, "Image moments (1st subimage)", gfsi, 0.5, 0.5, 1.0)

	subimage = img.SubImage(image.Rect(3,3,4,4)).(*image.Gray)
	gfsi = GrayWithFeatures{subimage,0,0.0,nil}
	testImageEdges(t, "Image edges (2nd subimage)", gfsi, 0.0, 0.0)
	testImageCentroid(t, "Image centroid (2nd subimage)", gfsi, 0.0, 0.0)
	testImageMoments(t, "Image moments (2nd subimage)", gfsi, 0.0, 0.0, 0.0)
}
