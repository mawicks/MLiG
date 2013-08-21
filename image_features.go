package ML

import (
	"image"
	"math"
)

type GrayWithFeatures struct {
	*image.Gray
	memoSeed int32
	memoValue float64
}

type featureSums struct {
	mass, xMass, yMass int32
	x2Mass, y2Mass, xyMass int64
	xEdges, yEdges int32
	rows, cols int
}

func iAbs(i int32) int32 {
	if i>=0 {
		return i
	}
	return -i
}	

func (gwf *GrayWithFeatures) featureSums() (fs featureSums) {
	fs.rows = gwf.Rect.Dx()
	fs.cols = gwf.Rect.Dy()

	// Work directly with pix array for performance reasons
	offset := 0
	for i:=0; i<fs.rows; i++ {
		lastPix := gwf.Pix[offset]
		for j:=0; j<fs.cols; j++ {
			index := offset + j
			pix := gwf.Pix[index]
			fs.mass += int32(pix)
			fs.xMass += int32(i)*int32(pix)
			fs.yMass += int32(j)*int32(pix)
			fs.x2Mass += int64(int32(i*i)*int32(pix))
			fs.y2Mass += int64(int32(j*j)*int32(pix))
			fs.xyMass += int64(int32(i*j)*int32(pix))
			fs.xEdges += iAbs(int32(pix)-int32(lastPix))
			lastPix = pix
		}
		offset += gwf.Stride
	}

	for j:=0; j<fs.cols; j++ {
		offset := j
		lastPix := gwf.Pix[offset]
		for i:=0; i<fs.rows; i++ {
			pix := gwf.Pix[offset]
			fs.yEdges += iAbs(int32(pix)-int32(lastPix))
			lastPix = pix
			offset += gwf.Stride
		}
	}
	return
}

func centroidFromSums(fs featureSums) (x, y float64) {
	if fs.mass != 0 {
		x,y = float64(fs.xMass)/float64(fs.mass),float64(fs.yMass)/float64(fs.mass)
	}
	return
}

func (gwf *GrayWithFeatures) Centroid () (x,y float64) {
	fs := gwf.featureSums()
	return centroidFromSums(fs)
}

func (gwf *GrayWithFeatures) Moments () (rxx, ryy, rxy float64) {
	fs := gwf.featureSums()
	if fs.mass != 0 {
		cX, cY := centroidFromSums(fs)
		
		r2xx := float64(fs.x2Mass)/float64(fs.mass) - cX*cX
		r2yy := float64(fs.y2Mass)/float64(fs.mass) - cY*cY
		
		if r2xx > 0.0 {
			rxx = math.Sqrt(r2xx)
		}
		
		if r2yy > 0.0 {
			ryy = math.Sqrt(r2yy)
		}

		if rxx > 0.0 && ryy > 0.0 {
			rxy = (float64(fs.xyMass)/float64(fs.mass) - cX*cY)/rxx/ryy
		}
	}
	return
}

// imageEdges returns the average number of vertical and horizontal
// edges in the image.  For vertical edges the average is taken over all rows.
// For horizontal edges the average is taken over all columns.
func (gwf *GrayWithFeatures) Edges() (vertical, horizontal float64) {
	fs := gwf.featureSums()
	vertical = float64(fs.xEdges)/float64(fs.rows)/float64(255.0)
	horizontal = float64(fs.yEdges)/float64(fs.cols)/float64(255.0)
	return
}

// randomRectangle() returns a random rectangle obtained from the
// random seed "s".  As entropy from "s" is consumed to produce a
// random rectangle, it may be used to generate additional random
// selections.  This entropy is removed.  The returned "s" is a
// revised "s" with all consumed entropy removed.

func randomRectangle (s int32, dx, dy int) (image.Rectangle,int32) {
	// Choose random window
	x1 := int(s % int32(dx))
	
	// As bits of "s" are consumed, remove them from s.
	s /= int32(dx)
	
	y1 := int(s % int32(dy))
	s /= int32(dy)
	
	x2 := x1 + 1 + int(s % int32(dx-x1))
	s /= int32(dx-x1)
	
	y2 := y1 + 1 + int(s % int32(dy-y1))
	s /= int32(dy-y1)

	return image.Rect(x1, x2, y1, y2), s
}

func (gwf *GrayWithFeatures) RandomFeature(s int32) float64 {
	// Select a random feature assuming "s" is a random 32-bit
	// integer.  The same "s" should *always* produce the same
	// result on the same image.  The same "s" should always
	// select the same "feature" (same window, same attribute,
	// etc.)  regardless of the data values.

	// Use memoized lookups in case called with same "s" is used
	// repeatedly (e.g., in sorts).

	if s == gwf.memoSeed {
		return gwf.memoValue
	}
	
	dx := gwf.Rect.Dx()
	dy := gwf.Rect.Dy()

	var subRect image.Rectangle

	if dx != 0 && dy != 0 {
		gwf.memoSeed = s
		subRect,s = randomRectangle (s, dx, dy)

		subimage := GrayWithFeatures{gwf.SubImage (subRect).(*image.Gray),0,0.0}
		
		switch (s%7) {
		case 0:
			gwf.memoValue,_ = subimage.Centroid()
		case 1:
			_,gwf.memoValue = subimage.Centroid()
		case 2:
			gwf.memoValue,_,_ = subimage.Moments()
		case 3:
			_,gwf.memoValue,_ = subimage.Moments()
		case 4:
			_,_,gwf.memoValue = subimage.Moments()
		case 5:
			gwf.memoValue,_ = subimage.Edges()
		case 6:
			_,gwf.memoValue = subimage.Edges()
		}
	}
	return gwf.memoValue
}
