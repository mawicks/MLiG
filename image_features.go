package ML

import (
	"image"
	"math"
)

type GrayWithFeatures struct {
	*image.Gray
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
