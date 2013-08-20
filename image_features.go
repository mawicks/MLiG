package ML

import "image"

type GrayWithFeatures struct {
	*image.Gray
}

type featureSums struct {
	mass, xMass, yMass int32
	x2Mass, y2Mass, xyMass int64
	xEdges, yEdges int32
}

func iAbs(i int32) int32 {
	if i>=0 {
		return i
	}
	return -i
}	

func (gwf *GrayWithFeatures) featureSums() (fs featureSums) {
	rows := gwf.Rect.Dx()
	cols := gwf.Rect.Dy()

	// Work directly with pix array for performance reasons
	for i:=0; i<rows; i++ {
		offset := 0
		lastPix := gwf.Pix[offset]
		for j:=0; j<cols; j++ {
			index := offset + j
			pix := gwf.Pix[index]
			fs.mass += int32(pix)
			fs.xMass += int32(i)*int32(pix)
			fs.yMass += int32(j)*int32(pix)
			fs.x2Mass += int64(int32(i*i)*int32(pix))
			fs.y2Mass += int64(int32(j*j)*int32(pix))
			fs.xyMass += int64(int32(i*j)*int32(pix))
			fs.xEdges += iAbs(int32(pix)-int32(lastPix))
		}
		offset += gwf.Stride
	}

	for j:=0; j<cols; j++ {
		offset := j
		lastPix := gwf.Pix[offset]
		for i:=0; i<rows; i++ {
			pix := gwf.Pix[offset]
			fs.yEdges += iAbs(int32(pix)-int32(lastPix))
			offset += gwf.Stride
		}
	}
	return
}

func imageCentroid (fs featureSums) (x,y int32) {
	return fs.xMass/fs.mass,fs.yMass/fs.mass
}

func imageMoments (fs featureSums) (Mxx, Myy, Mxy int32) {
	Mxx = int32(int64(fs.x2Mass - int64(fs.xMass)*int64(fs.xMass))/int64(fs.mass))
	Myy = int32(int64(fs.y2Mass - int64(fs.yMass)*int64(fs.yMass))/int64(fs.mass))
	Mxy = int32(int64(fs.xyMass - int64(fs.xMass)*int64(fs.yMass))/int64(fs.mass))
	return
}



















