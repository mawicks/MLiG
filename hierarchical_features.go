package ML

import (
	"errors"
	"fmt"
	"image"
	"io"
	"io/ioutil"
//	"os"
	"math"
)

type HierarchicalFeatures struct {
	*image.Gray
	memoSeed int32
	memoValue float64
	randomFeatureSelector RandomFeatureSelector
	cumMass, cumXMass, cumYMass []int64
	cumX2Mass, cumY2Mass, cumXYMass []int64
}

var hierarchicalDebug = ioutil.Discard

func NewHierarchicalFeatures(gs *image.Gray) *HierarchicalFeatures {
	l := len(gs.Pix)
	hf := HierarchicalFeatures{
		Gray: gs,
		memoSeed: -1,
		memoValue: 0.0,
		randomFeatureSelector: nil,
		cumMass: make([]int64,l),
		cumXMass: make([]int64,l),
		cumYMass: make([]int64,l),
		cumX2Mass: make([]int64,l),
		cumY2Mass: make([]int64,l),
		cumXYMass: make([]int64,l)}
	
	
	rows := gs.Rect.Dy()
	cols := gs.Rect.Dx()
	
	offset := 0
	for i:=0; i<rows; i++ {
		for j:=0; j<cols; j++ {
			index := offset + j
			pix := gs.Pix[index]
			hf.cumMass[index] = int64(pix)
			hf.cumXMass[index] = int64(int32(j)*int32(pix))
			hf.cumYMass[index] = int64(int32(i)*int32(pix))
			hf.cumX2Mass[index] = int64(int32(j*j)*int32(pix))
			hf.cumY2Mass[index] = int64(int32(i*i)*int32(pix))
			hf.cumXYMass[index] = int64(int32(i)*int32(j)*int32(pix))
			if i>0 {
				hf.cumMass[index]  += hf.cumMass[index-gs.Stride]
				hf.cumXMass[index] += hf.cumXMass[index-gs.Stride]
				hf.cumYMass[index] += hf.cumYMass[index-gs.Stride]
				hf.cumX2Mass[index] += hf.cumX2Mass[index-gs.Stride]
				hf.cumY2Mass[index] += hf.cumY2Mass[index-gs.Stride]
				hf.cumXYMass[index] += hf.cumXYMass[index-gs.Stride]
			}
			if j>0 {
				hf.cumMass[index]  += hf.cumMass[index-1]
				hf.cumXMass[index] += hf.cumXMass[index-1]
				hf.cumYMass[index] += hf.cumYMass[index-1]
				hf.cumX2Mass[index] += hf.cumX2Mass[index-1]
				hf.cumY2Mass[index] += hf.cumY2Mass[index-1]
				hf.cumXYMass[index] += hf.cumXYMass[index-1]
			}
			if i>0 && j > 0 {
				hf.cumMass[index]  -= hf.cumMass[index-gs.Stride-1]
				hf.cumXMass[index] -= hf.cumXMass[index-gs.Stride-1]
				hf.cumYMass[index] -= hf.cumYMass[index-gs.Stride-1]
				hf.cumX2Mass[index] -= hf.cumX2Mass[index-gs.Stride-1]
				hf.cumY2Mass[index] -= hf.cumY2Mass[index-gs.Stride-1]
				hf.cumXYMass[index] -= hf.cumXYMass[index-gs.Stride-1]
			}
		}
		offset += gs.Stride
	}
	return &hf
}

func (hf *HierarchicalFeatures) Dump(w io.Writer) {
	rows := hf.Gray.Rect.Dy()
	cols := hf.Gray.Rect.Dx()
	index := 0
	for j:=0; j<cols; j++ {
		if j != 0 {
			fmt.Fprintf(w, "  ")
		}
		fmt.Fprintf(w, "%14d", j)
	}
	fmt.Fprintf(w, "\n")

	for i:=0; i<rows; i++ {
		fmt.Fprintf(w, "%3d: ", i);
		for j:=0; j<cols; j++ {
			if j!=0 {
				fmt.Fprintf(w, "  ")
			}
			fmt.Fprintf(w, "%4d/%4d/%4d", hf.cumMass[index],hf.cumXMass[index],hf.cumYMass[index])
			index += 1
		}
		fmt.Fprintf(w, "\n")
	}
}

func (hf *HierarchicalFeatures) MassSums(r image.Rectangle) (mass, xMass, yMass, x2Mass, y2Mass, xyMass int64) {
	// Remember that Max.Y and Max.X are *outside* of the box
	// lowerRightIndex is the last point *inside* the box.

	lowerRightIndex := (r.Max.Y-1)*hf.Gray.Stride + (r.Max.X-1)
	mass  = hf.cumMass[lowerRightIndex]
	xMass = hf.cumXMass[lowerRightIndex]
	yMass = hf.cumYMass[lowerRightIndex]
	x2Mass = hf.cumX2Mass[lowerRightIndex]
	y2Mass = hf.cumY2Mass[lowerRightIndex]
	xyMass = hf.cumXYMass[lowerRightIndex]

	if r.Min.X>0 {
		lowerLeftIndex  := (r.Max.Y-1)*hf.Gray.Stride + (r.Min.X-1)
		mass  -= hf.cumMass[lowerLeftIndex]
		xMass -= hf.cumXMass[lowerLeftIndex]
		yMass -= hf.cumYMass[lowerLeftIndex]
		x2Mass -= hf.cumX2Mass[lowerLeftIndex]
		y2Mass -= hf.cumY2Mass[lowerLeftIndex]
		xyMass -= hf.cumXYMass[lowerLeftIndex]
	}

	if r.Min.Y>0 {
		upperRightIndex := (r.Min.Y-1)*hf.Gray.Stride + (r.Max.X-1)
		mass  -= hf.cumMass[upperRightIndex]
		xMass -= hf.cumXMass[upperRightIndex]
		yMass -= hf.cumYMass[upperRightIndex]
		x2Mass -= hf.cumX2Mass[upperRightIndex]
		y2Mass -= hf.cumY2Mass[upperRightIndex]
		xyMass -= hf.cumXYMass[upperRightIndex]
	}

	if r.Min.X>0 && r.Min.Y>0 {
		upperLeftIndex  := (r.Min.Y-1)*hf.Gray.Stride + (r.Min.X-1)
		mass  += hf.cumMass[upperLeftIndex]
		xMass += hf.cumXMass[upperLeftIndex]
		yMass += hf.cumYMass[upperLeftIndex]
		x2Mass += hf.cumX2Mass[upperLeftIndex]
		y2Mass += hf.cumY2Mass[upperLeftIndex]
		xyMass += hf.cumXYMass[upperLeftIndex]
	}

	return mass, xMass, yMass, x2Mass,y2Mass, xyMass
}

func (hf *HierarchicalFeatures) Centroid (r image.Rectangle) (xBar,yBar float64) {
	mass,xMass,yMass,_,_,_ := hf.MassSums(r)
	xBar = float64(xMass) / float64(mass)
	yBar = float64(yMass) / float64(mass)
	return xBar,yBar
}

func (hf *HierarchicalFeatures) RandomFeature(s int32) float64 {
//	fmt.Fprintf (hierarchicalDebug, "RandomFeature(s=%d)\n", s)
	// Use memoized lookups in case same "s" is used repeatedly
	// (e.g., in sorts).
//	fmt.Fprintf (hierarchicalDebug, "%*smemoSeed=%d, memoValue=%g\n", 3, "", hf.memoSeed, hf.memoValue)
	if s == hf.memoSeed {
		return hf.memoValue
	}
	
	hf.memoSeed = s
	hf.memoValue = hf.randomFeatureHelper(0, s, image.Rect(0, 0, hf.Gray.Rect.Dx(), hf.Gray.Rect.Dy()), 0.0, 0.0)
//	fmt.Fprintf (hierarchicalDebug, "\n")

	return hf.memoValue
}

func dbg (depth int, s string) {
//	fmt.Fprintf (hierarchicalDebug, "%*s\n", 3*depth, s)
}

// Select a random feature based on the entropy s.  The feature is
// selected from rectangle r.  This is a hierarchical feature which
// returns centroid location displacements relative to the passed
// coordinate (xBar0, yBar0).  This coordinate is the centroid of the
// parent rectangle, which contains the passed rectangle.  The parent
// rectangle is split into quadrants approximately at (xBar0,yBar0).
// Any returned centroid coordinates are relative to (xBar0,yBar0)
func (hf *HierarchicalFeatures) randomFeatureHelper(depth int, s int32, r image.Rectangle, xBar0,yBar0 float64) (result float64) {
//	fmt.Fprintf (hierarchicalDebug, "%*srandomFeatureHelper(s=%d, r=%v, xBar0=%g, yBar0=%g)\n", depth*3, "", s, r, xBar0, yBar0)

	if r.Dx() == 0 || r.Dy() == 0 {
		panic ("Rectangle should not have zero dimension")
	}

	// Select a random feature assuming "s" is a random 32-bit
	// integer.  The same "s" should *always* produce the same
	// result on the same image.  The same "s" should always
	// select the same "feature" (same window, same attribute,
	// etc.)  regardless of the data values.

	// Lowest two bits determine whether recursion should terminate
	// and, if so, which feature to select:

	mass,xMass,yMass,x2Mass,y2Mass,xyMass := hf.MassSums(r)

	// Centroid displacement is zero when mass is zero.
	xBar := xBar0
	yBar := yBar0
	sigma2X := 0.0
	sigma2Y := 0.0
	sigmaXY := 0.0

	if mass != 0 {
		xBar = float64(xMass)/float64(mass)
		yBar = float64(yMass)/float64(mass)
		sigma2X = float64(x2Mass)/float64(mass) - xBar*xBar
		sigma2Y = float64(y2Mass)/float64(mass) - yBar*yBar
		sigmaXY = float64(xyMass)/float64(mass) - xBar*yBar
	}

	// Recurse xx% of the time
	recurse := s % 2
	s = s / 2
	if recurse > 0 { // 70
		// When recursing deeper, next two bits of "s" select the quadrant to pass to the next layer
		// 00 - Upper left
		// 01 - Upper right
		// 10 - Lower left
		// 11 - Lower right
		quadrant := s % 4
		s = s >> 2
		
		x0 := int(math.Ceil(xBar))
		y0 := int(math.Ceil(yBar))
		
		switch (quadrant) {
		case 0:
			dbg(depth, "Upper left")
			r = image.Rect(r.Min.X,r.Min.Y,x0+1,y0+1)
		case 1:
			dbg(depth, "Upper right")
			r = image.Rect(x0,r.Min.Y,r.Max.X,y0+1)
		case 2:
			dbg(depth, "Lower left")
			r = image.Rect(r.Min.X,y0,x0+1,r.Max.Y)
		case 3:
			dbg(depth, "Lower right")
			r = image.Rect(x0,y0,r.Max.X,r.Max.Y)
		default:
			panic (errors.New("Default of quadrant selection switch.  This should never happen"))
		}
		result = hf.randomFeatureHelper (depth+1, s, r, xBar, yBar)
	} else {
		feature := s % 6
		// 00 - Terminate and return "mass"
		// 01 - Terminate and return xbar
		// 10 - Terminate and return ybar
		
		switch feature {
		case 0:
			dbg(depth, "*Returning mass*")
			result = float64(mass)
		case 1:
			dbg(depth, "*Returning x bar*")
			result = xBar - xBar0
		case 2:
			dbg(depth, "*Returning y bar*")
			result = yBar - yBar0
		case 3:
			dbg(depth, "*Returning sigma xx*")
			result = sigma2X
		case 4:
			dbg(depth, "*Returning sigma yy*")
			result = sigma2Y
		case 5:
			dbg(depth, "*Returning sigma xy*")
			result = sigmaXY
		default:
			panic (errors.New("Default of feature selection switch.  This should never happen"))
		}
	}
//	fmt.Fprintf (hierarchicalDebug, "%*s%g\n", 3*depth, "", result)
	return result
}














