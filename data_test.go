package ML

import (
	"fmt"
	"image"
	"testing"
	"time"
	"os"
)

func XTestGlassData (t *testing.T) {
	const filename string = "Data/glass.data"

	glassData := GlassData(filename)
	fmt.Fprintf (os.Stdout, "Read %d records from \"%s\"\n", len(glassData), filename)

	f := continuousFeatureEntropySplitter (glassData[0].outputCategories)
	ensemble := NewEnsemble()

	for i:=0; i<1000; i++ {
		newTree := NewTree(4, f)
		TrainBag(glassData, newTree)
		fmt.Printf ("Tree %d stats - size: %d  depth: %d\n", i, newTree.Size(), newTree.Depth())
		fmt.Printf ("Tree performance: %g\n", newTree.Estimate())
		ensemble.AddClassifier(newTree)
		mserror := ensemble.Error(glassData)
		if i % 100 == 0 {
			fmt.Printf ("Trees: %d: ensemble error=%g\n", i, mserror)
		}
	}		
}

func TestDigitData (t *testing.T) {
	const filename string = "Data/digits-train.csv"

	start := time.Now()
	fmt.Fprintf (os.Stdout, "%s: Reading input file...", start)
	digitData := DigitData(filename)
	fmt.Fprintf (os.Stdout, "Read %d records from \"%s\" (%s)\n", len(digitData), filename, time.Now().Sub(start))

	ShuffleData(digitData)
//	digitData = digitData[0:2000]

	for i,_ := range digitData {
		pix := make([]uint8, len(digitData[i].continuousFeatures))
		for i,f := range digitData[i].continuousFeatures {
			pix[i] = uint8(f)
		}
		grayImage := image.Gray{pix,28,image.Rect(0,0,28,28)}
//		gwf := GrayWithFeatures{&grayImage,0,0.0,nil}
		hf := NewHierarchicalFeatures(&grayImage)
		digitData[i].featureSelector = func (s int32) float64 { return hf.RandomFeature(s) }
	}

//	start = time.Now()
//	fmt.Fprintf (os.Stdout, "%s: Computing PCA basis...", start)
//	s,v := pcaBasis(digitData)
//	fmt.Printf ("done (%s).\n", time.Now().Sub(start))

//	start = time.Now()
//	fmt.Fprintf (os.Stdout, "%s: Changing basis and reducing order...", start)
//	pcaChangeBasis(digitData, s, v, .005)
//	fmt.Printf ("done (%s).\n", time.Now().Sub(start))

	f := continuousFeatureEntropySplitter (digitData[0].outputCategories)
	ensemble := NewEnsemble()

	for i:=0; i<1000; i++ {
		newTree := NewTree(30, f)
		TrainBag(digitData, newTree)
		ensemble.AddClassifier(newTree)
		fmt.Printf ("Tree %d stats - size: %d  depth: %d performance: %g\n", i, newTree.Size(), newTree.Depth(), newTree.Estimate())
		mserror := ensemble.Error(digitData)
		if i % 1 == 0 {
			fmt.Printf ("Trees: %d: ensemble error=%g\n", i, mserror)
		}
	}		
}
