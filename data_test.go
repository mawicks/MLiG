package ML

import (
	"testing"
	"fmt"
	"os"
)

func TestGlassData (t *testing.T) {
	const filename string = "Data/glass.data"

	glassData := GlassData(filename)
	fmt.Fprintf (os.Stdout, "Read %d records from \"%s\"\n", len(glassData), filename)

	f := continuousFeatureEntropySplitter (glassData[0].outputCategories)
	ensemble := NewTreeEnsemble()

	for i:=0; i<1000; i++ {
		newTree := NewBaggedTree(glassData, 4, f)
		fmt.Printf ("Tree %d stats - size: %d  depth: %d\n", i, newTree.Size(), newTree.Depth())
		ensemble.AddTree(newTree)
		mserror := ensemble.Error(glassData)
		if i % 100 == 0 {
			fmt.Printf ("Trees: %d: error=%g\n", i, mserror)
		}
	}		
}

func TestDigitData (t *testing.T) {
	const filename string = "Data/digits-train.csv"

	digitData := DigitData(filename)
	fmt.Fprintf (os.Stdout, "Read %d records from \"%s\"\n", len(digitData), filename)

	f := continuousFeatureEntropySplitter (digitData[0].outputCategories)
	ensemble := NewTreeEnsemble()

	for i:=0; i<1000; i++ {
		newTree := NewBaggedTree(digitData, 10, f)
		ensemble.AddTree(newTree)
		fmt.Printf ("Tree %d stats - size: %d  depth: %d\n", i, newTree.Size(), newTree.Depth())
		mserror := ensemble.Error(digitData)
		if i % 1 == 0 {
			fmt.Printf ("Trees: %d: error=%g\n", i, mserror)
		}
	}		
}

















