package ML

import (
	"math"
	"testing"
	"os"
)

func testDataMSE (t *testing.T, msg string, data []*Data, featureSelector FeatureSelector, output int, expectedSplit, expectedLeftError, expectedRightError float64, size int) {
	splitInfo := continuousFeatureMSESplit(data, featureSelector)
	if (splitInfo.splitValue != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, splitInfo.splitValue)
	}
	if math.Abs(splitInfo.leftSplitMetric-expectedLeftError) > 1e-12*math.Abs(expectedLeftError) {
		t.Errorf ("%s: expected left error: %v; got: %v", msg, expectedLeftError, splitInfo.leftSplitMetric)
	}
	if math.Abs(splitInfo.rightSplitMetric-expectedRightError) > 1e-12*math.Abs(expectedRightError) {
		t.Errorf ("%s: expected right error: %v; got: %v", msg, expectedRightError, splitInfo.rightSplitMetric)
	}
	if splitInfo.leftSplitSize != size {
		t.Errorf ("%s: expected left split size: %v; got: %v", msg, size, splitInfo.leftSplitSize)
	}
}

func TestContinuousFeatureMSESplit (t *testing.T) {
	test1 := []*Data {
		&Data{continuousFeatures: []float64 {3.0}, output: 7.0},
		&Data{continuousFeatures: []float64 {1.0}, output: 4.0},
		&Data{continuousFeatures: []float64 {2.0}, output: 4.0}}

	test2 := []*Data {
		&Data{continuousFeatures: []float64{2.0}, output: 5.0},
		&Data{continuousFeatures: []float64{1.0}, output: 3.0},
		&Data{continuousFeatures: []float64{3.0}, output: 6.0}}

	test3 := []*Data{
		&Data{continuousFeatures: []float64{2.0}, output: 5.0},
		&Data{continuousFeatures: []float64{2.0}, output: 3.0},
		&Data{continuousFeatures: []float64{1.0}, output: 3.0},
		&Data{continuousFeatures: []float64{3.0}, output: 6.0}}

	// Remember that left branch consists of values < split
	// right branch branch consists of values >= split

	featureSelector := featureSelectorFromIndex(0)

	testDataMSE (t, "test1", test1, featureSelector, 1, 3.0, 0.0, 0.0, 2)
	testDataMSE (t, "test2", test2, featureSelector, 1, 2.0, 0.0, 0.25, 1)
	testDataMSE (t, "test3", test3, featureSelector, 1, 3.0, 8.0/9.0, 0.0, 3)
}

func testDataEntropy (t *testing.T, msg string, data []*Data, featureSelector FeatureSelector, outputValueCount int, expectedSplit, expectedLeftEntropy, expectedRightEntropy float64, size int) {
	f := continuousFeatureEntropySplitter (outputValueCount)
	splitInfo := f(data, featureSelector)
	if (splitInfo.splitValue != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, splitInfo.splitValue)
	}
	if math.Abs(splitInfo.leftSplitMetric-expectedLeftEntropy) > 1e-12*math.Abs(expectedLeftEntropy) {
		t.Errorf ("%s: expected left entropy: %v; got: %v", msg, expectedLeftEntropy, splitInfo.leftSplitMetric)
	}
	if math.Abs(splitInfo.rightSplitMetric-expectedRightEntropy) > 1e-12*math.Abs(expectedRightEntropy) {
		t.Errorf ("%s: expected right entropy: %v; got: %v", msg, expectedRightEntropy, splitInfo.rightSplitMetric)
	}
	if splitInfo.leftSplitSize != size {
		t.Errorf ("%s: expected left split size: %v; got: %v", msg, size, splitInfo.leftSplitSize)
	}
}

func TestContinuousFeatureEntropySplit (t *testing.T) {
	test1 := []*Data {
		&Data{continuousFeatures: []float64{3.0}, output: 3.0},
		&Data{continuousFeatures: []float64{1.0}, output: 1.0},
		&Data{continuousFeatures: []float64{2.0}, output: 2.0},
		&Data{continuousFeatures: []float64{3.0}, output: 3.0}}
	
	test2 := []*Data {
		&Data{continuousFeatures: []float64{3.0}, output: 2.0},
		&Data{continuousFeatures: []float64{1.0}, output: 1.0},
		&Data{continuousFeatures: []float64{2.0}, output: 2.0},
		&Data{continuousFeatures: []float64{3.0}, output: 2.0}}

	featureSelector := featureSelectorFromIndex(0)

	// Remember that left branch consists of values < split
	// right branch branch consists of values >= split
	testDataEntropy (t, "test1", test1, featureSelector, 5, 3.0, 1.0, 0.0, 2)
	testDataEntropy (t, "test2", test2, featureSelector, 3, 2.0, 0.0, 0.0, 1)
}

func TestGrow (t *testing.T) {
	test := []*Data{
		&Data{continuousFeatures: []float64{0.0, 1.0}, output: 1.0},
		&Data{continuousFeatures: []float64{1.0, 1.0}, output: 1.0},
		&Data{continuousFeatures: []float64{2.0, 1.0}, output: 1.0},
		&Data{continuousFeatures: []float64{2.0, 2.0}, output: 2.0},
		&Data{continuousFeatures: []float64{3.0, 1.0}, output: 2.0},
		&Data{continuousFeatures: []float64{4.0, 0.0}, output: 2.0}}

	treeNode := NewTreeNode (math.MaxFloat64,math.MaxFloat64)
	f := continuousFeatureEntropySplitter (3)

	treeNode.grow(test, 128, f)
	for _,d := range test {
		if d.output != treeNode.classify(d.continuousFeatures) {
			t.Errorf ("%g classified as %g\n", d.output, treeNode.classify(d.continuousFeatures))
		}
	}
	
	treeNode.dump(os.Stdout, 0, 0)
}

