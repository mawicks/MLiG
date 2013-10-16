package ML

import (
	"math"
	"testing"
	"os"
)

func testDataMSE (t *testing.T, msg string, data []*Data, seed int32, output int, expectedSplit, expectedLeftError, expectedRightError float64, size int) {
	splitInfo := continuousFeatureSplit(data, seed, StatAccumulatorFactory())
	if (splitInfo.splitValue != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, splitInfo.splitValue)
	}
	if math.Abs(splitInfo.left.Metric()-expectedLeftError) > 1e-12*math.Abs(expectedLeftError) {
		t.Errorf ("%s: expected left error: %v; got: %v", msg, expectedLeftError, splitInfo.left.Metric())
	}
	if math.Abs(splitInfo.right.Metric()-expectedRightError) > 1e-12*math.Abs(expectedRightError) {
		t.Errorf ("%s: expected right error: %v; got: %v", msg, expectedRightError, splitInfo.right.Metric())
	}
	if splitInfo.left.Count() != size {
		t.Errorf ("%s: expected left split size: %v; got: %v", msg, size, splitInfo.left.Count())
	}
}

func TestContinuousFeatureMSESplit (t *testing.T) {
	test1 := []*Data {
		&Data{continuousFeatures: []float64 {3.0}, output: 7.0, featureSelector: func (int32) float64 {return 3.0}},
		&Data{continuousFeatures: []float64 {1.0}, output: 4.0, featureSelector: func (int32) float64 {return 1.0}},
		&Data{continuousFeatures: []float64 {2.0}, output: 4.0, featureSelector: func (int32) float64 {return 2.0}}}

	test2 := []*Data {
		&Data{continuousFeatures: []float64{2.0}, output: 5.0, featureSelector: func (int32) float64 {return 2.0}},
		&Data{continuousFeatures: []float64{1.0}, output: 3.0, featureSelector: func(int32) float64 {return 1.0}},
		&Data{continuousFeatures: []float64{3.0}, output: 6.0, featureSelector: func(int32) float64 {return 3.0}}}

	test3 := []*Data{
		&Data{continuousFeatures: []float64{2.0}, output: 5.0, featureSelector: func(int32) float64 {return 2.0}},
		&Data{continuousFeatures: []float64{2.0}, output: 3.0, featureSelector: func(int32) float64 {return 2.0}},
		&Data{continuousFeatures: []float64{1.0}, output: 3.0, featureSelector: func(int32) float64 {return 1.0}},
		&Data{continuousFeatures: []float64{3.0}, output: 6.0, featureSelector: func(int32) float64 {return 3.0}}}

	// Remember that left branch consists of values < split
	// right branch branch consists of values >= split

	testDataMSE (t, "test1", test1, 0, 1, 3.0, 0.0, 0.0, 2)
	testDataMSE (t, "test2", test2, 0, 1, 2.0, 0.0, 0.25, 1)
	testDataMSE (t, "test3", test3, 0, 1, 3.0, 8.0/9.0, 0.0, 3)
}

func testDataEntropy (t *testing.T, msg string, data []*Data, seed int32, outputValueCount int, expectedSplit, expectedLeftEntropy, expectedRightEntropy float64, size int) {
//	f := continuousFeatureEntropySplitter (outputValueCount)
//	splitInfo := f(data, seed)
	splitInfo := continuousFeatureSplit (data, seed, EntropyAccumulatorFactory(outputValueCount))
	if (splitInfo.splitValue != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, splitInfo.splitValue)
	}
	if math.Abs(splitInfo.left.Metric()-expectedLeftEntropy) > 1e-12*math.Abs(expectedLeftEntropy) {
		t.Errorf ("%s: expected left entropy: %v; got: %v", msg, expectedLeftEntropy, splitInfo.left.Count())
	}
	if math.Abs(splitInfo.right.Metric()-expectedRightEntropy) > 1e-12*math.Abs(expectedRightEntropy) {
		t.Errorf ("%s: expected right entropy: %v; got: %v", msg, expectedRightEntropy, splitInfo.right.Count())
	}
	if splitInfo.left.Count() != size {
		t.Errorf ("%s: expected left split size: %v; got: %v", msg, size, splitInfo.left.Count())
	}
}

func TestContinuousFeatureEntropySplit (t *testing.T) {
	test1 := []*Data {
		&Data{continuousFeatures: []float64{3.0}, output: 3.0, featureSelector: func (int32) float64 {return 3.0}},
		&Data{continuousFeatures: []float64{1.0}, output: 1.0, featureSelector: func (int32) float64 {return 1.0}},
		&Data{continuousFeatures: []float64{2.0}, output: 2.0, featureSelector: func (int32) float64 {return 2.0}},
		&Data{continuousFeatures: []float64{3.0}, output: 3.0, featureSelector: func (int32) float64 {return 3.0}}}
	
	test2 := []*Data {
		&Data{continuousFeatures: []float64{3.0}, output: 2.0, featureSelector: func (int32) float64 {return 3.0}},
		&Data{continuousFeatures: []float64{1.0}, output: 1.0, featureSelector: func (int32) float64 {return 1.0}},
		&Data{continuousFeatures: []float64{2.0}, output: 2.0, featureSelector: func (int32) float64 {return 2.0}},
		&Data{continuousFeatures: []float64{3.0}, output: 2.0, featureSelector: func (int32) float64 {return 2.0}}}

	// Remember that left branch consists of values < split
	// right branch branch consists of values >= split
	testDataEntropy (t, "test1", test1, 0, 5, 3.0, 1.0, 0.0, 2)
	testDataEntropy (t, "test2", test2, 0, 3, 2.0, 0.0, 0.0, 1)
}

func TestGrow (t *testing.T) {
	test := []*Data{
		&Data{continuousFeatures: []float64{0.0, 1.0}, output: 1.0,
			featureSelector: func (s int32) float64 {return []float64{0.0,1.0}[s%2]}},
		&Data{continuousFeatures: []float64{1.0, 1.0}, output: 1.0,
			featureSelector: func (s int32) float64 {return []float64{1.0,1.0}[s%2]}},
		&Data{continuousFeatures: []float64{2.0, 1.0}, output: 1.0,
			featureSelector: func (s int32) float64 {return []float64{2.0,1.0}[s%2]}},
		&Data{continuousFeatures: []float64{2.0, 2.0}, output: 2.0,
			featureSelector: func (s int32) float64 {return []float64{2.0,2.0}[s%2]}},
		&Data{continuousFeatures: []float64{3.0, 1.0}, output: 2.0,
			featureSelector: func (s int32) float64 {return []float64{3.0,1.0}[s%2]}},
		&Data{continuousFeatures: []float64{4.0, 0.0}, output: 2.0,
			featureSelector: func (s int32) float64 {return []float64{4.0,0.0}[s%2]}}}

	factory := EntropyAccumulatorFactory(3)
	accumulator := factory()
	for _,d := range test {
		accumulator.Add(d.output)
	}
	treeNode := NewTreeNode (accumulator)

//	f := continuousFeatureEntropySplitter (3)

	treeNode.grow(test, 10, 1, 128, factory)
	for _,d := range test {
		if d.output != treeNode.classify(d.featureSelector) {
			t.Errorf ("%g classified as %g\n", d.output, treeNode.classify(d.featureSelector))
		}
	}
	
	treeNode.dump(os.Stdout, 0, 0)
}
