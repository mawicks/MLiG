package ML

import (
	"math"
	"testing"
)

func testDataMSE (t *testing.T, msg string, data [][]float64, feature, output int, expectedSplit, expectedError float64) {
	split, error,_  := continuousFeatureMSESplit(data, feature, output)
	if (split != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, split)
	}
	if math.Abs(error-expectedError) > 1e-12*math.Abs(expectedError) {
		t.Errorf ("%s: expected error: %v; got: %v", msg, expectedError, error)
	}
}

func TestContinuousFeatureMSESplit (t *testing.T) {
	test1 := [][]float64{
		{3.0, 7.0},
		{1.0, 4.0},
		{2.0, 4.0}}

	test2 := [][]float64{
		{2.0, 5.0},
		{1.0, 3.0},
		{3.0, 6.0}}

	test3 := [][]float64{
		{2.0, 5.0},
		{2.0, 3.0},
		{1.0, 3.0},
		{3.0, 6.0}}

	// Remember that left branch consists of values < split
	// right branch branch consists of values >= split
	testDataMSE (t, "test1", test1, 0, 1, 3.0, 0.0)
	testDataMSE (t, "test2", test2, 0, 1, 2.0, 0.25)
	testDataMSE (t, "test3", test3, 0, 1, 3.0, 8.0/9.0)
}

func testDataEntropy (t *testing.T, msg string, data [][]float64, feature, output, outputValueCount int, expectedSplit, expectedEntropy float64) {
	split, error,_  := continuousFeatureEntropySplit(data, feature, output, outputValueCount)
	if (split != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, split)
	}
	if math.Abs(error-expectedEntropy) > 1e-12*math.Abs(expectedEntropy) {
		t.Errorf ("%s: expected error: %v; got: %v", msg, expectedEntropy, error)
	}
}

func TestContinuousFeatureEntropySplit (t *testing.T) {
	test1 := [][]float64{
		{3.0, 3.0},
		{1.0, 1.0},
		{2.0, 2.0},
		{3.0, 3.0}}
	
	test2 := [][]float64{
		{3.0, 2.0},
		{1.0, 1.0},
		{2.0, 2.0},
		{3.0, 2.0}}
	
	// Remember that left branch consists of values < split
	// right branch branch consists of values >= split
	testDataEntropy (t, "test1", test1, 0, 1, 5, 2.0, -(2.*math.Log2(2./3.)+1.*math.Log2(1./3))/3.)
	testDataEntropy (t, "test2", test2, 0, 1, 3, 2.0, 0.0)
}
