package ML

import (
	"math"
	"testing"
)

func testDataMSE (t *testing.T, msg string, data [][]float64, feature, output int, expectedSplit, expectedLeftError, expectedRightError float64, size int) {
	split, leftError, rightError, leftSize := continuousFeatureMSESplit(data, feature, output)
	if (split != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, split)
	}
	if math.Abs(leftError-expectedLeftError) > 1e-12*math.Abs(expectedLeftError) {
		t.Errorf ("%s: expected left error: %v; got: %v", msg, expectedLeftError, leftError)
	}
	if math.Abs(rightError-expectedRightError) > 1e-12*math.Abs(expectedRightError) {
		t.Errorf ("%s: expected right error: %v; got: %v", msg, expectedRightError, rightError)
	}
	if leftSize != size {
		t.Errorf ("%s: expected left split size: %v; got: %v", msg, size, leftSize)
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
	testDataMSE (t, "test1", test1, 0, 1, 3.0, 0.0, 0.0, 2)
	testDataMSE (t, "test2", test2, 0, 1, 2.0, 0.0, 0.25, 1)
	testDataMSE (t, "test3", test3, 0, 1, 3.0, 8.0/9.0, 0.0, 3)
}

func testDataEntropy (t *testing.T, msg string, data [][]float64, feature, output, outputValueCount int, expectedSplit, expectedLeftEntropy, expectedRightEntropy float64, size int) {
	split, leftEntropy, rightEntropy, leftSize := continuousFeatureEntropySplit(data, feature, output, outputValueCount)
	if (split != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, split)
	}
	if math.Abs(leftEntropy-expectedLeftEntropy) > 1e-12*math.Abs(expectedLeftEntropy) {
		t.Errorf ("%s: expected left entropy: %v; got: %v", msg, expectedLeftEntropy, leftEntropy)
	}
	if math.Abs(rightEntropy-expectedRightEntropy) > 1e-12*math.Abs(expectedRightEntropy) {
		t.Errorf ("%s: expected right entropy: %v; got: %v", msg, expectedRightEntropy, rightEntropy)
	}
	if leftSize != size {
		t.Errorf ("%s: expected left split size: %v; got: %v", msg, size, leftSize)
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
	testDataEntropy (t, "test1", test1, 0, 1, 5, 2.0, 0.0, -(2.*math.Log2(2./3.)+1.*math.Log2(1./3))/3., 1)
	testDataEntropy (t, "test2", test2, 0, 1, 3, 2.0, 0.0, 0.0, 1)
}
