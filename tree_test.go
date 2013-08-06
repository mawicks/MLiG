package ML

import (
	"testing"
)

func testData (t *testing.T, msg string, data [][]float64, feature, output int, expectedSplit, expectedError float64) {
	split, error := continuousFeatureMSESplit(data, feature, output)
	if (split != expectedSplit) {
		t.Errorf ("%s: expected split: %v; got: %v", msg, expectedSplit, split)
	}
	if (error != expectedError) {
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
	testData (t, "test1", test1, 0, 1, 3.0, 0.0)
	testData (t, "test2", test2, 0, 1, 2.0, 0.25)
	testData (t, "test3", test3, 0, 1, 0.0, 0.0)
}
