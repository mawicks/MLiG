package ML

import (
	"fmt"
	"testing"
)

func checkMeanAndVariance(t *testing.T, msg string, a StatAccumulator, mean, variance float64) {
	if a.Metric() != variance {
		t.Errorf ("Variance() %s is %v; expected %v", msg, a.Metric(), variance)
	}

	if a.Estimate() != mean {
		t.Errorf ("Mean() %s is %v; expected %v", msg, a.Estimate(), mean)
	}
}

func TestStatAccumulator (t *testing.T) {
	test1 := []float64 { 3.0, 6.0, 18.0}
	var a StatAccumulator

	checkMeanAndVariance(t, "after initialization", a, 0.0, 0.0)

	for _,v := range test1 {
		a.Add(v)
	}

	s := fmt.Sprintf ("on data %v", test1)

	checkMeanAndVariance(t, s, a, 9.0, 42.00)

	a.Remove(3.0)

	checkMeanAndVariance(t, s, a, 12.0, 36.0)

	a.Remove(6.0)

	checkMeanAndVariance(t, s, a, 18.0, 0.0)
}
