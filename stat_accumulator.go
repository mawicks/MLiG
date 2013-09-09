package ML

import (
	"errors"
	"fmt"
	"io"
)

// A new StatAccumulator may be declared without initialization.
// The Go default initialization is correct.
type StatAccumulator struct {
	count int
	weightedCount float64
	sum float64
	sumOfSquares float64
}

func (sa *StatAccumulator) Add(x, weight float64) {
	sa.count += 1
	sa.weightedCount += weight
	sa.sum += weight*x
	sa.sumOfSquares += weight*x*x
}

func (sa *StatAccumulator) Remove(x, weight float64) {
	if sa.count <= 0 {
		panic(errors.New("More calls to Remove() than to Add()"))
	}
	sa.count -= 1
	sa.weightedCount -= weight
	sa.sum -= weight*x
	sa.sumOfSquares -= weight*x*x
}

func (sa *StatAccumulator) Count() int {
	return sa.count
}

func (sa *StatAccumulator) Metric() float64 {
	result := 0.0
	if sa.count > 0 {
		result = (sa.sumOfSquares - sa.sum*sa.sum/sa.weightedCount)/sa.weightedCount
	}
	if result < 0.0 {
		panic(errors.New("Error is negative"))
	}
	return result
}

func (sa *StatAccumulator) Estimate() float64 {
	result := 0.0
	if sa.count > 0 {
		result = sa.sum/sa.weightedCount
	}
	return result
}

func (sa *StatAccumulator) Clear() {
	sa.count = 0
	sa.weightedCount = 0.0
	sa.sum = 0.0
	sa.sumOfSquares = 0.0
}

func (sa *StatAccumulator) Dump(w io.Writer, indent int) {
	fmt.Fprintf (w, "%*scount: %d, weightedCount: %g, sum(x): %g, sum(x^2): %g\n", indent, "", sa.count, sa.weightedCount, sa.sum, sa.sumOfSquares)
}

