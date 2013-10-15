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
	sum float64
	sumOfSquares float64
}

func StatAccumulatorFactory() func() CVAccumulator {
	return func () CVAccumulator {
		return &StatAccumulator{}
	}
}

func (sa *StatAccumulator) Add(x float64) {
	sa.count += 1
	sa.sum += x
	sa.sumOfSquares += x*x
}

func (sa *StatAccumulator) Remove(x float64) {
	if sa.count == 0 {
		panic(errors.New("More calls to Remove() than to Add()"))
	}
	sa.count -= 1
	sa.sum -= x
	sa.sumOfSquares -= x*x
}

func (sa *StatAccumulator) Count() int {
	return sa.count
}

func (sa *StatAccumulator) Metric() float64 {
	n := float64(sa.count)
	result := 0.0
	if sa.count > 0 {
		result = (sa.sumOfSquares - sa.sum*sa.sum/n)/n
	}
	if result < 0.0 {
		panic(errors.New("Error is negative"))
	}
	return result
}

func (sa *StatAccumulator) Estimate() float64 {
	result := 0.0
	if sa.count > 0 {
		result = sa.sum/float64(sa.count)
	}
	return result
}

func (sa *StatAccumulator) Clear() {
	sa.count = 0
	sa.sum = 0.0
	sa.sumOfSquares = 0.0
}

func (sa *StatAccumulator) Dump(w io.Writer, indent int) {
	fmt.Fprintf (w, "%*scount: %d, sum(x): %g, sum(x^2): %g\n", indent, "", sa.sum, sa.sumOfSquares)
}

