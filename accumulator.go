package ML

import (
	"io"
)

type ErrorAccumulator interface {
	Add (data, weight float64)
	// Count() returns the number of elements added (not the weighted count)
	Count() int
	WeightedCount() float64
	Clear()
	Estimate() float64
	Dump(io.Writer, int)
}

type CVAccumulator interface {
	ErrorAccumulator
	Remove (data, weight float64)
	Metric() float64
}

