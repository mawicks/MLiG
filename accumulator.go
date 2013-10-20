package ML

import (
	"io"
)

type ErrorAccumulator interface {
	Clear()
	Add (float64)
	Count() int
	Estimate() float64
	Clone() ErrorAccumulator
	Dump(io.Writer, int)
}

type CVAccumulator interface {
	ErrorAccumulator
	Remove (float64)
	Metric() float64
}

type WeightedErrorAccumulator interface {
	Clear()
	Add (category, weight float64)
	Count() int
	WeightedCount() float64
	Estimate() float64
	Clone() WeightedErrorAccumulator
	Dump(io.Writer, int)
}

type WeightedCVAccumulator interface {
	WeightedErrorAccumulator
	Remove (cateogry, weight float64)
	Metric() float64
}
