package ML

import (
	"io"
)

type ErrorAccumulator interface {
	Add (float64)
	Count() int
	Clear()
	Estimate() float64
	Dump(io.Writer, int)
}

type CVAccumulator interface {
	ErrorAccumulator
	Remove (float64)
	Metric() float64
}

