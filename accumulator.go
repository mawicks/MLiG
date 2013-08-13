package ML

type ErrorAccumulator interface {
	Add (float64)
	Count() int
	Clear()
	Metric() float64
}

type CVAccumulator interface {
	ErrorAccumulator
	Remove (float64)
	Estimate() float64
}

