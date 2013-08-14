package ML

type ErrorAccumulator interface {
	Add (float64)
	Count() int
	Clear()
	Estimate() float64
}

type CVAccumulator interface {
	ErrorAccumulator
	Remove (float64)
	Metric() float64
}

