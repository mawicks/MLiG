package ML

type CVAccumulator interface {
	Add (float64)
	Count() int
	Clear()
	Metric() float64
	Remove (float64)
	Estimate() float64
}

