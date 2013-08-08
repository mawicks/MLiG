package ML

type CVAccumulator interface {
	Count() int
	Clear()
	Add (float64)
	Remove (float64)
	Metric() float64 }
