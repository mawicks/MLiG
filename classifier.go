package ML

type Classifier interface {
	Classify (func (int32) float64) float64
	Train(trainset []*Data)

	// Accumulate statistics for this classifier
	Add(error, weight float64)
	// Return an error estimate
	Estimate() float64
}

