package ML

type Classifier interface {
	Classify (func (int32) float64) CVAccumulator
	Train(trainset []*Data)

	// Accumulate statistics for this classifier
	Add(error float64)
	// Return an error estimate
	Estimate() float64
}











