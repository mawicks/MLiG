package ML

type Classifier interface {
	Classify (feature []float64) float64
	Train(trainset []*Data)
}

