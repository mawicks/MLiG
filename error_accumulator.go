package ML

type errorAccumulator struct {
	totalCount int
	errorCount int
}

func (ea *errorAccumulator) Add(error float64) {
	ea.totalCount += 1
	if error != 0.0 {
		ea.errorCount += 1
	}
}

func (ea *errorAccumulator) Count() int {
	return ea.totalCount
}

func (ea *errorAccumulator) Metric() float64 {
	if ea.totalCount > 0 {
		return float64(ea.errorCount)/float64(ea.totalCount)
	}
	return 0.0
}

func (ea *errorAccumulator) Clear() {
	ea.errorCount = 0
	ea.totalCount = 0
}











