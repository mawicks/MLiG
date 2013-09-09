package ML

import (
	"fmt"
	"io"
)

type errorAccumulator struct {
	totalCount int
	weightedTotalCount float64
	errorCount float64
}

func (ea *errorAccumulator) Add(error, weight float64) {
	ea.totalCount += 1
	ea.weightedTotalCount += weight
	if error != 0.0 {
		ea.errorCount += weight
	}
}

func (ea *errorAccumulator) Count() int {
	return ea.totalCount
}

func (ea *errorAccumulator) Estimate() float64 {
	if ea.totalCount > 0 {
		return ea.errorCount/ea.weightedTotalCount
	}
	return 0.0
}

func (ea *errorAccumulator) Clear() {
	ea.totalCount = 0
	ea.weightedTotalCount = 0.0
	ea.errorCount = 0.0
}

func (ea *errorAccumulator) Dump(w io.Writer, indent int) {
	fmt.Fprintf (w, "%*scount: %d, weightedCount: %g, errorCount: %g\n", indent, "", ea.totalCount, ea.weightedTotalCount, ea.errorCount)
}










