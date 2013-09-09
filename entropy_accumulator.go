package ML

import (
	"errors"
	"fmt"
	"io"
	"math"
)

type EntropyAccumulator struct {
	counts []float64
	totalCount int
	weightedTotalCount float64
}

func NewEntropyAccumulator(categoryValueCount int) *EntropyAccumulator {
	return &EntropyAccumulator{
		counts: make([]float64, categoryValueCount),
		totalCount: 0,
		weightedTotalCount: 0.0}
}

func (ea *EntropyAccumulator) Add(category, weight float64) {
	if int(category) >= len(ea.counts) || (int(category) < 0) {
		panic (fmt.Sprintf ("Attempt to add to category %g but only %d categories", category, len(ea.counts)))
	}
	ea.totalCount += 1
	ea.weightedTotalCount += weight
	ea.counts[int(category)] += weight
}

func (ea *EntropyAccumulator) Remove(category, weight float64) {
	if ea.totalCount == 0 || ea.counts[int(category)] == 0 {
		panic(errors.New(fmt.Sprintf("More calls to Remove() than to Add() for category %v", int(category))))
	}
	ea.totalCount -= 1
	ea.weightedTotalCount -= weight
	ea.counts[int(category)] -= weight
}

func (ea *EntropyAccumulator) Count() int {
	return ea.totalCount
}

func (ea *EntropyAccumulator) Estimate() float64 {
	maxCount := 0.0
	result := 0.0
	for i,count := range ea.counts {
		if count > maxCount {
			maxCount = count
			result = float64(i)
		}
	}
	return result
}

func (ea *EntropyAccumulator) Metric() float64 {
	entropy := 0.0
	for _,count := range ea.counts {
		if count>0.0 {
			p := count/ea.weightedTotalCount
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

func (ea *EntropyAccumulator) Clear() {
	for i,_ := range ea.counts {
		ea.counts[i] = 0.0
	}
	ea.totalCount = 0
	ea.weightedTotalCount = 0.0
}

func (ea *EntropyAccumulator) Dump(w io.Writer, indent int) {
	fmt.Fprintf (w, "%*scount: %d ", indent, "", ea.totalCount)
	fmt.Fprintf (w, "%*sweighted count: %g ", indent, "", ea.weightedTotalCount)
	for i,c := range ea.counts {
		fmt.Fprintf (w, "  %d:%g", i, c)
	}
	fmt.Fprintf(w, "\n")
}
