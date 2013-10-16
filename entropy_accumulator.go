package ML

import (
	"errors"
	"fmt"
	"io"
	"math"
)

type EntropyAccumulator struct {
	counts []int
	totalCount int
}

func (ea EntropyAccumulator) String() string {
	s := fmt.Sprintf ("entropyaccumulator[%d]: ", ea.totalCount)
	for i,c := range ea.counts {
		s = s + fmt.Sprintf(" %d:%d", i, c)
	}
	return s
}

func NewEntropyAccumulator(categoryValueCount int) *EntropyAccumulator {
	return &EntropyAccumulator{
		counts: make([]int, categoryValueCount),
		totalCount: 0}
}

func EntropyAccumulatorFactory (categoryValueCount int) func() CVAccumulator {
	return func() CVAccumulator {
		return NewEntropyAccumulator (categoryValueCount)
	}
}

func (ea *EntropyAccumulator) Clone() ErrorAccumulator {
	counts := make([]int, len(ea.counts))
	copy(counts,ea.counts)
	return &EntropyAccumulator{
		counts: counts,
		totalCount: ea.totalCount}
}

func (ea *EntropyAccumulator) Add(category float64) {
	if int(category) >= len(ea.counts) || (int(category) < 0) {
		panic (fmt.Sprintf ("Attempt to add to category %g but only %d categories", category, len(ea.counts)))
	}
	ea.totalCount += 1
	ea.counts[int(category)] += 1
}

func (ea *EntropyAccumulator) Remove(category float64) {
	if ea.totalCount == 0 || ea.counts[int(category)] == 0 {
		panic(errors.New(fmt.Sprintf("More calls to Remove() than to Add() for category %v", int(category))))
	}
	ea.totalCount -= 1
	ea.counts[int(category)] -= 1
}

func (ea *EntropyAccumulator) Count() int {
	return ea.totalCount
}

func (ea *EntropyAccumulator) Estimate() float64 {
	maxCount := 0
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
		if count != 0 {
			p := float64(count)/float64(ea.totalCount)
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

func (ea *EntropyAccumulator) Clear() {
	for i,_ := range ea.counts {
		ea.counts[i] = 0
	}
	ea.totalCount = 0
}

func (ea *EntropyAccumulator) Dump(w io.Writer, indent int) {
	fmt.Fprintf (w, "%*scount: %d ", indent, "", ea.totalCount)
	for i,c := range ea.counts {
		fmt.Fprintf (w, "  %d:%d", i, c)
	}
	fmt.Fprintf(w, "\n")
}
