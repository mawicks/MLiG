package ML

import (
	"errors"
	"fmt"
	"math"
)

// A new EntropyAccumulator may be declared without initialization.
// The Go default initialization is correct.
type EntropyAccumulator struct {
	counts []int
	totalCount int
}

func NewEntropyAccumulator(categoryValueCount int) *EntropyAccumulator {
	return &EntropyAccumulator{
		counts: make([]int, categoryValueCount),
		totalCount: 0}
}

func (ea *EntropyAccumulator) Add(category float64) {
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


