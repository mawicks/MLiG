package ML

import (
	"errors"
	"fmt"
	"math"
)

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

