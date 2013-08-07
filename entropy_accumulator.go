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

func (ea *EntropyAccumulator) Add(category int) {
	ea.totalCount += 1
	ea.counts[category] += 1
}

func (ea *EntropyAccumulator) Remove(category int) {
	if ea.totalCount == 0 || ea.counts[category] == 0 {
		panic(errors.New(fmt.Sprintf("More calls to Remove() than to Add() for category %v", category)))
	}
	ea.totalCount -= 1
	ea.counts[category] -= 1
}

func (ea *EntropyAccumulator) Entropy() float64 {
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


