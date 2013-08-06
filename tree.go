package ML

import (
	"math"
	"sort"
)

type Feature interface {
	// compareTo() is only required to work for features of the same type
	// e.g., continuous or categorical.
	compareTo (f Feature) int
}

type FeatureType int

const (
	CATEGORICAL FeatureType =  iota
	CONTINUOUS
)


type treeNode struct {
	left *treeNode
	right *treeNode
	
	featureType FeatureType // Either CATEGORICAL OR CONTINUOUS

	// In a leaf node, featureIndex is < 0, and the two partition
	// pointers are nil.  In a non-leaf node, featureIndex is a
	// valid index, and both the left and right partition pointers
	// are non-nil.
	featureIndex int

	// In a non-leaf node, value is the splitting value.  Values
	// greater than or equal to the splitting value belong to the right
	// subtree.  Others belong to the left subtree.  In a leaf
	// node value is the assigned label.
	splitValue float64
}

// Classify the passed feature vector.
func (tree *treeNode) classify(feature []float64) float64 {
	// Leaf node?
	if tree.featureIndex < 0 {
		return tree.splitValue
	} else {
		switch  {
		case feature[tree.featureIndex] <= tree.splitValue:
			return tree.left.classify(feature)
		default:
			return tree.right.classify(feature)
		}
	}
}

type sortableData struct {
	data [][]float64
	feature int
}

func (s sortableData) Len() int {
	return len(s.data)
}

func (s sortableData) Less(i, j int) bool {
	return s.data[i][s.feature] < s.data[j][s.feature]
}

func (s sortableData) Swap(i, j int) {
	s.data[i],s.data[j] = s.data[j],s.data[i]
}

func splitContinuousFeature (data [][]float64, labels []float64, feature int, output int) float64 {
	var (
		leftStats, rightStats StatAccumulator
	)

	s := sortableData{data, feature}
	sort.Sort(s)

	for _,row := range data {
		rightStats.Add(row[output])
	}

	bestError := math.MaxFloat64
	bestSplitValue := - math.MaxFloat64
	previousCandidate := - math.MaxFloat64

	for _,row := range data {
		if (row[feature] != previousCandidate) {
			error := math.Max(leftStats.Variance(), rightStats.Variance())
			if error < bestError {
				bestError = error
				bestSplitValue = row[feature]
			}
		}
		leftStats.Add(row[output])
		rightStats.Remove(row[output])

		previousCandidate = row[feature]
	}
	
	return bestSplitValue
}


















