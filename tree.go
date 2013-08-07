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

// Split the data with a continuously valued output variable along the
// continuously valued feature axis.  Return the feature value for the
// split, the mean-squared error after the split, and ok if an error
// reducing split exists.
func continuousFeatureMSESplit (data [][]float64, feature int, output int) (splitValue, mse float64, ok bool) {
	var (
		leftStats, rightStats StatAccumulator
	)

	s := sortableData{data, feature}
	sort.Sort(s)

	for _,row := range data {
		rightStats.Add(row[output])
	}

	bestError := rightStats.Variance()
	splitValue = - math.MaxFloat64
	previousSplitCandidate := - math.MaxFloat64

	for i,row := range data {
		if (i != 0 && row[feature] != previousSplitCandidate) {
			error := math.Max(rightStats.Variance(), leftStats.Variance())
			if error < bestError {
				bestError = error
				splitValue = row[feature]
				ok = true
			}
		}
		leftStats.Add(row[output])
		rightStats.Remove(row[output])

		previousSplitCandidate = row[feature]
	}
	return splitValue, bestError, ok
}

// Split the data with a categorical output variable along the feature
// axis.  Return the feature value for the split, the entropy after
// the split, and ok if an entropy reducing split exists.
func continuousFeatureEntropySplit (data [][]float64, feature int, output int, categoryRange int) (splitValue, entropy float64, ok bool) {
	leftEntropy := NewEntropyAccumulator(categoryRange)
	rightEntropy := NewEntropyAccumulator(categoryRange)

	s := sortableData{data, feature}
	sort.Sort(s)

	for _,row := range data {
		rightEntropy.Add(int(row[output]))
	}

	bestEntropy := rightEntropy.Entropy();
	splitValue = - math.MaxFloat64
	previousSplitCandidate := - math.MaxFloat64

	for i,row := range data {
		if (i != 0 && row[feature] != previousSplitCandidate) {
			entropy := math.Max(rightEntropy.Entropy(), leftEntropy.Entropy())
			if entropy < bestEntropy {
				bestEntropy = entropy
				splitValue = row[feature]
				ok = true
			}
		}
		leftEntropy.Add(int(row[output]))
		rightEntropy.Remove(int(row[output]))

		previousSplitCandidate = row[feature]
	}
	return splitValue, bestEntropy, ok
}
