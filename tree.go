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
	// node splitValue is the assigned label.
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

type SplitInfo struct {
	splitValue, leftSplitMetric, rightSplitMetric float64
	leftSplitSize, rightSplitSize int
}


// Split the data with a continuously valued output variable along the
// continuously valued feature axis.  Return the feature value for the
// split, the left and right partition metric after the split, and
// the size of the left split.  The returned size will be zero if the
// error cannot be reduced.
func continuousFeatureSplit (data [][]float64, feature int, output int, left, right CVAccumulator) (splitInfo SplitInfo) {

	s := sortableData{data, feature}
	sort.Sort(s)

	for _,row := range data {
		right.Add(row[output])
	}

	bestMetric := right.Metric()
	splitInfo = SplitInfo {
		splitValue: - math.MaxFloat64,
		leftSplitMetric: bestMetric,
		rightSplitMetric: bestMetric,
		leftSplitSize: 0,
		rightSplitSize: len(data) }
	
	previousSplitCandidate := - math.MaxFloat64

	for i,row := range data {
		if (i != 0 && row[feature] != previousSplitCandidate) {
			leftMetric := left.Metric()
			rightMetric := right.Metric()
			error := math.Max(leftMetric,rightMetric)
			if error < bestMetric {
				bestMetric = error
				splitInfo = SplitInfo {
					splitValue: row[feature],
					leftSplitMetric: leftMetric,
					rightSplitMetric: rightMetric,
					leftSplitSize: left.Count(),
					rightSplitSize: right.Count() }
			}
		}
		left.Add(row[output])
		right.Remove(row[output])

		previousSplitCandidate = row[feature]
	}
	return
}

// Split the data with a continuously valued output variable along the
// continuously valued feature axis.  Return the feature value for the
// split, the left and right mean-squared error after the split, and
// the size of the left split.  The returned size will be zero if the
// error cannot be reduced.
func continuousFeatureMSESplit (data [][]float64, feature int, output int) (splitInfo SplitInfo) {
	var (
		left, right StatAccumulator
	)
	return continuousFeatureSplit (data, feature, output, &left, &right)
}

// Split the data with a categorical output variable along the feature
// axis.  Return the feature value for the split, the entropies of the
// left and right splits, the size of the left split.  The returned
// size will be zero if the entropy cannot be reduced.
func continuousFeatureEntropySplit (data [][]float64, feature int, output int, categoryRange int) (splitInfo SplitInfo) {
	left := NewEntropyAccumulator(categoryRange)
	right := NewEntropyAccumulator(categoryRange)

	return continuousFeatureSplit (data, feature, output, left, right)
}
