package ML

import (
	"math"
	"sort"
	"math/rand"
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

type Data struct {
	continuousFeatures []float64
	categoricalFeatures []int
	output float64
	outputCategories int // 0 means continuous
}

type sortableData struct {
	data []*Data
	sortFeature int
}

func (s sortableData) Len() int {
	return len(s.data)
}

func (s sortableData) Less(i, j int) bool {
	return s.data[i].continuousFeatures[s.sortFeature] < s.data[j].continuousFeatures[s.sortFeature]
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
func continuousFeatureSplit (data []*Data, featureIndex int, left, right CVAccumulator) (splitInfo SplitInfo) {

	s := sortableData{data, featureIndex}
	sort.Sort(s)

	for _,row := range data {
		right.Add(row.output)
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
		if (i != 0 && row.continuousFeatures[featureIndex] != previousSplitCandidate) {
			leftMetric := left.Metric()
			rightMetric := right.Metric()
			error := math.Max(leftMetric,rightMetric)
			if error < bestMetric {
				bestMetric = error
				splitInfo = SplitInfo {
					splitValue: row.continuousFeatures[featureIndex],
					leftSplitMetric: leftMetric,
					rightSplitMetric: rightMetric,
					leftSplitSize: left.Count(),
					rightSplitSize: right.Count() }
			}
		}
		left.Add(row.output)
		right.Remove(row.output)

		previousSplitCandidate = row.continuousFeatures[featureIndex]
	}
	return
}

// Split the data with a continuously valued output variable along the
// continuously valued feature axis.  Return the feature value for the
// split, the left and right mean-squared error after the split, and
// the size of the left split.  The returned size will be zero if the
// error cannot be reduced.
func continuousFeatureMSESplit (data []*Data, featureIndex int) (splitInfo SplitInfo) {
	var (
		left, right StatAccumulator
	)
	return continuousFeatureSplit (data, featureIndex, &left, &right)
}

// Split the data with a categorical output variable along the feature
// axis.  Return the feature value for the split, the entropies of the
// left and right splits, the size of the left split.  The returned
// size will be zero if the entropy cannot be reduced.
func continuousFeatureEntropySplit (data []*Data, featureIndex int, categoryRange int) (splitInfo SplitInfo) {
	left := NewEntropyAccumulator(categoryRange)
	right := NewEntropyAccumulator(categoryRange)

	return continuousFeatureSplit (data, featureIndex, left, right)
}

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

func splitData (data []*Data, left[]*Data, right[]*Data) {
}

func (tree *treeNode) Grow (data []*Data) {
	if (len(data) == 0) {
		return
	}
	// FIXME:
	categoryRange := 10
	featuresToTest := 5

	var bestSplitInfo SplitInfo
	bestSplitMetric := math.MaxFloat64

	for i:= 0; i<featuresToTest; i++ {
		candidateFeatureIndex := int(rand.Int31n(int32(len(data[0].continuousFeatures))))
		candidateSplitInfo := continuousFeatureEntropySplit(data, candidateFeatureIndex, categoryRange)
		candidateMetric := math.Max(candidateSplitInfo.leftSplitMetric,candidateSplitInfo.rightSplitMetric)
		if candidateSplitInfo.leftSplitSize != 0 && candidateMetric < bestSplitMetric {
			bestSplitInfo = candidateSplitInfo
			bestSplitMetric = candidateMetric
			tree.splitValue = candidateSplitInfo.splitValue
			tree.featureIndex = i
		}
	}
	if bestSplitMetric != math.MaxFloat64 {
		leftData := make([]*Data, bestSplitInfo.leftSplitSize)
		rightData := make([]*Data, bestSplitInfo.rightSplitSize)

		splitData(data, leftData, rightData)
		tree.left = &treeNode{}
		tree.left.Grow(leftData)
		tree.right = &treeNode{}
		// FIXME: split data
		tree.right.Grow(rightData)
	}
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



