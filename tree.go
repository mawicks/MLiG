package ML

import (
	"fmt"
	"io"
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
	outputCategories int // 0 means no output, 1 means continuous

	oobAccumulator ErrorAccumulator
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
	splitValue float64
	leftSplitMetric, rightSplitMetric float64
	compositeSplitMetric float64
	leftSplitSize, rightSplitSize int
}


// Split the data with a continuously valued output variable along the
// continuously valued feature axis.  Return the feature value for the
// split, the left and right partition metric after the split, and
// the size of the left split.  The returned size will be zero if the
// error cannot be reduced.
func continuousFeatureSplit (data []*Data, featureIndex int, left, right CVAccumulator) (splitInfo SplitInfo) {
	left.Clear()
	right.Clear()
	
	s := sortableData{data, featureIndex}
	sort.Sort(s)

	for _,row := range data {
		right.Add(row.output)
	}

	rightMetric := right.Metric()

	splitInfo = SplitInfo {
		// Initialize splitValue with the *output* estimate
		splitValue: right.Estimate(),
		leftSplitMetric: 0,
		rightSplitMetric: rightMetric,
		compositeSplitMetric: rightMetric,
		leftSplitSize: 0,
		rightSplitSize: len(data) }
	
	previousSplitCandidate := - math.MaxFloat64

	for i,row := range data {
		if (i != 0 && row.continuousFeatures[featureIndex] != previousSplitCandidate) {
			leftMetric := left.Metric()
			rightMetric := right.Metric()
			leftCount := left.Count()
			rightCount := right.Count()
			error := (float64(leftCount)*leftMetric + float64(rightCount)*rightMetric)/float64(leftCount+rightCount)
			if error < splitInfo.compositeSplitMetric {
				splitInfo = SplitInfo {
					splitValue: row.continuousFeatures[featureIndex],
					leftSplitMetric: leftMetric,
					rightSplitMetric: rightMetric,
					compositeSplitMetric: error,
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
func continuousFeatureMSESplit (data[] *Data, featureIndex int) SplitInfo {
	var (
		left, right StatAccumulator
	)
	return continuousFeatureSplit (data, featureIndex, &left, &right)
}

// Split the data with a categorical output variable along the feature
// axis.  Return the feature value for the split, the entropies of the
// left and right splits, the size of the left split.  The returned
// size will be zero if the entropy cannot be reduced.
func continuousFeatureEntropySplitter (categoryRange int) func ([]*Data, int) SplitInfo {
	return func (data []*Data, featureIndex int) SplitInfo {
		left := NewEntropyAccumulator(categoryRange)
		right := NewEntropyAccumulator(categoryRange)
		
		return continuousFeatureSplit (data, featureIndex, left, right)
	}
}

type treeNode struct {
	left *treeNode
	right *treeNode
	
	featureType FeatureType // Either CATEGORICAL OR CONTINUOUS

	// metric provides the performance of this node on the sample
	// data.  For a leaf node, it represents the irreducible
	// performance on the data set that could not be split.  For a
	// non-leaf node, it should be the max of the metrics of the
	// left and right decendents.
	metric float64

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

func NewTreeNode (metric float64) *treeNode {
	return &treeNode{
		featureType: CONTINUOUS,
		featureIndex: -1,
		metric: metric,
		// initialization of splitValue is not very important as it is not used if featureIndex < 0
		splitValue: math.MaxFloat64,
	}
}

// splitData() splits "data" into a "left" and "right" portions based on "splitValue" and "splitFeatureIndex".
func splitData (data []*Data, splitValue float64, splitFeatureIndex int, left[]*Data, right[]*Data) {
	leftCount := 0
	leftSize := len(left)
	rightCount := 0
	rightSize := len(right)
	for _,dp := range data {
		if dp.continuousFeatures[splitFeatureIndex] < splitValue {
			left[leftCount] = dp
			leftCount += 1
		} else {
			right[rightCount] = dp
			rightCount += 1
		}
	}
	if leftSize != leftCount  || rightSize != rightCount {
		panic ("Split sizes are not as expected in splitData()")
	}
}

// Dump() produces a visual representation of the tree on the io.Writer.  The parameter depth
// is the depth of this node in the tree.
func (tree *treeNode) Dump(w io.Writer, index, depth int) {
	indent := 4*depth + 1
	if (tree.featureIndex < 0) {
		fmt.Fprintf (w, "%*c %2d - Leaf node - output: %g; metric: %g\n", indent, ' ', index, tree.splitValue, tree.metric)
	} else {
		fmt.Fprintf (w, "%*c %2d - Split on feature %d at value %g; metric: %g\n", indent, ' ', index, tree.featureIndex, tree.splitValue, tree.metric)

		fmt.Fprintf (w, "%*c %2d - Left branch (feature %d < %g):\n", indent, ' ', index, tree.featureIndex, tree.splitValue)
		tree.left.Dump(w, index+1, depth+1)

		fmt.Fprintf (w, "%*c %2d - Right branch (feature %d >= %g):\n", indent, ' ', index, tree.featureIndex, tree.splitValue)
		tree.right.Dump(w, index+2, depth+1)
	}
}

func (tree *treeNode) Grow(data []*Data, featuresToTest int, continuousFeatureSplit func ([]*Data, int) SplitInfo) {
	if (len(data) == 0) {
		return
	}

	var bestSplitInfo SplitInfo

	for i:= 0; i<featuresToTest; i++ {
		candidateFeatureIndex := int(rand.Int31n(int32(len(data[0].continuousFeatures))))
		candidateSplitInfo := continuousFeatureSplit(data, candidateFeatureIndex)
		tree.splitValue = candidateSplitInfo.splitValue
		if candidateSplitInfo.leftSplitSize != 0 && candidateSplitInfo.compositeSplitMetric < tree.metric {
			bestSplitInfo = candidateSplitInfo
			tree.metric = candidateSplitInfo.compositeSplitMetric
			tree.featureIndex = candidateFeatureIndex
		}
	}

	if tree.featureIndex >= 0 {
		tree.splitValue = bestSplitInfo.splitValue
		
		leftData := make([]*Data, bestSplitInfo.leftSplitSize)
		rightData := make([]*Data, bestSplitInfo.rightSplitSize)
		
		splitData(data, tree.splitValue, tree.featureIndex, leftData, rightData)
		
		tree.left = NewTreeNode(bestSplitInfo.leftSplitMetric)
		tree.right = NewTreeNode(bestSplitInfo.rightSplitMetric)

		tree.left.Grow(leftData, featuresToTest, continuousFeatureSplit)
		tree.right.Grow(rightData, featuresToTest, continuousFeatureSplit)
	}
}

// Classify (or predict) the passed feature vector.
func (tree *treeNode) Classify(feature []float64) float64 {
	// Leaf node?
	if tree.featureIndex < 0 {
		return tree.splitValue
	} else {
		switch  {
		case feature[tree.featureIndex] < tree.splitValue:
			return tree.left.Classify(feature)
		default:
			return tree.right.Classify(feature)
		}
	}
}

func (tree *treeNode) Size() int {
	if tree.featureIndex < 0 {
		return 1
	} else {
		return tree.left.Size() + tree.right.Size()
	}
}

func (tree *treeNode) Depth() int {
	result := 0
	if tree.featureIndex < 0 {
		result = 1
	} else {
		result = 1 + tree.left.Depth()
		rightDepth := tree.right.Depth()
		if rightDepth >= result {
			result = 1 + rightDepth
		}
	}
	return result
}


















