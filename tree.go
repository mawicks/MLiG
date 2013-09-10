package ML

import (
	"fmt"
	"io"
	"math"
	"sort"
	"math/rand"
)

// featureComponent structure is used for random subspace features
// where the data is partitioned on a random linear combination of
// elements of the feature vector.  
type featureComponent struct {
	index int
	weight float64 }


const (
	CATEGORICAL FeatureType =  iota
	CONTINUOUS
)

type SplitInfo struct {
	splitValue float64
	leftSplitMetric, rightSplitMetric float64
	leftEstimate, rightEstimate float64
	compositeSplitMetric float64
	leftSplitSize, rightSplitSize int
}


// Split the data with a continuously valued output variable along the
// continuously valued feature axis.  Return the feature value for the
// split, the left and right partition metric after the split, and
// the size of the left split.  The returned size will be zero if the
// error cannot be reduced.
func continuousFeatureSplit (data []*Data, seed int32, left, right CVAccumulator) (splitInfo SplitInfo) {
	left.Clear()
	right.Clear()
	
	s := sortableData{data, seed}
	sort.Sort(s)

	for _,row := range data {
		right.Add(row.output, row.weight)
	}

	rightMetric := right.Metric()

	splitInfo = SplitInfo {
		// Initialize splitValue with the *output* estimate
		splitValue: 0.0,
		leftEstimate: left.Estimate(),
		rightEstimate: right.Estimate(),
		leftSplitMetric: 0,
		rightSplitMetric: rightMetric,
		compositeSplitMetric: rightMetric,
		leftSplitSize: 0,
		rightSplitSize: len(data) }
	
	previousSplitCandidate := - math.MaxFloat64

	for i,row := range data {
		fv := row.featureSelector(seed)
		if (i != 0 && fv != previousSplitCandidate) {
			leftMetric := left.Metric()
			rightMetric := right.Metric()
			leftCount := left.Count()
			rightCount := right.Count()
			error := (left.WeightedCount()*leftMetric + right.WeightedCount()*rightMetric)/(left.WeightedCount()+right.WeightedCount())
			if error < splitInfo.compositeSplitMetric && leftCount >0 && rightCount > 0 {
				splitInfo = SplitInfo {
					splitValue: fv,
					leftEstimate: left.Estimate(),
					rightEstimate: right.Estimate(),
					leftSplitMetric: leftMetric,
					rightSplitMetric: rightMetric,
					compositeSplitMetric: error,
					leftSplitSize: leftCount,
					rightSplitSize: rightCount }
			}
		}
		left.Add(row.output, row.weight)
		right.Remove(row.output, row.weight)

		previousSplitCandidate = fv
	}
	return
}

// Split the data with a continuously valued output variable along the
// continuously valued feature axis.  Return the feature value for the
// split, the left and right mean-squared error after the split, and
// the size of the left split.  The returned size will be zero if the
// error cannot be reduced.
func continuousFeatureMSESplit (data[] *Data, seed int32) SplitInfo {
	var (
		left, right StatAccumulator
	)
	return continuousFeatureSplit (data, seed, &left, &right)
}

// Split the data with a categorical output variable along the feature
// axis.  Return the feature value for the split, the entropies of the
// left and right splits, the size of the left split.  The returned
// size will be zero if the entropy cannot be reduced.
func continuousFeatureEntropySplitter (categoryRange int) func ([]*Data, int32) SplitInfo {
	return func (data []*Data, seed int32) SplitInfo {
		left := NewEntropyAccumulator(categoryRange)
		right := NewEntropyAccumulator(categoryRange)
		
		return continuousFeatureSplit (data, 
			seed, left, right)
	}
}

type Tree struct {
	root *treeNode
	featuresToTest int
	randomSubspace []featureComponent
	continuousFeatureSplit func ([]*Data, int32) SplitInfo
	errorAccumulator ErrorAccumulator
}

func NewTree (featuresToTest int, cfSplitter func ([]*Data, int32) SplitInfo) *Tree {
	return &Tree{
		root: nil,
		featuresToTest: featuresToTest,
		continuousFeatureSplit: cfSplitter,
		errorAccumulator: &errorAccumulator{}}
}

func (tree *Tree) Train(trainingSet[] *Data) {
	tree.root = NewTreeNode(math.MaxFloat64,math.MaxFloat64)
	tree.root.grow(trainingSet,
		tree.featuresToTest,
		tree.continuousFeatureSplit)
}

func (tree *Tree) Classify(featureSelector func(int32) float64) float64 {
	return tree.root.classify(featureSelector)
}

func (tree *Tree) Add(error, weight float64) {
	tree.errorAccumulator.Add(error, weight)
}

func (tree *Tree) Estimate() float64 {
	return tree.errorAccumulator.Estimate()
}

func (tree *Tree) Dump(w io.Writer) {
	tree.root.dump(w, 0, 0)
}

func (tree *Tree) Size() int {
	return tree.root.size()
}

func (tree *Tree) Depth() int {
	return tree.root.depth()
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

	// In a leaf node, featureSelector is nil and the two partition
	// pointers are nil.  In a non-leaf node, featureIndex is a
	// valid function and both the left and right partition pointers
	// are non-nil.
	seed int32

	// In a non-leaf node, value is the splitting value.  Values
	// greater than or equal to the splitting value belong to the right
	// subtree.  Others belong to the left subtree.
	splitValue float64

	// In a leaf node outputValue is the assigned label.  In a non-leaf node,
	// It's an estimate if you stop at that node.
	outputValue float64
}

func NewTreeNode (metric, defaultOutput  float64) *treeNode {
	return &treeNode{
		featureType: CONTINUOUS,
		seed: -1,
		metric: metric,
		outputValue: defaultOutput,
		// initialization of splitValue is not very important as it is not used if featureIndex < 0
		splitValue: math.MaxFloat64}
}

// splitData() splits "data" into a "left" and "right" portions based on "splitValue" and "splitFeatureIndex".
func splitData (data []*Data, splitValue float64, seed int32, left[]*Data, right[]*Data) {
	leftCount := 0
	leftSize := len(left)
	rightCount := 0
	rightSize := len(right)
	for _,dp := range data {
		if dp.featureSelector(seed) < splitValue {
			if (leftCount == leftSize) {
				fmt.Printf("leftSize=%d; rightSize=%d; leftCount=%d; rightCount=%d\n", leftSize, rightSize, leftCount, rightCount)
				fmt.Printf("splitValue=%g, feature=%g\n", splitValue, dp.featureSelector(seed))
				panic ("Split sizes are not as expected in splitData()")
			}
			left[leftCount] = dp
			leftCount += 1
		} else {
			if (rightCount == rightSize) {
				fmt.Printf("leftSize=%d; rightSize=%d; leftCount=%d; rightCount=%d\n", leftSize, rightSize, leftCount, rightCount)
				fmt.Printf("splitValue=%g, feature=%g\n", splitValue, dp.featureSelector(seed))
				panic ("Split sizes are not as expected in splitData()")
			}
			right[rightCount] = dp
			rightCount += 1
		}
	}
	if leftSize != leftCount  || rightSize != rightCount {
	}
}

// Dump() produces a visual representation of the tree on the io.Writer.  The parameter depth
// is the depth of this node in the tree.
func (tree *treeNode) dump(w io.Writer, index, depth int) {
	indent := 4*depth + 1
	if tree.seed == -1 {
		fmt.Fprintf (w, "%*c %2d - Leaf node - output: %g; metric: %g\n", indent, ' ', index, tree.outputValue, tree.metric)
	} else {
		fmt.Fprintf (w, "%*c %2d - Split on feature ??? at value %g; metric: %g\n", indent, ' ', index, tree.splitValue, tree.metric)

		fmt.Fprintf (w, "%*c %2d - Left branch (feature ??? < %g):\n", indent, ' ', index, tree.splitValue)
		tree.left.dump(w, index+1, depth+1)

		fmt.Fprintf (w, "%*c %2d - Right branch (feature ??? >= %g):\n", indent, ' ', index, tree.splitValue)
		tree.right.dump(w, index+2, depth+1)
	}
}

func randomSubspace(featureVectorSize int) []featureComponent {
	componentDim := featureVectorSize
	
	for i:=0; i<200; i++ {
		r := int(rand.Int31n(int32(featureVectorSize))) + 1
		if r < componentDim {
			componentDim = r
		}
	}

	result := make([]featureComponent,componentDim)
	for i:=0; i<componentDim; i++ {
		result[i].index = int(rand.Int31n(int32(featureVectorSize)))
		result[i].weight = rand.Float64()*2.0 - 1.0;
	}
	return result
}

// grow() grows the tree based on the test set "data."  "featureSelector" is a function
// of a feature record returning the abstract feature value.  continuousFeatureSplit
// is the splitting function (e.g.,  MSE Error or entropy).
func (tree *treeNode) grow(data []*Data, featuresToTest int, continuousFeatureSplitter func ([]*Data, int32) SplitInfo) {
	if (len(data) == 0) {
		return
	}

	var bestSplitInfo SplitInfo

	for i:= 0; i<featuresToTest; i++ {
//		candidateSeed := rand.Int31n(int32(len(data[0].continuousFeatures)))
		candidateSeed := rand.Int31()
		candidateSplitInfo := continuousFeatureSplitter(data, candidateSeed)
		if candidateSplitInfo.leftSplitSize > 0 && candidateSplitInfo.compositeSplitMetric < tree.metric {
			bestSplitInfo = candidateSplitInfo
			tree.metric = candidateSplitInfo.compositeSplitMetric
			tree.seed = candidateSeed
		}
	}

	if tree.seed != -1 {
		tree.splitValue = bestSplitInfo.splitValue
		
		leftData := make([]*Data, bestSplitInfo.leftSplitSize)
		rightData := make([]*Data, bestSplitInfo.rightSplitSize)
		
		splitData(data, tree.splitValue, tree.seed, leftData, rightData)
		
		tree.left = NewTreeNode(bestSplitInfo.leftSplitMetric,bestSplitInfo.leftEstimate)
		tree.right = NewTreeNode(bestSplitInfo.rightSplitMetric,bestSplitInfo.rightEstimate)

		tree.left.grow(leftData, featuresToTest, continuousFeatureSplitter)
		tree.right.grow(rightData, featuresToTest, continuousFeatureSplitter)
	}
}

// Classify (or predict) the passed feature vector.
func (tree *treeNode) classify(featureSelector func(int32) float64) float64 {
	// Leaf node?
	if tree.seed == -1 {
		return tree.outputValue
	} else {
		switch  {
		case featureSelector(tree.seed) < tree.splitValue:
			return tree.left.classify(featureSelector)
		default:
			return tree.right.classify(featureSelector)
		}
	}
}

func (tree *treeNode) size() int {
	if tree.seed == -1 {
		return 1
	} else {
		return tree.left.size() + tree.right.size()
	}
}

func (tree *treeNode) depth() int {
	result := 0
	if tree.seed == -1 {
		result = 1
	} else {
		result = 1 + tree.left.depth()
		rightDepth := tree.right.depth()
		if rightDepth >= result {
			result = 1 + rightDepth
		}
	}
	return result
}
