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
	compositeSplitMetric float64
	splitValue float64
	left, right CVAccumulator
}

func (si *SplitInfo) String() string {
	s := fmt.Sprintf ("{ compositeSplitMetric: %g splitValue %g\n", si.compositeSplitMetric, si.splitValue)
	s = s + fmt.Sprintf ("   left: [%v]\n   right: [%v]\n}", si.left, si.right)
	return s
}

// Split the data with a continuously valued output variable along the
// continuously valued feature axis.  Return the feature value for the
// split, the left and right partition metric after the split, and
// the size of the left split.  The returned size will be zero if the
// error cannot be reduced.
func continuousFeatureSplit (data []*Data, seed int32, accumulatorFactory func() CVAccumulator) (splitInfo SplitInfo) {
	left := accumulatorFactory()
	right := accumulatorFactory()

	left.Clear()
	right.Clear()
	
	s := sortableData{data, seed}
	sort.Sort(s)

	for _,row := range data {
		right.Add(row.output)
	}

	rightMetric := right.Metric()

	splitInfo = SplitInfo {
		// Initialize splitValue with the *output* estimate
		splitValue: 0.0,
		compositeSplitMetric: rightMetric,
		left: left.Clone().(CVAccumulator),
		right: right.Clone().(CVAccumulator)}
	
	previousSplitCandidate := - math.MaxFloat64

	for i,row := range data {
		fv := row.featureSelector(seed)
		if (i != 0 && fv != previousSplitCandidate) {
			if (fv <= previousSplitCandidate) {
				fmt.Printf ("Sanity check: fv=%g <= previousSplitCandidate=%g\n", fv, previousSplitCandidate)
			}
			leftMetric := left.Metric()
			rightMetric := right.Metric()
			leftCount := left.Count()
			rightCount := right.Count()
			error := (float64(leftCount)*leftMetric + float64(rightCount)*rightMetric)/float64(leftCount+rightCount)
			if error < splitInfo.compositeSplitMetric && leftCount > 0 && rightCount > 0 {
				sv := 0.5*(previousSplitCandidate+fv)
				// When previousSplitCandidate and fv are extremely close, split value can exactly match
				// previousSplitCandidate.  In that case, use the upper value (fv)
				if sv == previousSplitCandidate {
					sv = fv
				}
				splitInfo.left = left.Clone().(CVAccumulator)
				splitInfo.right = right.Clone().(CVAccumulator)
				splitInfo.splitValue = sv
				splitInfo.compositeSplitMetric = error
			}
		}
		left.Add(row.output)
		right.Remove(row.output)

		previousSplitCandidate = fv
	}

	leftCount := 0
	rightCount := 0
	maxBeforeSplit := -math.MaxFloat64
	minAfterSplit := math.MaxFloat64
	if left.Count() != 0 && right.Count() != 0 {
		for _,row := range data {
			fv := row.featureSelector(seed)
			if (fv < splitInfo.splitValue) {
				leftCount += 1
				if fv > maxBeforeSplit {
					maxBeforeSplit = fv
				}
			} else {
				rightCount += 1
				if fv < minAfterSplit {
					minAfterSplit = fv
				}
			}
		}
		if leftCount != left.Count() || rightCount != right.Count() {
			fmt.Printf ("leftCount=%d,leftSplitSize=%d,rightCount=%d,rightSplitSize=%d\nmaxbeforeSplit=%g,splitValue=%g,minAfterSplit=%g\n",
				leftCount, left.Count(),rightCount,right.Count(),maxBeforeSplit,splitInfo.splitValue,minAfterSplit)
		}
	}
	return
}

type Tree struct {
	root *treeNode
	maxDepth int
	minLeafSize int
	featuresToTry int
	randomSubspace []featureComponent
	accumulatorFactory func() CVAccumulator
	errorAccumulator ErrorAccumulator
}

func NewTree (accumulatorFactory func() CVAccumulator) *Tree {
	return &Tree{
		root: nil,
		maxDepth: int(math.MaxInt32),
		minLeafSize: 1,
		featuresToTry: 1,
		accumulatorFactory: accumulatorFactory,
		errorAccumulator: &errorAccumulator{}}
}

func (tree *Tree) SetMaxDepth(depth int) {
	tree.maxDepth = depth
}
 
func (tree *Tree) SetMinLeafSize(size int) {
	tree.minLeafSize = size
}

func (tree *Tree) SetFeaturesToTry(n int) {
	tree.featuresToTry = n
}

func (tree *Tree) Train(trainingSet[] *Data) {
	statistics := tree.accumulatorFactory()
	for _,d := range trainingSet {
		statistics.Add(d.output)
	}
	tree.root = NewTreeNode(statistics)
	tree.root.grow(trainingSet,
		tree.maxDepth,
		tree.minLeafSize,
		tree.featuresToTry,
		tree.accumulatorFactory)
}

func (tree *Tree) Classify(featureSelector func(int32) float64) CVAccumulator {
	return tree.root.classify(featureSelector)
}

func (tree *Tree) Add(error float64) {
	tree.errorAccumulator.Add(error)
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

func (tree *Tree) Leaves() int {
	return tree.root.leaves()
}

type treeNode struct {
	featureType FeatureType // Either CATEGORICAL OR CONTINUOUS

	// statistics contains the accumulated statistics for the
	// training data used to generate this node.
	statistics CVAccumulator

	// In a leaf node, "seed" is -1, and the "left" and "right"
	// pointers are nil.  In a non-leaf node, "seed" is the seed
	// that defines the random feature used to split this node.
	seed int32
	left *treeNode
	right *treeNode
	
	// In a non-leaf node, splitValue is the splitting value.
	// Values greater than or equal to the splitting value belong
	// to the right subtree.  Others belong to the left subtree.
	splitValue float64
}

func NewTreeNode (statistics CVAccumulator) *treeNode {
	return &treeNode{
		featureType: CONTINUOUS,
		seed: -1,
		statistics: statistics,
		// initialization of splitValue is not very important as it is not used if seed < 0
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
		fmt.Fprintf (w, "%*c %2d - Leaf node - output: %g; metric: %g\n", indent, ' ', index, tree.statistics.Estimate(), tree.statistics.Metric())
	} else {
		fmt.Fprintf (w, "%*c %2d - Split on feature ??? at value %g; metric: %g\n", indent, ' ', index, tree.splitValue, tree.statistics.Metric())

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
func (tree *treeNode) grow(data []*Data, maxDepth, minLeafSize, featuresToTry int, accumulatorFactory func() CVAccumulator) {
	if (len(data) == 0) {
		return
	}

	if maxDepth == 0 {
		return
	}

	var bestSplitInfo SplitInfo
	bestMetric := tree.statistics.Metric()

	for i:= 0; i<featuresToTry; i++ {
//		candidateSeed := rand.Int31n(int32(len(data[0].continuousFeatures)))
		candidateSeed := rand.Int31()
		candidateSplitInfo := continuousFeatureSplit(data, candidateSeed, accumulatorFactory)

		if candidateSplitInfo.left.Count() >= minLeafSize && 
			candidateSplitInfo.right.Count() >= minLeafSize &&
			candidateSplitInfo.compositeSplitMetric < bestMetric {
			bestSplitInfo = candidateSplitInfo
			tree.seed = candidateSeed
			bestMetric = candidateSplitInfo.compositeSplitMetric
		}
	}
	
	if tree.seed != -1 {
		tree.splitValue = bestSplitInfo.splitValue
		
		leftData := make([]*Data, bestSplitInfo.left.Count())
		rightData := make([]*Data, bestSplitInfo.right.Count())
		
		splitData(data, tree.splitValue, tree.seed, leftData, rightData)
		
		tree.left = NewTreeNode(bestSplitInfo.left)
		tree.right = NewTreeNode(bestSplitInfo.right)

		tree.left.grow(leftData, maxDepth-1, minLeafSize, featuresToTry, accumulatorFactory)
		tree.right.grow(rightData, maxDepth-1, minLeafSize, featuresToTry, accumulatorFactory)
	}
}

// Classify (or predict) the passed feature vector.
func (tree *treeNode) classify(featureSelector func(int32) float64) CVAccumulator {
	// Leaf node?
	if tree.seed == -1 {
		return tree.statistics
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
		return 1 + tree.left.size() + tree.right.size()
	}
}

func (tree *treeNode) depth() int {
	result := 0
	if tree.seed != -1 {
		result = 1 + tree.left.depth()
		rightDepth := 1 + tree.right.depth()
		if rightDepth > result {
			result = rightDepth
		}
	}
	return result
}

func (tree *treeNode) leaves() int {
	result := 0
	if tree.seed == -1 {
		result = 1
	} else {
		result = tree.left.leaves() + tree.right.leaves()
	}
	return result
}
