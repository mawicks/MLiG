package ML

import ( 
	"math"
)

type TreeEnsemble struct {
	errorAccumulator ErrorAccumulator
	trees []*treeNode
}

func NewTreeEnsemble() *TreeEnsemble {
	return &TreeEnsemble{
		errorAccumulator: &errorAccumulator{},
		trees: make([]*treeNode,0,1000)}
}

func (te *TreeEnsemble) AddTree (data[]*Data, featuresToTest int, continuousFeatureSplit func ([]*Data, int) SplitInfo) {
	trainSize := 2*len(data)/3

	// Shuffle data and take first "trainSize" samples as the bag or training set.
	ShuffleData(data)
	trainSet := data[0:trainSize]
	newTree := NewTreeNode(math.MaxFloat64)
	newTree.Grow(trainSet, featuresToTest, continuousFeatureSplit)
	te.trees = append(te.trees, newTree)
	
	// Use remaining samples as the "out-of-bag" test set.  Each
	// tree gets its own out-of-bag test set.  All of the
	// classifications for all trees are accumulated within the
	// test record's oobAccumulator.  The ensemble classification
	// (over all trees used to classify the record, which is not
	// all trees) may be retrieved by oobAccumulator.Estimate().
	testSet := data[trainSize:]
	for _,d := range testSet {
		prediction := newTree.Classify(d.continuousFeatures)
		d.oobAccumulator.Add (prediction)
	}
}

func (te *TreeEnsemble) Error (data[]*Data) float64 {
	te.errorAccumulator.Clear()
	for _,d := range data {
		// Only use records that were classified by at least one tree.
		if d.oobAccumulator.Count() != 0 {
			te.errorAccumulator.Add(d.output - d.oobAccumulator.Estimate())
		}
	}
	return te.errorAccumulator.Estimate()
}
