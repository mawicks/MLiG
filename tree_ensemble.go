package ML

import ( 
	"math"
)

type TreeEnsemble struct {
	errorAccumulator CVAccumulator
	trees []*treeNode
}

func NewTreeEnsemble(errorAccumulator CVAccumulator) *TreeEnsemble {
	return &TreeEnsemble{
		errorAccumulator: errorAccumulator,
		trees: make([]*treeNode,0,1000)}
}

func (te *TreeEnsemble) AddTree (data[]*Data, featuresToTest int, continuousFeatureSplit func ([]*Data, int) SplitInfo) {
	trainSize := 2*len(data)/3
	// Shuffle data and take first "trainSize" samples as the "bag" or training set.
	ShuffleData(data)
	trainSet := data[0:trainSize]
	newTree := NewTreeNode(math.MaxFloat64)
	newTree.Grow(trainSet, featuresToTest, continuousFeatureSplit)
	te.trees = append(te.trees, newTree)
	
	// Use remaining samples as the "out-of-bag" test set.
	testSet := data[trainSize:]
	for _,d := range testSet {
		prediction := newTree.Classify(d.continuousFeatures)
		d.oobAccumulator.Add (prediction)
	}
}

func (te *TreeEnsemble) Error (data[]*Data) float64 {
	te.errorAccumulator.Clear()
	for _,d := range data {
		if d.oobAccumulator.Count() != 0 {
			te.errorAccumulator.Add(d.output - d.oobAccumulator.Estimate())
		}
	}
	return te.errorAccumulator.Metric()
}
