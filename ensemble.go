package ML

import (
//	"fmt"
//	"os"
)

type Ensemble struct {
	errorAccumulator ErrorAccumulator
	classifiers []Classifier
}

func NewEnsemble() *Ensemble {
	return &Ensemble{
		errorAccumulator: &errorAccumulator{},
		classifiers: make([]Classifier,0,1000)}
}

func TrainBag (data[]*Data, classifier Classifier) {
	trainSize := 2*len(data)/3

	// Shuffle data and take first "trainSize" samples as the bag or training set.
	ShuffleData(data)
	trainSet := data[0:trainSize]
	classifier.Train (trainSet)
	
	// Use remaining samples as the "out-of-bag" test set.  Each
	// classifier gets its own out-of-bag test set.  All of the
	// classifications for all classifiers are accumulated within
	// the test record's oobAccumulator.  The ensemble
	// classification (over all classifiers used to classify the
	// record, which is not all classifiers) may be retrieved by
	// oobAccumulator.Estimate().
	testSet := data[trainSize:]
	for _,d := range testSet {
		prediction := classifier.Classify(d.featureSelector).Estimate()
		d.oobAccumulator.Add (prediction)
		classifier.Add (d.output - prediction)
	}
}

func (te *Ensemble) AddClassifier (newClassifier Classifier) {
	te.classifiers = append(te.classifiers, newClassifier)
}

func (te *Ensemble) Error (data[]*Data) float64 {
	te.errorAccumulator.Clear()
	for _,d := range data {
		// Only use records that were classified by at least one classifier
		if d.oobAccumulator.Count() != 0 {
			estimate := d.oobAccumulator.Estimate()
			te.errorAccumulator.Add(d.output - estimate)
			if d.output != estimate {
//				fmt.Fprintf(os.Stdout, "%g Misclassified as %g:\n", d.output, estimate)
//				d.oobAccumulator.Dump(os.Stdout, 5)
			}
		}
	}
	return te.errorAccumulator.Estimate()
}
