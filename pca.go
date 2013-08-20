package ML

import (
	"fmt"
	"github.com/skelterjohn/go.matrix"
	"time"
)

func pcaBasis (data []*Data) (s,v *matrix.DenseMatrix) {
	if data == nil || len(data) == 0 {
		return nil,nil
	}

	featureCount := len(data[0].continuousFeatures)
	dataCount := len(data)
	
	if featureCount > dataCount {
		panic ("Number of features cannot exceed number of data points")
	}
	
	featureMeans := make([]float64, featureCount)

	for _,dv := range data {
		for i,f := range dv.continuousFeatures {
			featureMeans[i] += f
		}
	}
	for i,_ := range featureMeans {
		featureMeans[i] /= float64(dataCount)
	}

	A := matrix.Zeros(dataCount, featureCount)

	for i,dv := range data {
		for j,f := range dv.continuousFeatures {
			A.Set(i,j,f-featureMeans[j])
		}
	}
	// Perform QR() before SVD to avoid full U computation.
	// Because of its size, computation of full U is not feasible.
	// The go.matrix SVD() has no option for avoiding U
	// computation, so perform SVD on R obtained from QR (after
	// dumping lower rows) to reduce size of U.  The singular
	// values () and right singular vectors (V) of A and R are
	// the same.

	start := time.Now()
	fmt.Printf ("%s: Performing initial QR factorization...", start)
	_,R := A.QR()
	fmt.Printf ("done (%s).\n", time.Now().Sub(start))

	// Retain only square portion of R.
	SmallerA := R.GetMatrix(0, 0, featureCount, featureCount).Copy()

	// Release memory before performing another large matrix factorization..
	A = nil; R = nil

	var err error

	start = time.Now()
	fmt.Printf ("%sPerforming SVD...", time.Now())
	_,s,v,err = SmallerA.SVD()
	fmt.Printf ("done (%s).\n", time.Now().Sub(start))

	if err == nil {
		return s,v
	} else {
		panic(err)
	}
}

func pcaChangeBasis (data []*Data, S, V *matrix.DenseMatrix, significance float64) {
	if data == nil || len(data) == 0 {
		return
	}

	featureCount := len(data[0].continuousFeatures)

	if (featureCount != S.Rows() ||
		featureCount != S.Cols() || 
		featureCount != V.Rows()) {
		panic (fmt.Sprintf("Inconsistent dimension in pcaChangeBasis: features: %d S(%dx%d), V(%dx%d)",
			featureCount, S.Rows(), S.Cols(), V.Rows(), V.Cols()))
	}
	newFeatureCount := 0
	sigma0 := S.Get(0,0)

	for newFeatureCount=1; newFeatureCount<featureCount; newFeatureCount++ {
		if S.Get(newFeatureCount,newFeatureCount) <= significance*sigma0 {
			break
		}
	}

	fmt.Printf ("pcaChangeBasis:  Retaining %d most significant features\n", newFeatureCount)
	
	// Retain only the leftmost newFeatureCount columns of V
	V = V.GetMatrix(0, 0, featureCount, newFeatureCount)
	
	for _,d := range data {
		F := matrix.MakeDenseMatrix(d.continuousFeatures, 1, featureCount)
		newFeatures := matrix.Product(F,V)
		d.continuousFeatures = newFeatures.Array()
	}
}
