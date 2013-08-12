package ML

import (
	"math/rand"
)

func ShuffleData (data []*Data) {
	n := len(data)
	for i,_ := range data {
		j := int(rand.Int31n(int32(n)))
		data[i],data[j] = data[j],data[i]
	}
}