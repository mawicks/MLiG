package ML

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

func GlassData(filename string) []*Data {
	result := make([]*Data,0)

	var file io.ReadCloser
	var err error

	if file, err = os.Open(filename); err != nil {
		panic (fmt.Sprintf ("Unable to open file \"%s\" for input", filename))
	}
	
	csvReader := csv.NewReader(file)

	var fields []string

	for fields,err = csvReader.Read(); err==nil; fields,err = csvReader.Read() {
		if len(fields) == 11 {
			features := make([]float64,9)
			for i:=0; i<9; i++ {
				features[i],err = strconv.ParseFloat(fields[i+1],64)
				if err!=nil {
					panic (fmt.Sprintf("Numeric value expected: %s", fields[i+1]))
				}
			}
			output,err := strconv.ParseFloat(fields[10],64)
			if err!=nil {
				panic (fmt.Sprintf("Numeric value expected: %s", fields[10]))
			}
			result = append(result,
				&Data {
					continuousFeatures: features,
					categoricalFeatures: nil,
					output: output,
					outputCategories: 10,
					oobAccumulator: NewEntropyAccumulator(10)})
		}
	}

	if err != io.EOF {
		panic (err)
	}

	file.Close()

	return result
}
