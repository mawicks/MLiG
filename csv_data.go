package ML

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

func fieldTypeCount (legend string, c rune) (result int) {
	for _,ft := range []rune(legend) {
		if ft == c {
			result += 1
		}
	}
	return result
}

// CSVData opens "filename" and interprets the data according to the characters in "legend".
// Each character represent a single field as follows:
// i - ignored field
// f - feature (continuous-valued or categorical)
// k - key (string)
// r - regression output
// c - categorical output

func CSVData(legend string, filename string, outputCategories, skip int) []*Data {
	result := make([]*Data,0)

	var file io.ReadCloser
	var err error

	if file, err = os.Open(filename); err != nil {
		panic (fmt.Sprintf ("Unable to open file \"%s\" for input", filename))
	}
	
	var output float64
	csvReader := csv.NewReader(file)

	var fields []string
	var key string

	recordCount := 0
	for fields,err = csvReader.Read(); err==nil; fields,err = csvReader.Read() {
		recordCount += 1
		// Default key is the record number
		// Any field may be used as the key by using the "k" indicator in legend.
		key = strconv.FormatInt(int64(recordCount),10)
		
		if len(fields) != len(legend) {
			panic (fmt.Sprintf("Wrong number of fields: got %d expected %d", len(fields), len(legend)))
		}
		if (skip > 0) {
			skip--
			continue
		}
		

		features := make([]float64,fieldTypeCount(legend, 'f'))
		featureCount := 0
		for i,c := range legend {
			switch c {
			case 'k':	// Key
				key = fields[i]
			case 'f':	// Feature
				features[featureCount],err = strconv.ParseFloat(fields[i],64)
				if err!=nil {
					panic (fmt.Sprintf("Numeric value expected: %s", fields[i]))
				}
				featureCount += 1
			case 'i':	// Ignored
			case 'r':	// Regression output
				output,err = strconv.ParseFloat(fields[i],64)
				if err!=nil {
					panic (fmt.Sprintf("Numeric value expected: %s", fields[i]))
				}
			case 'c':	// Categorical output
				output,err = strconv.ParseFloat(fields[i],64)
				if err!=nil {
					panic (fmt.Sprintf("Numeric value expected: %s", fields[i]))
				}
				if output > float64(outputCategories) {
					panic (fmt.Sprintf("Output value %g larger than output categories: %d", output, outputCategories))
				}
			}
			
		}
		var errorAccumulator ErrorAccumulator
		if outputCategories == 1 {
			errorAccumulator = &StatAccumulator{}
		} else if outputCategories > 1 {
			errorAccumulator = NewEntropyAccumulator(outputCategories)
		}
		
		result = append(result, &Data {
			key: key,
			weight: 1.0,
			continuousFeatures: features,
			categoricalFeatures: nil,
			output: output,
			outputCategories: outputCategories,
			oobAccumulator: errorAccumulator})
	}
	
	if err != io.EOF {
		panic (err)
	}
	
	file.Close()
	
	return result
}










