package ML

func GlassData(filename string) []*Data {
	return CSVData("ifffffffffc", filename, 8, 0)
}

func DigitData(filename string) []*Data {
	legend := "c"
	for i:=0; i<784; i++ {
		legend = legend + "f"
	}

	data := CSVData(legend, filename, 10, 1)

	for _,d := range data {
		f := d.Features()
		
		rowSums := make([]float64, 28)
		colSums := make([]float64, 28)

		topBottom := 0.0;
		leftRight := 0.0
		
		for i:=0; i<28; i++ {
			rowIndex := i*28
			for j:=0; j<28; j++ {
				rowSums[i] += f[rowIndex+j]
			}
			if i < 14 {
				topBottom += rowSums[i]
			} else {
				topBottom -= rowSums[i]
			}
		}

		for j:=0; j<28; j++ {
			offset := j
			for i:=0; i<28; i++ {
				colSums[j] += f[offset]
				offset += 28
			}
			if j < 14 {
				leftRight -= colSums[j]
			} else {
				leftRight += colSums[j]
			}
		}

//		d.AppendFeatures(rowSums)
//		d.AppendFeatures(colSums)
		d.AppendFeatures([]float64{topBottom})
		d.AppendFeatures([]float64{leftRight})
	}
	return data
}













