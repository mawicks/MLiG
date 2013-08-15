package ML

func GlassData(filename string) []*Data {
	return CSVData("ifffffffffc", filename, 8, 0)
}

func DigitData(filename string) []*Data {
	legend := "c"
	for i:=0; i<784; i++ {
		legend = legend + "f"
	}
	return CSVData(legend, filename, 10, 1)
}
