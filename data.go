package ML

type Feature interface {
	// compareTo() is only required to work for features of the same type
	// e.g., continuous or categorical.
	compareTo (f Feature) int
}

type FeatureType int

type Data struct {
	continuousFeatures []float64
	categoricalFeatures []int
	output float64
	outputCategories int // 0 means no output, 1 means continuous

	featureSelector func (int32) float64
	oobAccumulator ErrorAccumulator
}

func (d *Data) AppendFeatures(af []float64) {
	d.continuousFeatures = append(d.continuousFeatures, af...)
}

func (d *Data) Features() []float64 {
	return d.continuousFeatures
}

type sortableData struct {
	data []*Data
	seed int32
}

func (s sortableData) Len() int {
	return len(s.data)
}

func (s sortableData) Less(i, j int) bool {
	return s.data[i].featureSelector(s.seed) < s.data[j].featureSelector(s.seed)
}

func (s sortableData) Swap(i, j int) {
	s.data[i],s.data[j] = s.data[j],s.data[i]
}











