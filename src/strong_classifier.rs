type WeakClassifier<'a> = super::weak_classifier::WeakClassifier<'a>;
type Classification = super::Classification;

pub struct StrongClassifier<'a> {
    pub classifiers: Vec<WeakClassifier<'a>>,
    pub weights: Vec<f64>,
}

impl<'a> StrongClassifier<'a> {
    pub fn evaluate(&self) -> Classification {
        unimplemented!();
    }
}
