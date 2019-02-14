use std::f64;

type WeakClassifier<'a> = super::weak_classifier::WeakClassifier<'a>;
type Classification = super::Classification;
type Matrix = ndarray::Array2<i64>;

#[derive(Debug)]
pub struct StrongClassifier<'a> {
    pub classifiers: Vec<WeakClassifier<'a>>,
    weights: Vec<f64>,
    threshold: f64,
}

impl<'a> StrongClassifier<'a> {
    /// Create a new strong classifier (finding the threshold that removes false
    /// negatives) from an ensemble of weak classifiers and trained weights.
    pub fn new() -> StrongClassifier<'a> {
        StrongClassifier {
            classifiers: Vec::new(),
            weights: Vec::new(),
            threshold: 0.,
        }
    }

    /// Makes a weighted classification prediction using the ensemble of classifiers.
    pub fn evaluate(&self, img: &Matrix) -> Classification {
        if self.evaluate_raw(img) - self.threshold >= 0. {
            Classification::Face
        } else {
            Classification::NonFace
        }
    }

    fn evaluate_raw(&self, img: &Matrix) -> f64 {
        let mut weighted_score = 0.;

        for (classifier, weight) in self.classifiers.iter().zip(self.weights.iter()) {
            weighted_score += weight * classifier.evaluate_raw(img) as f64;
        }

        weighted_score
    }

    /// Computes the error for an ensemble of classifiers (for a given threshold).
    pub fn compute_error(&self, input_samples: &Vec<(Matrix, Classification)>) -> (f64, f64, f64) {
        let mut num_false_negatives: f64 = 0.;
        let mut num_false_positives: f64 = 0.;

        for (img, label) in input_samples {
            let classification = self.evaluate(img);

            if classification != *label {
                match classification {
                    Classification::Face => num_false_positives += 1.,
                    Classification::NonFace => num_false_negatives += 1.,
                };
            }
        }

        (
            num_false_positives / input_samples.len() as f64,
            num_false_negatives / input_samples.len() as f64,
            (num_false_positives + num_false_negatives) / input_samples.len() as f64,
        )
    }

    /// Sets the threshold for this strong classifier (assuming the other fields have
    /// been initialized). Returns a copy of the updated weight value.
    fn update_threshold(&mut self, input_samples: &Vec<(Matrix, Classification)>) -> f64 {
        let mut min_score = f64::INFINITY;

        // Compute the minimal score of a face, and set that to be the threshold
        let mut face_scores = Vec::new();
        for (img, classification) in input_samples {
            if *classification == Classification::NonFace {
                continue;
            }

            let mut score = 0.;
            for (classifier, weight) in self.classifiers.iter().zip(self.weights.iter()) {
                score += weight * classifier.evaluate_raw(img) as f64;
            }

            face_scores.push(score);
        }

        face_scores.sort_by(|a, b| a.partial_cmp(&b).unwrap());

        // TODO make sure the vector is long enough
        self.threshold = face_scores[2];

        min_score
    }

    /// Adds a weak classifier to the ensemble (taking ownership of it), and its
    /// associated weight.
    pub fn add_weak_classifier(
        &mut self,
        classifier: WeakClassifier<'a>,
        weight: f64,
        input_samples: &Vec<(Matrix, Classification)>,
    ) {
        self.classifiers.push(classifier);
        self.weights.push(weight);

        self.update_threshold(input_samples);
    }
}
