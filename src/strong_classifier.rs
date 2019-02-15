use serde::{Serialize, Deserialize};
use std::f64;

type WeakClassifier = super::weak_classifier::WeakClassifier;
type Classification = super::Classification;
type Matrix = ndarray::Array2<i64>;
type MatrixView<'a> = ndarray::ArrayView2<'a, i64>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrongClassifier {
    pub classifiers: Vec<WeakClassifier>,
    weights: Vec<f64>,
    threshold: f64,
}

impl StrongClassifier {
    /// Create a new strong classifier (finding the threshold that removes false
    /// negatives) from an ensemble of weak classifiers and trained weights.
    pub fn new() -> StrongClassifier {
        StrongClassifier {
            classifiers: Vec::new(),
            weights: Vec::new(),
            threshold: 0.,
        }
    }

    /// Makes a weighted classification prediction using the ensemble of classifiers.
    pub fn evaluate(&self, img: &MatrixView) -> Classification {
        if self.evaluate_raw(img) >= 0. {
            Classification::Face
        } else {
            Classification::NonFace
        }
    }

    fn evaluate_raw(&self, img: &MatrixView) -> f64 {
        let mut weighted_score = 0.;

        for (classifier, weight) in self.classifiers.iter().zip(self.weights.iter()) {
            weighted_score += weight * classifier.evaluate_raw(img) as f64;
        }

        weighted_score - self.threshold
    }

    /// Computes the error for an ensemble of classifiers (for a given threshold).
    pub fn compute_error(&self, input_samples: &Vec<(Matrix, Classification)>) -> (f64, f64, f64) {
        let mut num_false_negatives: f64 = 0.;
        let mut num_false_positives: f64 = 0.;
        let mut num_negatives = 0.;

        for (img, label) in input_samples {
            let classification = self.evaluate(&img.view());

            if *label == Classification::NonFace {
                num_negatives += 1.;
            }

            if classification != *label {
                match classification {
                    Classification::Face => num_false_positives += 1.,
                    Classification::NonFace => num_false_negatives += 1.,
                };
            }
        }

        println!("Computing error, len is {}", input_samples.len());
        (
            num_false_positives / num_negatives,
            num_false_negatives / (input_samples.len() as f64 - num_negatives),
            (num_false_positives + num_false_negatives) / input_samples.len() as f64,
        )
    }

    /// Sets the threshold for this strong classifier (assuming the other fields have
    /// been initialized). Returns a copy of the updated weight value.
    fn update_threshold(&mut self, input_samples: &Vec<(Matrix, Classification)>) -> f64 {
        // Compute the minimal score of a face, and set that to be the threshold
        let mut face_scores = Vec::new();
        for (img, classification) in input_samples {
            if *classification == Classification::NonFace {
                continue;
            }

            let mut score = 0.;
            for (classifier, weight) in self.classifiers.iter().zip(self.weights.iter()) {
                score += weight * classifier.evaluate_raw(&img.view()) as f64;
            }

            face_scores.push(score);
        }

        face_scores.sort_by(|a, b| a.partial_cmp(&b).unwrap());

        let ind = (face_scores.len() as f64 * 0.05).floor() as usize;
        self.threshold = face_scores[ind];

        self.threshold
    }

    /// Adds a weak classifier to the ensemble (taking ownership of it), and its
    /// associated weight.
    pub fn add_weak_classifier(
        &mut self,
        classifier: WeakClassifier,
        weight: f64,
        input_samples: &Vec<(Matrix, Classification)>,
    ) {
        self.classifiers.push(classifier);
        self.weights.push(weight);

        self.update_threshold(input_samples);
    }
}
