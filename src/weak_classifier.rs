use indicatif::{ProgressBar, ProgressStyle};
use std::f64;

type Feature = super::features::HaarFeature;
type Toggle = super::features::Sign;
type Matrix = ndarray::Array2<i64>;
type Classification = super::Classification;

#[derive(Clone, Debug)]
pub struct WeakClassifier<'a> {
    feature: &'a Feature,
    toggle: Toggle,
    threshold: i64,
}

impl<'a> WeakClassifier<'a> {
    pub fn new(feature: &'a Feature, threshold: i64, toggle: Toggle) -> WeakClassifier {
        WeakClassifier {
            feature,
            threshold,
            toggle,
        }
    }

    fn get_optimal(
        feature: &'a Feature,
        training_samples: &Vec<(Matrix, Classification)>,
        distribution_t: &Vec<f64>,
        t_pos: f64,
        t_neg: f64,
    ) -> (WeakClassifier<'a>, f64) {
        // A vector of tuples (score, distribution, true label)
        let mut scores: Vec<(i64, f64, Classification)> =
            Vec::with_capacity(training_samples.len());
        for (sample, dist) in training_samples.iter().zip(distribution_t.iter()) {
            scores.push((feature.evaluate(&sample.0), *dist, sample.1));
        }
        scores.sort_by(|a, b| a.0.cmp(&b.0));

        // Initialize s_pos, best_error, best_toggle, and best_threshold.
        // This is just an incremental way of computing the weighted sum from the paper.
        // let (a, b) = (s_pos + t_neg - s_neg, s_neg + t_pos - s_pos);
        // let (mut best_error, mut best_toggle) = if a <= b {
        //     (a, Toggle::Positive)
        // } else {
        //     (b, Toggle::Negative)
        // };
        // let mut best_threshold = scores[0].0;

        let mut best_threshold = 0;
        let mut best_toggle = Toggle::Positive;
        let mut best_error = 2.;
        let mut s_pos = 0.;
        let mut s_neg = 0.;
        for (score, dist, label) in scores.iter().skip(1) {
            if *label == Classification::Face {
                s_pos += dist;
            } else {
                s_neg += dist;
            }
            // println!("spos {} sneg {} sum {}", s_pos, s_neg, s_pos + s_neg);

            let (a, b) = (s_pos + t_neg - s_neg, s_neg + t_pos - s_pos);
            // assert!((t_pos + t_neg - 1.).abs() < 0.0001);
            let error = a.min(b);
            if error < best_error {
                best_error = error;
                best_threshold = *score;
                best_toggle = if a < b {
                    Toggle::Positive
                } else {
                    Toggle::Negative
                };
            }
        }
        // assert!(false);

        (
            WeakClassifier::new(feature, best_threshold, best_toggle),
            best_error,
        )
    }

    /// Finds the optimal (attaining the lowest empirical loss) weak classifier for
    /// each feature, returning a vector of optimal weak classifiers.
    fn get_optimals(
        features: &'a Vec<Feature>,
        training_samples: &Vec<(Matrix, Classification)>,
        distribution_t: &Vec<f64>,
    ) -> Vec<(WeakClassifier<'a>, f64)> {
        assert!(training_samples.len() == distribution_t.len());

        println!("Running a search over {} features...", features.len());
        let pb = ProgressBar::new(features.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar().template("[{elapsed_precise}] {wide_bar} ({eta})"),
        );

        // The total positive and negative weights
        let mut t_pos: f64 = 0.;
        let mut t_neg: f64 = 0.;
        for ((_, label), dist) in training_samples.iter().zip(distribution_t.iter()) {
            if *label == Classification::Face {
                t_pos += dist;
            } else {
                t_neg += dist;
            }
        }
        // println!("tpos and tneg computed as {} + {} = {}", t_pos, t_neg, t_pos + t_neg);

        let mut classifiers: Vec<(WeakClassifier, f64)> = Vec::with_capacity(features.len());
        for feature in features {
            classifiers.push(Self::get_optimal(
                &feature,
                training_samples,
                distribution_t,
                t_pos,
                t_neg,
            ));

            pb.inc(1);
        }

        pb.finish_with_message("done");

        classifiers
    }

    /// Returns the best decision stump over the set of optimal stumps.
    pub fn best_stump(
        features: &'a Vec<Feature>,
        training_samples: &Vec<(Matrix, Classification)>,
        distribution_t: &Vec<f64>,
    ) -> (WeakClassifier<'a>, f64) {
        let mut weak_classifiers = Self::get_optimals(features, training_samples, distribution_t);

        // Select the best classifier based on error rate.
        // Sorting is more expensive than a linear search, but there aren't that many
        // and it works better with this memory model.
        weak_classifiers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        weak_classifiers[0].clone()
    }

    /// Evaluate the weak classifier on an input image.
    pub fn evaluate(&self, img: &Matrix) -> Classification {
        if self.evaluate_raw(img) >= 0 {
            Classification::Face
        } else {
            Classification::NonFace
        }
    }

    /// Return the raw score of the evaluated feature.
    pub fn evaluate_raw(&self, img: &Matrix) -> i64 {
        self.toggle * (self.feature.evaluate(img) - self.threshold)
    }

    /// Computes the weighted error of the weak classifier
    pub fn compute_error(
        &self,
        input_samples: &Vec<(Matrix, Classification)>,
        weights: &Vec<f64>,
    ) -> f64 {
        let mut weighted_error = 0.;

        for ((sample, label), weight) in input_samples.iter().zip(weights.iter()) {
            let classification = self.evaluate(sample);

            if classification != *label {
                weighted_error += *weight;
            }
        }

        weighted_error
    }
}
