#![feature(type_alias_enum_variants)]

extern crate indicatif;
extern crate ndarray;
extern crate rayon;

mod features;
mod preprocess;
mod strong_classifier;
mod util;
mod weak_classifier;

use features::HaarFeature;
use std::f64;
use std::ops::Mul;
use strong_classifier::StrongClassifier;
use weak_classifier::WeakClassifier;
// use indicatif::{ProgressBar, ProgressStyle};

pub type Matrix = ndarray::Array2<i64>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Classification {
    Face,
    NonFace,
}

impl Mul<f64> for Classification {
    type Output = f64;

    fn mul(self, rhs: f64) -> f64 {
        match self {
            Classification::Face => rhs,
            Classification::NonFace => -1. * rhs,
        }
    }
}

impl Mul for Classification {
    type Output = f64;

    fn mul(self, rhs: Classification) -> f64 {
        match self {
            Classification::Face => rhs * 1f64,
            Classification::NonFace => rhs * -1f64,
        }
    }
}

/// A cascaded learner.
pub struct Learner {
    max_cascade_depth: u8,

    training_inputs: Vec<(Matrix, Classification)>,

    haar_features: Vec<HaarFeature>,
}

impl Learner {
    pub fn new(
        faces_dir: &str,
        background_dir: &str,
        max_cascade_depth: u8,
    ) -> Learner {
        // Load the data (faces followed by background, in tuples with class labels)
        let (training_inputs, nfaces, nbackgrounds) =
            preprocess::load_and_preprocess_data(faces_dir, background_dir);

        let (maxw, maxh) = training_inputs[0].0.dim();

        // Note that the stride and step size are arbitrarily set to 4 and 4.
        // This pretty dramatically cuts down training time by restricting the search
        // space.
        Learner {
            max_cascade_depth,
            training_inputs,
            haar_features: features::init_haar_features(maxw, maxh, 6, 6),
        }
    }

    /// Creates a strong classifier from a single round of boosting.
    /// Returns a strong learner/committee.
    fn run_boosting(&self) -> StrongClassifier {
        let mut strong = StrongClassifier::new();

        // To avoid getting stuck to do outliers, we limit the number of total weak
        // learners we add to the classifier in a given boosting.
        let mut boosting_round = 0;
        let mut distribution = vec![1. / self.training_inputs.len() as f64; self.training_inputs.len()];
        loop {
            boosting_round += 1;

            let (best_classifier, best_error): (WeakClassifier, f64) = WeakClassifier::best_stump(
                &self.haar_features,
                &self.training_inputs,
                &distribution,
            );

            let alpha_t = (0.5) * ((1. - best_error) / best_error).ln();
            strong.add_weak_classifier(best_classifier, alpha_t, &self.training_inputs);

            // Turn this into a strong learner by itself and return
            if best_error == 0. {
                println!("Found a single weak classifier that had 0 error, returning early");
                unimplemented!()
            }

            // Update the distribution weights
            // let normalization_factor: f64 = 2. * (best_error * (1. - best_error)).sqrt();
            let mut newtot = 0.;
            for (i, sample) in self.training_inputs.iter().enumerate() {
                // The classification result multiplies like -1 and 1
                let classification = strong.classifiers.last().unwrap().evaluate(&sample.0);
                distribution[i] =
                    (distribution[i]) * (classification * sample.1 * -1. * alpha_t).exp();
                newtot += distribution[i];
            }

            distribution = distribution.iter().map(|x| x / newtot).collect();

            let (fpr, fnr, overall) = strong.compute_error(&self.training_inputs);

            println!("Finished boosting round {}", boosting_round);
            println!(
                "Currently have {} weak classifiers with FPR {} and FNR {} and overall error {}",
                strong.classifiers.len(),
                fpr,
                fnr,
                overall
            );

            if fpr <= 0.3 {
                break;
            }
        }

        strong
    }

    pub fn train(&mut self) {
        assert!(self.training_inputs.len() == 4000);
        println!("Beginning training...");

        let mut strong_learners: Vec<StrongClassifier> =
            Vec::with_capacity(self.max_cascade_depth as usize);

        let mut cascade_round = 0;
        loop {
            cascade_round += 1;
            println!("Starting cascade round {}", cascade_round);

            strong_learners.push(self.run_boosting());

            // Remove examples that are classified as negative from the set of inputs
            // that gets fed into the next layer in the cascade. This removes a trivial
            // amount of false negatives (2), which isn't a big deal.
            let mut new_inputs = Vec::new();
            for (sample, label) in &self.training_inputs {
                if strong_learners.last().unwrap().evaluate(&sample) == Classification::Face {
                    new_inputs.push((sample.clone(), *label));
                }
            }
            self.training_inputs = new_inputs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Checks that sums are computed correctly for integral images
    fn classifications_multiply_correctly() {
        let mut label = Classification::Face;
        assert!(label * -1. == -1.);

        let mut classification = Classification::Face;
        assert!(label * classification * -1. == -1.);
        classification = Classification::NonFace;
        assert!(label * classification * -1. == 1.);

        label = Classification::NonFace;
        assert!(label * -1. == 1.);

        classification = Classification::Face;
        assert!(label * classification * -1. == 1.);
        classification = Classification::NonFace;
        assert!(label * classification * -1. == -1.);
    }
}
