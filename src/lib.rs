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

pub type Matrix = ndarray::Array2<i32>;

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

pub struct Learner {
    max_weak_learners_per_level: u8,
    max_cascade_depth: u8,

    training_inputs: Vec<(Matrix, Classification)>,

    haar_features: Vec<HaarFeature>,

    distribution: Vec<f64>,
}

impl Learner {
    pub fn new(
        faces_dir: &str,
        background_dir: &str,
        max_weak_learners_per_level: u8,
        max_cascade_depth: u8,
    ) -> Learner {
        // Load the data (faces followed by background, in tuples with class labels)
        let (training_inputs, nfaces, nbackgrounds) =
            preprocess::load_and_preprocess_data(faces_dir, background_dir);

        let (maxw, maxh) = training_inputs[0].0.dim();
        // let mut distribution: Vec<f64> = Vec::with_capacity(nfaces + nbackgrounds);
        // for i in 0..(nfaces + nbackgrounds) {
        //     if i < nfaces {
        //         distribution.push(1. / (nbackgrounds + nfaces) as f64);
        //     } else {
        //         distribution.push(-1. / (nbackgrounds + nfaces) as f64);
        //     }
        // }

        // Note that the stride and step size are arbitrarily set to 4 and 4.
        // This pretty dramatically cuts down training time by restricting the search
        // space.
        Learner {
            max_weak_learners_per_level,
            max_cascade_depth,
            training_inputs,
            haar_features: features::init_haar_features(maxw, maxh, 8, 8),
            distribution: vec![1. / (nbackgrounds + nfaces) as f64; nfaces + nbackgrounds],
        }
    }

    /// Creates a strong classifier from a single round of boosting.
    /// Returns a strong learner/committee.
    fn run_boosting(&mut self) -> StrongClassifier {
        let mut strong = StrongClassifier::new();

        // To avoid getting stuck to do outliers, we limit the number of total weak
        // learners we add to the classifier in a given boosting.
        for boosting_round in 0..self.max_weak_learners_per_level {
            println!("In boosting round {}", boosting_round);

            let (best_classifier, best_error): (WeakClassifier, f64) = WeakClassifier::best_stump(
                &self.haar_features,
                &self.training_inputs,
                &mut self.distribution,
            );

            let alpha_t = (1. / 2.) * ((1. - best_error) / best_error).ln();
            strong.add_weak_classifier(best_classifier, alpha_t, &self.training_inputs);

            // Turn this into a strong learner by itself and return
            if best_error == 0. {
                println!("Found a single weak classifier that had 0 error, returning early");
                println!("{:?}", strong.classifiers);
                // TODO update
                unimplemented!()
                // return StrongClassifier {
                //     weights: vec![alpha_t],
                //     classifiers: vec![best_classifier],
                // };
            }

            // Update the distribution weights
            let normalization_factor = 2. * (best_error * (1. - best_error)).sqrt();
            for (i, sample) in self.training_inputs.iter().enumerate() {
                // The classification result multiplies like -1 and 1
                let classification = strong.classifiers.last().unwrap().evaluate(&sample.0);
                self.distribution[i] = (self.distribution[i] / normalization_factor)
                    * (classification * sample.1 * -1. * alpha_t).exp();
            }

            let (fpr, fnr, overall) = strong.compute_error(&self.training_inputs);

            println!("Boosting round {}: Currently have {} weak classifiers with FPR {} and FNR {} and overall error {}", boosting_round, strong.classifiers.len(), fpr, fnr, overall);
        }

        // TODO fix
        // StrongClassifier {
        //     weights,
        //     classifiers: boosting_classifiers,
        // }
        unimplemented!()
    }

    pub fn train(&mut self) {
        assert!(self.training_inputs.len() == 4000);
        println!("Beginning training...");

        let strong_learners: Vec<StrongClassifier> =
            Vec::with_capacity(self.max_cascade_depth as usize);

        for cascade_round in 0..self.max_cascade_depth {
            println!("Starting cascade round {}", cascade_round);

            let classifier: StrongClassifier = self.run_boosting();
        }

        // let pb = ProgressBar::new(self.training_inputs.len() as u64);
        // pb.set_style(
        //     ProgressStyle::default_bar().template("[{elapsed_precise}] {wide_bar} ({eta})"),
        // );
        // for (f, _) in &self.training_inputs {
        //     for ff in &self.haar_features {
        //         ff.evaluate(&f);
        //     }
        //     pb.inc(1);
        // }
        // pb.finish_with_message("done");
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
