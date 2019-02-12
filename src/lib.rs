#![feature(type_alias_enum_variants)]

extern crate indicatif;
extern crate ndarray;

mod features;
mod preprocess;
mod util;
mod weak_classifier;

use features::HaarFeature;
// use indicatif::{ProgressBar, ProgressStyle};
use weak_classifier::WeakClassifier;

pub type Matrix = ndarray::Array2<i32>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Classification {
    Face,
    NonFace,
}

pub struct Learner {
    training_inputs: Vec<(Matrix, Classification)>,
    haar_features: Vec<HaarFeature>,

    distribution: Vec<f32>,
}

impl Learner {
    pub fn new(faces_dir: &str, background_dir: &str) -> Learner {
        // Load the data (faces followed by background, in tuples with class labels)
        let (training_inputs, nfaces, nbackgrounds) =
            preprocess::load_and_preprocess_data(faces_dir, background_dir);

        // Note that the stride and step size are arbitrarily set to 4 and 4.
        // This pretty dramatically cuts down training time by restricting the search
        // space.
        let (maxw, maxh) = training_inputs[0].0.dim();
        Learner {
            training_inputs,
            haar_features: features::init_haar_features(maxw, maxh, 4, 4),
            distribution: vec![1. / (nbackgrounds + nfaces) as f32; nfaces + nbackgrounds],
        }
    }
    pub fn train(&mut self) {
        assert!(self.training_inputs.len() == 4000);

        println!("Made it to the start of training");
        // unimplemented!();
        WeakClassifier::get_optimals(
            &self.haar_features,
            &self.training_inputs,
            &mut self.distribution,
        );

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
