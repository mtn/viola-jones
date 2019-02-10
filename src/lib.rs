extern crate indicatif;
extern crate ndarray;

pub mod features;
pub mod preprocess;
mod util;
pub mod weak_classifier;

use features::HaarFeature;
use indicatif::{ProgressBar, ProgressStyle};
use weak_classifier::WeakClassifier;

pub type Matrix = ndarray::Array2<i32>;

pub enum Classification {
    Face,
    NonFace,
}

pub struct Learner {
    num_weak_classifiers: u32,

    training_inputs: Vec<(Matrix, Classification)>,
    // faces: Vec<Matrix>,
    // backgrounds: Vec<Matrix>,
    haar_features: Vec<HaarFeature>,

    // Weights are kept separate by class for convenience (faces == positive)
    weights: Vec<f32>,
    // face_weights: Vec<f32>,
    // background_weights: Vec<f32>,
    weak_classifiers: Vec<WeakClassifier>,
}

impl Learner {
    pub fn new(faces_dir: &str, background_dir: &str, num_weak_classifiers: u32) -> Learner {
        // Load the data (faces followed by background, in tuples with class labels)
        let (training_inputs, nfaces, nbackgrounds) =
            preprocess::load_and_preprocess_data(faces_dir, background_dir);

        let mut weights = Vec::with_capacity(nfaces + nbackgrounds);
        for (i, w) in weights.iter_mut().enumerate() {
            if i < nfaces {
                *w = 1. / (2. * nfaces as f32);
            } else {
                *w = 1. / (2. * nbackgrounds as f32);
            }
        }

        // Note that the minimum window dimensions are set arbitrarily to 8x8
        let (maxw, maxh) = training_inputs[0].0.dim();
        Learner {
            num_weak_classifiers,
            training_inputs,
            weights,
            haar_features: features::init_haar_features(8, 8, maxw, maxh),
            weak_classifiers: Vec::new(),
        }
    }

    pub fn train(&mut self) {
        assert!(self.training_inputs.len() == 4000);

        println!("Made it to the start of training");

        let pb = ProgressBar::new(self.training_inputs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar().template("[{elapsed_precise}] {wide_bar} ({eta})"),
        );
        for (f, _) in &self.training_inputs {
            for ff in &self.haar_features {
                ff.evaluate(&f);
            }
            pb.inc(1);
        }
        pb.finish_with_message("done");
    }
}
