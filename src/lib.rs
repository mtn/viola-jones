extern crate ndarray;

pub mod features;
pub mod preprocess;
pub mod weak_classifier;
mod util;

use features::HaarFeature;
use weak_classifier::WeakClassifier;

pub type Matrix = ndarray::Array2<i32>;

pub struct Learner {
    faces: Vec<Matrix>,
    backgrounds: Vec<Matrix>,

    haar_features: Vec<HaarFeature>,

    // Weights are kept separate by class for convenience (faces == positive)
    face_weights: Vec<f32>,
    background_weights: Vec<f32>,

    weak_classifiers: Vec<WeakClassifier>,
}

impl Learner {
    pub fn new(faces_dir: &str, background_dir: &str) -> Learner {
        let (faces, backgrounds) = preprocess::load_and_preprocess_data(faces_dir, background_dir);
        let (nfaces, nbackgrounds) = (faces.len(), backgrounds.len());
        // Note that the minimum window dimensions are set arbitrarily
        let (maxw, maxh) = faces[0].dim();
        Learner {
            faces,
            backgrounds,
            face_weights: vec![1. / (2. * nfaces as f32); nfaces],
            background_weights: vec![1. / (2. * nbackgrounds as f32); nbackgrounds],
            haar_features: features::init_haar_features(8, 8, maxw, maxh),
            weak_classifiers: Vec::new(),
        }
    }

    pub fn train(&mut self) {
        assert!(self.faces.len() == 2000);
        assert!(self.backgrounds.len() == 2000);
    }
}
