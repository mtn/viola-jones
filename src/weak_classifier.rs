use indicatif::{ProgressBar, ProgressStyle};

type Feature = super::features::HaarFeature;
type Toggle = super::features::Sign;
type Matrix = ndarray::Array2<i32>;
type Classification = super::Classification;

pub struct WeakClassifier<'a> {
    feature: &'a Feature,
    toggle: Toggle,
    threshold: i32,
}

impl<'a> WeakClassifier<'a> {
    pub fn new(feature: &'a Feature, threshold: i32, toggle: Toggle) -> WeakClassifier {
        WeakClassifier {
            feature, threshold, toggle
        }
    }

    /// Finds the optimal weak classifier for each feature, returning a vector them all.
    pub fn get_optimals(
        features: &'a Vec<Feature>,
        training_samples: &Vec<(Matrix, Classification)>,
        distribution_t: &Vec<f32>,
    ) -> Vec<WeakClassifier<'a>> {
        assert!(training_samples.len() == distribution_t.len());

        println!("Running a search over {} features...", features.len());
        let pb = ProgressBar::new(features.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar().template("[{elapsed_precise}] {wide_bar} ({eta})"),
        );

        // The total positive and negative weights
        let mut t_pos: f32 = 0.;
        let mut t_neg: f32 = 0.;
        for ((_, label), dist) in training_samples.iter().zip(distribution_t.iter()) {
            if *label == Classification::Face {
                t_pos += dist;
            } else {
                t_neg += dist;
            }
        }

        let mut classifiers: Vec<WeakClassifier> = Vec::with_capacity(features.len());
        for feature in features {
            // A vector of tuples (score, distribution)
            let mut scores: Vec<(i32, f32, Classification)> =
                Vec::with_capacity(training_samples.len());
            for (sample, dist) in training_samples.iter().zip(distribution_t.iter()) {
                scores.push((feature.evaluate(&sample.0), *dist, sample.1));
            }
            scores.sort_by(|a, b| a.0.cmp(&b.0));

            // Initialize s_pos, best_error, best_toggle, and best_threshold
            let mut s_pos: f32 = if distribution_t[0] < 0. {
                0.
            } else {
                distribution_t[0]
            };
            let mut s_neg: f32 = if distribution_t[0] >= 0. {
                0.
            } else {
                -distribution_t[0]
            };
            let (a, b) = (s_pos + t_neg - s_neg, s_neg + t_pos - s_pos);
            let (mut best_error, mut best_toggle) = if a <= b {
                (a, Toggle::Positive)
            } else {
                (b, Toggle::Negative)
            };
            let mut best_threshold = scores[0].0;

            for (score, dist, label) in scores.iter().skip(1) {
                if *label == Classification::Face {
                    s_pos += dist;
                } else {
                    s_neg -= dist;
                }

                let (a, b) = (s_pos + t_neg - s_neg, s_neg + t_pos - s_pos);
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

            classifiers.push(WeakClassifier::new(feature, best_threshold, best_toggle));

//             println!("{:?}", scores);
//             println!("{}", scores.len());
//             assert!(false);

            pb.inc(1);
        }

        pb.finish_with_message("done");

        classifiers
    }
}
