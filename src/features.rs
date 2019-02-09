/// Haar Feature definitions and computation methods.
/// Design is somewhat based on PistonDevelopers/imageproc.

use super::Matrix;


pub struct HaarFeature {
    feature_type: HaarFeatureType,
    tl_pos: bool,
    w: usize,
    h: usize,
    x: usize,
    y: usize,
}

pub enum HaarFeatureType {
    TwoVertical,
    TwoHorizontal,
    ThreeHorizontal,
    TwoByTwo,
}

impl HaarFeature {
    pub fn new(feature_type: HaarFeatureType, w: usize, h: usize, x: usize, y: usize) -> HaarFeature {
        let tl_pos = match &feature_type {
            HaarFeatureType::TwoVertical => true,
            HaarFeatureType::TwoHorizontal => false,
            HaarFeatureType::ThreeHorizontal => true,
            HaarFeatureType::TwoByTwo => true,
        };
        HaarFeature {
            feature_type: feature_type,
            tl_pos,
            w,
            h,
            x,
            y,
        }
    }

    /// Evaluate the Haar feature on the integral image
    pub fn evaluate(&self, img: &Matrix) -> u32 {
        match &self.feature_type {
            _ => unimplemented!(),
            // HaarFeatureType::TwoVertical => {
            //     // img[[self.x + self.w, self.h + self.y]] + img[[self.]]
            // }
        }
    }
}

/// Create a set of features that can be applies to every training input image.
/// Assumes that all training examples are the same shape.
pub fn init_haar_features(minw: usize, minh: usize, maxw: usize, maxh: usize) -> Vec<HaarFeature> {
    let mut haar_features = Vec::new();
    for w in minw..=maxw {
        for h in minh..=maxh {
            for x in 0..=(maxw - w) {
                for y in 0..=(maxh - h) {
                    if x + 2 * w <= maxw {
                        haar_features.push(HaarFeature::new(HaarFeatureType::TwoHorizontal, w, h, x, y));
                    }
                    if y + 2 * h <= maxh {
                        haar_features.push(HaarFeature::new(HaarFeatureType::TwoVertical, w, h, x, y));
                    }
                    if x + 3 * w <= maxw {
                        haar_features.push(HaarFeature::new(HaarFeatureType::ThreeHorizontal, w, h, x, y));
                    }
                    if x + 2 * w <= maxw && y + 2 * h < maxh {
                        haar_features.push(HaarFeature::new(HaarFeatureType::TwoByTwo, w, h, x, y));
                    }
                }
            }
        }
    }

    println!("Starting with {} Haar features", haar_features.len());
    haar_features
}
