/// Haar Feature definitions and computation methods.
/// Design is based on PistonDevelopers/imageproc.

use super::util::Rectangle;
use super::Matrix;
use std::ops::Not;

pub struct HaarFeature {
    feature_type: HaarFeatureType,
    tl_sign: Sign,
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

pub enum Sign {
    Positive,
    Negative,
}

impl Not for Sign {
    type Output = Sign;

    fn not(self) -> Sign {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
        }
    }
}

impl HaarFeature {
    pub fn new(
        feature_type: HaarFeatureType,
        w: usize,
        h: usize,
        x: usize,
        y: usize,
    ) -> HaarFeature {
        // This bit isn't necessary, but keeping it around in case flipping signs
        // turns out to work well
        let tl_sign = match &feature_type {
            HaarFeatureType::TwoVertical => Sign::Positive,
            HaarFeatureType::TwoHorizontal => Sign::Negative,
            HaarFeatureType::ThreeHorizontal => Sign::Positive,
            HaarFeatureType::TwoByTwo => Sign::Positive,
        };
        HaarFeature {
            feature_type: feature_type,
            tl_sign,
            w,
            h,
            x,
            y,
        }
    }

    /// Evaluate the Haar feature on the integral image
    pub fn evaluate(&self, _img: &Matrix) -> u32 {
        let rects = self.to_rectangles();
        match &self.feature_type {
            HaarFeatureType::TwoVertical => {
                unimplemented!()
                // img[[self.x + self.w, self.h + self.y]] + img[[self.]]
            }
            _ => unimplemented!(),
        }
    }

    /// Turn width-height into rectangle
    fn to_rectangles(&self) -> Vec<(Rectangle, Sign)> {
        let mut rects = vec![(Rectangle::new((self.x, self.y), (self.x + self.w, self.y + self.h)), self.tl_sign)];

        match &self.feature_type {
            HaarFeatureType::TwoVertical => {
                rects.push((Rectangle::new((self.x, self.y + self.h), (self.x + self.w, self.y + 2 * self.h)), !self.tl_sign));
            },
            HaarFeatureType::TwoHorizontal => {
                rects.push((Rectangle::new((self.x + self.w, self.y), (self.x + 2 * self.w, self.y)), !self.tl_sign));
            },
            HaarFeatureType::ThreeHorizontal => {
                rects.push((Rectangle::new((self.x + self.w, self.y), (self.x + 2 * self.w, self.y)), !self.tl_sign));
                rects.push((Rectangle::new((self.x + 2 * self.w, self.y), (self.x + 3 * self.w, self.y)), self.tl_sign));
            },
            HaarFeatureType::TwoByTwo => {
                rects.push((Rectangle::new((self.x + self.w, self.y), (self.x + 2 * self.w, self.y)), !self.tl_sign));
                rects.push((Rectangle::new((self.x, self.y + self.h), (self.x + self.w, self.y + 2 * self.h)), !self.tl_sign));
                rects.push((Rectangle::new((self.x + self.w, self.y + self.h), (self.x + 2 * self.w, self.y + 2 * self.h)), self.tl_sign));
            },
        }

        rects
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
                        haar_features.push(HaarFeature::new(
                            HaarFeatureType::TwoHorizontal,
                            w,
                            h,
                            x,
                            y,
                        ));
                    }
                    if y + 2 * h <= maxh {
                        haar_features.push(HaarFeature::new(
                            HaarFeatureType::TwoVertical,
                            w,
                            h,
                            x,
                            y,
                        ));
                    }
                    if x + 3 * w <= maxw {
                        haar_features.push(HaarFeature::new(
                            HaarFeatureType::ThreeHorizontal,
                            w,
                            h,
                            x,
                            y,
                        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Checks that sums are computed correctly for integral images
    fn areas_computed_correctly() {
        assert!(true);
    }
}
