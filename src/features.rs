/// Haar Feature definitions and computation methods.
/// Design is based on PistonDevelopers/imageproc.
use super::util::{compute_area, Rectangle};
use super::Matrix;
use std::ops::{Mul, Not};

#[derive(Debug)]
pub struct HaarFeature {
    feature_type: HaarFeatureType,
    tl_sign: Sign,
    w: usize,
    h: usize,
    x: usize,
    y: usize,
}

#[derive(Debug)]
pub enum HaarFeatureType {
    TwoVertical,
    TwoHorizontal,
    ThreeHorizontal,
    TwoByTwo,
}

#[derive(Copy, Clone, Debug)]
pub enum Sign {
    Positive,
    Negative,
}

impl Not for Sign {
    type Output = Self;

    fn not(self) -> Self {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
        }
    }
}

impl Mul<i32> for Sign {
    type Output = i32;

    fn mul(self, rhs: i32) -> i32 {
        match self {
            Sign::Positive => rhs,
            Sign::Negative => -1 * rhs,
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
        // turns out to work well. From the paper, white regions are subtracted
        // from grey ones.
        let tl_sign = match &feature_type {
            HaarFeatureType::TwoVertical => Sign::Positive,
            HaarFeatureType::TwoHorizontal => Sign::Negative,
            HaarFeatureType::ThreeHorizontal => Sign::Negative,
            HaarFeatureType::TwoByTwo => Sign::Negative,
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

    /// Evaluate the Haar feature on the integral image.
    /// No bounds checking is done up-front.
    pub fn evaluate(&self, img: &Matrix) -> i32 {
        let rects = self.to_rectangles();
        let mut score = 0;

        for (rect, sgn) in rects {
            score += sgn * compute_area(img, &rect);
        }

        score
    }

    /// Turn width-height into rectangle
    fn to_rectangles(&self) -> Vec<(Rectangle, Sign)> {
        let mut rects = vec![(
            Rectangle::new((self.x, self.y), (self.x + self.w, self.y + self.h)),
            self.tl_sign,
        )];

        match &self.feature_type {
            HaarFeatureType::TwoVertical => {
                rects.push((
                    Rectangle::new(
                        (self.x, self.y + self.h),
                        (self.x + self.w, self.y + 2 * self.h),
                    ),
                    !self.tl_sign,
                ));
            }
            HaarFeatureType::TwoHorizontal => {
                rects.push((
                    Rectangle::new(
                        (self.x + self.w, self.y),
                        (self.x + 2 * self.w, self.y + self.h),
                    ),
                    !self.tl_sign,
                ));
            }
            HaarFeatureType::ThreeHorizontal => {
                rects.push((
                    Rectangle::new(
                        (self.x + self.w, self.y),
                        (self.x + 2 * self.w, self.y + self.h),
                    ),
                    !self.tl_sign,
                ));
                rects.push((
                    Rectangle::new(
                        (self.x + 2 * self.w, self.y),
                        (self.x + 3 * self.w, self.y + self.h),
                    ),
                    self.tl_sign,
                ));
            }
            HaarFeatureType::TwoByTwo => {
                rects.push((
                    Rectangle::new(
                        (self.x + self.w, self.y),
                        (self.x + 2 * self.w, self.y + self.h),
                    ),
                    !self.tl_sign,
                ));
                rects.push((
                    Rectangle::new(
                        (self.x, self.y + self.h),
                        (self.x + self.w, self.y + 2 * self.h),
                    ),
                    !self.tl_sign,
                ));
                rects.push((
                    Rectangle::new(
                        (self.x + self.w, self.y + self.h),
                        (self.x + 2 * self.w, self.y + 2 * self.h),
                    ),
                    self.tl_sign,
                ));
            }
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

// use ::preprocess::compute_integral_image;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::compute_integral_image;
    use ndarray::Array;

    #[test]
    fn two_vert_evaluates_correctly() {
        let two_vert1 = HaarFeature::new(HaarFeatureType::TwoVertical, 1, 1, 0, 2);
        let two_vert2 = HaarFeature::new(HaarFeatureType::TwoVertical, 2, 2, 0, 0);

        let m1 = compute_integral_image(&Array::ones((4, 4)));
        assert!(two_vert1.evaluate(&m1) == 0);
        assert!(two_vert2.evaluate(&m1) == 0);

        let mut m2 = Array::ones((4, 4));
        for y in 2..4 {
            for x in 0..4 {
                m2[[y, x]] = -1;
            }
        }
        let m2 = compute_integral_image(&m2);
        assert!(two_vert1.evaluate(&m2) == 0);
        assert!(two_vert2.evaluate(&m2) == 8);
    }

    #[test]
    fn two_horiz_evaluates_correctly() {
        let two_horiz1 = HaarFeature::new(HaarFeatureType::TwoHorizontal, 1, 1, 2, 2);
        let two_horiz2 = HaarFeature::new(HaarFeatureType::TwoHorizontal, 2, 2, 0, 0);
        let two_horiz3 = HaarFeature::new(HaarFeatureType::TwoHorizontal, 1, 1, 1, 0);

        let m1 = compute_integral_image(&Array::ones((4, 4)));
        assert!(two_horiz1.evaluate(&m1) == 0);
        assert!(two_horiz2.evaluate(&m1) == 0);
        assert!(two_horiz3.evaluate(&m1) == 0);

        let mut m2 = Array::ones((4, 4));
        for y in 0..4 {
            for x in 2..4 {
                m2[[y, x]] = -1;
            }
        }

        let m2 = compute_integral_image(&m2);
        assert!(two_horiz1.evaluate(&m2) == 0);
        assert!(two_horiz2.evaluate(&m2) == -8);
        assert!(two_horiz3.evaluate(&m2) == -2);
    }

    #[test]
    fn three_horiz_evaluates_correctly() {
        let three_horiz1 = HaarFeature::new(HaarFeatureType::ThreeHorizontal, 1, 1, 1, 1);
        let three_horiz2 = HaarFeature::new(HaarFeatureType::ThreeHorizontal, 1, 1, 0, 0);
        let three_horiz3 = HaarFeature::new(HaarFeatureType::ThreeHorizontal, 2, 1, 0, 0);
        let three_horiz4 = HaarFeature::new(HaarFeatureType::ThreeHorizontal, 1, 2, 0, 0);
        let three_horiz5 = HaarFeature::new(HaarFeatureType::ThreeHorizontal, 2, 2, 0, 3);

        let m1 = compute_integral_image(&Array::ones((4, 4)));
        assert!(three_horiz1.evaluate(&m1) == -1);
        assert!(three_horiz2.evaluate(&m1) == -1);

        let mut m2 = Array::ones((6, 6));
        for y in 0..6 {
            for x in 3..6 {
                m2[[y, x]] = -1;
            }
        }

        let m2 = compute_integral_image(&m2);
        assert!(three_horiz1.evaluate(&m2) == 1);
        assert!(three_horiz2.evaluate(&m2) == -1);
        assert!(three_horiz3.evaluate(&m2) == 0);
        assert!(three_horiz4.evaluate(&m2) == -2);
        assert!(three_horiz5.evaluate(&m2) == 0);
    }

    #[test]
    fn two_by_two_evaluates_correctly() {
        let two_by_two1 = HaarFeature::new(HaarFeatureType::TwoByTwo, 1, 1, 1, 1);
        let two_by_two2 = HaarFeature::new(HaarFeatureType::TwoByTwo, 1, 1, 0, 0);
        let two_by_two3 = HaarFeature::new(HaarFeatureType::TwoByTwo, 2, 1, 0, 0);
        let two_by_two4 = HaarFeature::new(HaarFeatureType::TwoByTwo, 1, 2, 0, 0);
        let two_by_two5 = HaarFeature::new(HaarFeatureType::TwoByTwo, 2, 2, 2, 2);

        let m1 = compute_integral_image(&Array::ones((4, 4)));
        assert!(two_by_two1.evaluate(&m1) == 0);
        assert!(two_by_two2.evaluate(&m1) == 0);

        let mut m2 = Array::ones((6, 6));
        for y in 0..6 {
            for x in 3..6 {
                m2[[y, x]] = -1;
            }
        }

        let m2 = compute_integral_image(&m2);
        assert!(two_by_two1.evaluate(&m2) == 0);
        assert!(two_by_two2.evaluate(&m2) == 0);
        assert!(two_by_two3.evaluate(&m2) == 0);
        assert!(two_by_two4.evaluate(&m2) == 0);
        assert!(two_by_two5.evaluate(&m2) == 0);
    }
}
