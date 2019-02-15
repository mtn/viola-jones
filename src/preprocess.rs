/// Functions for loading the pre-processing data
extern crate image;

use super::{Classification, Matrix};
use image::DynamicImage;
use ndarray::Array;
use std::fs;

/// Take two lists of integral images and flatten them into a list of (img, label) tuples
fn flatten_to_classlist(
    integral_faces: Vec<Matrix>,
    integral_backgrounds: Vec<Matrix>,
) -> Vec<(Matrix, super::Classification)> {
    let mut out = Vec::with_capacity(integral_faces.len() + integral_backgrounds.len());

    for i in integral_faces {
        out.push((i, super::Classification::Face));
    }
    for i in integral_backgrounds {
        out.push((i, super::Classification::NonFace));
    }

    out
}

pub fn load_and_preprocess_data(
    faces_dir: &str,
    background_dir: &str,
) -> Vec<(Matrix, Classification)> {
    let faces = load_imgs_from_dir(faces_dir);
    let backgrounds = load_imgs_from_dir(background_dir);

    let integral_faces = compute_integral_images(faces);
    let integral_backgrounds = compute_integral_images(backgrounds);

    let flattened = flatten_to_classlist(integral_faces, integral_backgrounds);

    flattened
}

/// Load an opened training image into a matrix
fn training_img_as_matrix(img: DynamicImage) -> Matrix {
    // raw_pixels gives a flat vector of the form [r1,g1,b1,r2,g2,b2,...]
    let raw_pixels = img.raw_pixels();
    assert!(raw_pixels.len() == 64 * 64 * 3);

    let mut out_pixels: Vec<i64> = Vec::with_capacity(64 * 64);
    // Average over the colors (doing integer division)
    for i in 0..(64 * 64) {
        let start_ind = i * 3;
        let mut out_px = 0;
        out_px += raw_pixels[start_ind] / 3;
        out_px += raw_pixels[start_ind + 1] / 3;
        out_px += raw_pixels[start_ind + 2] / 3;

        out_pixels.push(out_px as i64);
    }

    let pixel_arr = Array::from_vec(out_pixels);

    pixel_arr
        .into_shape((64, 64))
        .expect("Failed to transform pixel array into matrix")
}

/// Load an opened test into a matrix
/// TODO abstract into function that works over all image dimensions
fn test_img_as_matrix(img: DynamicImage) -> Matrix {
    // raw_pixels is just a raw array of pixels, for some reason. Maybe there's something
    // in the jpg spec that indicates when an image isn't rgb.
    let raw_pixels: Vec<i64> = img.raw_pixels().iter().map(|x| *x as i64).collect();
    let max_pixel = raw_pixels.iter().cloned().fold(0, i64::max);
    assert!(max_pixel <= 255);

    let pixel_arr = Array::from_vec(raw_pixels);

    pixel_arr
        .into_shape((1600, 1280))
        .expect("Failed to transform pixel array into matrix")
}

/// Returns a vector of matrices loaded from the input directory
fn load_imgs_from_dir(dir_name: &str) -> Vec<Matrix> {
    let imgs = fs::read_dir(dir_name).expect("Data directory not found");

    let mut loaded: Vec<Matrix> = Vec::new();
    for img_path in imgs {
        let img_path = img_path
            .expect("Failed while computing a input file path")
            .path();

        let ext = img_path.extension();
        if let None = ext {
            println!(
                "Ignoring input file while loading data: {}",
                img_path.display()
            );
            continue;
        } else if "jpg" == ext.unwrap() {
            let img = image::open(img_path).expect("Failed to open image");
            loaded.push(training_img_as_matrix(img));
        }
    }

    loaded
}

/// Compute the integral image for a matrix. This is not done in place so that the
/// output can be zero-padded.
pub fn compute_integral_image(img: &Matrix) -> Matrix {
    let (w, h) = img.dim();
    let mut integral = Matrix::zeros((w + 1, h + 1));

    for row in 0..w {
        for col in 0..h {
            integral[[row + 1 as usize, col + 1 as usize]] = img[[row as usize, col as usize]];
        }
    }

    for row in 0..=w {
        for col in 1..=h {
            integral[[row as usize, col as usize]] += integral[[row as usize, col - 1 as usize]];
        }
    }
    for col in 0..=h {
        for row in 1..=w {
            integral[[row as usize, col as usize]] += integral[[row - 1 as usize, col as usize]];
        }
    }

    integral
}

/// Compute the integral images for a set of image matrices
fn compute_integral_images(imgs: Vec<Matrix>) -> Vec<Matrix> {
    // Unfortunately ndarray doesn't have something like np's cumsum yet
    let mut integral_imgs: Vec<Matrix> = Vec::with_capacity(imgs.len());
    for img in imgs.iter() {
        integral_imgs.push(compute_integral_image(img));
    }

    integral_imgs
}

/// Returns a set of integral images corresponding to windows in the test
/// image, and a top-right coordinate in the image.
pub fn load_test_image(test_img_path: &str) -> (Matrix, Vec<(usize, usize)>) {
    let test_img = image::open(test_img_path).expect("Failed to open test image");
    let test_img_mat = test_img_as_matrix(test_img);
    assert!((1600, 1280) == test_img_mat.dim());

    let test_integral = compute_integral_image(&test_img_mat);
    let sliding_coords = get_sliding_window_coords(1600, 1280, 64, 1);

    (test_integral, sliding_coords)
}

/// Compute the top-left coordinates of a square window sliding over a space rectangle
/// of dimensions (xmax, ymax).
fn get_sliding_window_coords(xmax: usize, ymax: usize, window_side_len: usize, stride: usize) -> Vec<(usize, usize)> {
    let mut coords = Vec::new();
    for x in (0..xmax).step_by(stride) {
        for y in (0..ymax).step_by(stride) {
            if x + window_side_len < xmax && y + window_side_len < ymax {
                coords.push((x, y));
            }
        }
    }

    coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImage, Rgba};

    #[test]
    // Builds a purely red (255, 0, 0) 64x64 input and checks that it's
    // correctly turned into the corresponding grayscale matrix
    fn image_averages_correctly() {
        let (w, h) = (64, 64);
        let mut img = image::DynamicImage::new_rgb8(w, h);

        for x in 0..w {
            for y in 0..h {
                img.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }

        let mat = training_img_as_matrix(img);

        assert!(mat.ndim() == 2);
        assert!(mat.dim() == (64, 64));

        for x in 0..w {
            for y in 0..h {
                assert!(mat[[y as usize, x as usize]] == 255 / 3);
            }
        }
    }

    #[test]
    // Checks that the integral image is being computed correctly on a simple 4x4 example
    fn integral_images_computed_correctly() {
        let inp: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let exp: Vec<i64> = vec![
            0, 0, 0, 0, 0, 0, 1, 3, 6, 10, 0, 6, 14, 24, 36, 0, 15, 33, 54, 78, 0, 28, 60, 96, 136,
        ];

        let inp_mat = Array::from_vec(inp)
            .into_shape((4, 4))
            .expect("Failed to transform input array into matrix");
        let exp_mat = Array::from_vec(exp)
            .into_shape((5, 5))
            .expect("Failed to transform input array into matrix");

        let int_inp_mat = compute_integral_image(&inp_mat);

        assert!(int_inp_mat.dim() == (5, 5));
        assert!(int_inp_mat == exp_mat);

        let inp: Vec<i64> = vec![
            5, 4, 3, 2, 1, 4, 3, 2, 1, 5, 3, 2, 1, 5, 4, 2, 1, 5, 4, 3, 1, 5, 4, 3, 2,
        ];
        let exp: Vec<i64> = vec![
            0, 0, 0, 0, 0, 0, 0, 5, 9, 12, 14, 15, 0, 9, 16, 21, 24, 30, 0, 12, 21, 27, 35, 45, 0,
            14, 24, 35, 47, 60, 0, 15, 30, 45, 60, 75,
        ];

        let inp_mat = Array::from_vec(inp)
            .into_shape((5, 5))
            .expect("Failed to transform input array into matrix");
        let exp_mat = Array::from_vec(exp)
            .into_shape((6, 6))
            .expect("Failed to transform input array into matrix");

        let int_inp_mat = compute_integral_image(&inp_mat);

        assert!(int_inp_mat.dim() == (6, 6));
        assert!(int_inp_mat == exp_mat);
    }

    #[test]
    fn correct_sliding_windows_computed() {
        let xmax = 10;
        let ymax = 10;
        let stride = 3;
        let window_side_len = 4;

        let mut sliding_window_result = get_sliding_window_coords(xmax, ymax, stride, window_side_len);
        sliding_window_result.sort();
        let expected = vec![(0,0), (0,3), (3,0), (3,3)];

        assert!(sliding_window_result.len() == 4);
    }
}
