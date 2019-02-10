/// Functions for loading the pre-processing data
extern crate image;

use super::Matrix;
use image::DynamicImage;
use ndarray::Array;
use std::fs;

/// Load input faces and backgrounds into matrices
pub fn load_and_preprocess_data(
    faces_dir: &str,
    background_dir: &str,
) -> (Vec<Matrix>, Vec<Matrix>) {
    let faces = load_imgs_from_dir(faces_dir);
    let backgrounds = load_imgs_from_dir(background_dir);

    let integral_faces = compute_integral_images(faces);
    let integral_backgrounds = compute_integral_images(backgrounds);

    (integral_faces, integral_backgrounds)
}

/// Load an opened image into a matrix
fn img_as_matrix(img: DynamicImage) -> Matrix {
    // raw_pixels gives a flat vector of the form [r1,g1,b1,r2,g2,b2,...]
    let raw_pixels = img.raw_pixels();
    assert!(raw_pixels.len() == 64 * 64 * 3);

    let mut out_pixels: Vec<u32> = Vec::with_capacity(64 * 64);
    // Average over the colors (doing integer division)
    for i in 0..(64 * 64) {
        let start_ind = i * 3;
        let mut out_px = 0;
        out_px += raw_pixels[start_ind] / 3;
        out_px += raw_pixels[start_ind + 1] / 3;
        out_px += raw_pixels[start_ind + 2] / 3;

        out_pixels.push(out_px as u32);
    }

    let pixel_arr = Array::from_vec(out_pixels);

    pixel_arr
        .into_shape((64, 64))
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
            loaded.push(img_as_matrix(img));
        }
    }

    loaded
}

/// Compute the integral image for a matrix. This is not done in place so that the
/// output can be zero-padded.
fn compute_integral_image(img: &Matrix) -> Matrix {
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImage, GenericImageView, Rgba};

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

        let mat = img_as_matrix(img);

        assert!(mat.ndim() == 2);
        assert!(mat.dim() == (64, 64));

        for x in 0..w {
            for y in 0..h {
                assert!(mat[[x as usize, y as usize]] == 255 / 3);
            }
        }
    }

    #[test]
    // Checks that the integral image is being computed correctly on a simple 4x4 example
    fn integral_images_computed_correctly() {
        let inp: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let exp: Vec<u32> = vec![
            0, 0, 0, 0, 0, 0, 1, 3, 6, 10, 0, 6, 14, 24, 36, 0, 15, 33, 54, 78, 0, 28, 60, 96, 136,
        ];

        let mut inp_mat = Array::from_vec(inp)
            .into_shape((4, 4))
            .expect("Failed to transform input array into matrix");
        let exp_mat = Array::from_vec(exp)
            .into_shape((5, 5))
            .expect("Failed to transform input array into matrix");

        let int_inp_mat = compute_integral_image(&inp_mat);

        println!("{:?}", int_inp_mat);

        assert!(int_inp_mat.dim() == (5, 5));
        assert!(int_inp_mat == exp_mat);
    }
}
