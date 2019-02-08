extern crate ndarray;
extern crate image;

use image::DynamicImage;
use ndarray::{Array2, Array};
use std::fs;

fn img_as_matrix(img: DynamicImage) -> Array2<u8> {
    // raw_pixels gives a flat vector of the form [r1,g1,b1,r2,g2,b2,...]
    let raw_pixels = img.raw_pixels();
    assert!(raw_pixels.len() == 64 * 64 * 3);

    let mut out_pixels: Vec<u8> = Vec::with_capacity(64*64);
    // Average over the colors (doing integer division)
    for i in 0..(64*64) {
        let start_ind = i * 3;
        let mut out_px = 0;
        out_px += raw_pixels[start_ind] / 3;
        out_px += raw_pixels[start_ind + 1] / 3;
        out_px += raw_pixels[start_ind + 2] / 3;

        out_pixels.push(out_px);
    }

    let pixel_arr = Array::from_vec(out_pixels);

    pixel_arr.into_shape((64, 64)).unwrap()
}

// pub fn load_data(faces_dir: &str, background_dir: &str) {
//     let faces_imgs = fs::read_dir(faces_dir).unwrap();
//     let backgrounds_imgs = fs::read_dir(background_dir).unwrap();

//     let mut faces: Vec<DMatrix<u8>> = Vec::new();
//     for face_img_path in faces_imgs {
//         let face_img_path = face_img_path.unwrap().path();
//         if "jpg" == face_img_path.extension().unwrap() {
//             let img = image::open(face_img_path).unwrap();
//             faces.push(img_as_matrix(img));
//         }
//     }
//     // println!("hi there {} {}", faces_dir, background_dir);

// }

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;
    use image::GenericImage;
    use image::Rgba;

    #[test]
    // Take in a purely red (204, 0, 0) 64x64 input and checks that it's
    // correctly turned into the corresponding grayscale matrix
    fn image_averages_correctly() {
        let (w, h) = (64, 64);
        let mut img = image::DynamicImage::new_rgb8(w, h);

        for x in 0..w {
            for y in 0..h {
                img.put_pixel(x, y, Rgba([255,0,0, 255]));
                let ppx = img.get_pixel(x, y);
            }
        }

        let mat = img_as_matrix(img);

        assert!(mat.ndim() == 2);
        assert!(mat.dim() == (64, 64));

        for x in 0..w {
            for y in 0..h {
                // 255 / 3 == 85
                assert!(mat[[x as usize, y as usize]] == 85);
            }
        }
    }
}
