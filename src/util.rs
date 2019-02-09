/// General utility functions

use super::Matrix;
use std::cmp;

/// Compute the area of a block within an (assumed) integral image
pub fn compute_area(img: &Matrix, x: usize, y: usize, w: usize, h: usize) -> u32 {
    if w == 0 || h == 0 {
        return 0
    }

    // Width and height are 1-indexed
    // let w = w - 1;
    // let h = h - 1;

    // println!("{} {} {} {}", img[[x+w, y+h]] , img[[x, y]] , img[[x, y+h]] , img[[x+w, y]]);
    img[[x+w, y+h]] + img[[x, y]] - img[[x, y+h]] - img[[x+w, y]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    // Checks that sums are computed correctly for integral images
    fn areas_computed_correctly() {
        // Integral img from [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        let inp: Vec<u32> = vec![1, 3, 6, 10, 6, 14, 24, 36, 15, 33, 54, 78, 28, 60, 96, 136];

        let mut img = Array::from_vec(inp)
            .into_shape((4, 4))
            .expect("Failed to transform input array into matrix");

        println!("{:?}", img);
        println!("{}", compute_area(&img, 0, 0, 2, 2));
        assert!(compute_area(&img, 0, 0, 2, 2) == 10);
    }
}
