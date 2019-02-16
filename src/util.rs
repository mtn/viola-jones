/// General utility functions
type MatrixView<'a> = ndarray::ArrayView2<'a, i64>;

#[derive(Debug)]
pub struct Rectangle {
    pub xmin: usize,
    pub xmax: usize,
    pub ymin: usize,
    pub ymax: usize,
}

impl Rectangle {
    pub fn new(p1: (usize, usize), p2: (usize, usize)) -> Rectangle {
        assert!(p1.0 <= 64 && p1.1 <= 64 && p2.0 <= 64 && p2.1 <= 64);
        assert!(p1.0 <= p2.0);
        assert!(p1.1 <= p2.1);

        Rectangle {
            xmin: p1.0,
            xmax: p2.0,
            ymin: p1.1,
            ymax: p2.1,
        }
    }
}

/// Compute the area of a block within an (assumed) padded integral image
pub fn compute_area(img: &MatrixView, r: &Rectangle) -> i64 {
    img[[r.ymax, r.xmax]] + img[[r.ymin, r.xmin]] - img[[r.ymin, r.xmax]] - img[[r.ymax, r.xmin]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    // Checks that sums are computed correctly for integral images
    fn areas_computed_correctly() {
        // Integral img from [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        let inp: Vec<i64> = vec![
            0, 0, 0, 0, 0, 0, 1, 3, 6, 10, 0, 6, 14, 24, 36, 0, 15, 33, 54, 78, 0, 28, 60, 96, 136,
        ];

        let img = Array::from_vec(inp)
            .into_shape((5, 5))
            .expect("Failed to transform input array into matrix");

        assert!(compute_area(&img.view(), &Rectangle::new((2, 2), (2, 2))) == 0);
        assert!(compute_area(&img.view(), &Rectangle::new((0, 0), (2, 2))) == 14);
        assert!(compute_area(&img.view(), &Rectangle::new((0, 0), (4, 4))) == 136);
        assert!(compute_area(&img.view(), &Rectangle::new((1, 1), (4, 4))) == 99);
        assert!(compute_area(&img.view(), &Rectangle::new((1, 1), (2, 2))) == 6);
        assert!(compute_area(&img.view(), &Rectangle::new((1, 1), (3, 3))) == 34);
    }
}
