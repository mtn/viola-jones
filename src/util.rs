/// General utility functions
use super::Matrix;

#[derive(Debug)]
pub struct Rectangle {
    pub xmin: usize,
    pub xmax: usize,
    pub ymin: usize,
    pub ymax: usize,
}

impl Rectangle {
    pub fn new(p1: (usize, usize), p2: (usize, usize)) -> Rectangle {
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
pub fn compute_area(img: &Matrix, r: &Rectangle) -> u32 {
    if r.xmax - r.xmin == 0 || r.ymax - r.ymin == 0 {
        return 0;
    }

    // println!("{} {} {} {}", img[[x+w, y+h]] , img[[x, y]] , img[[x, y+h]] , img[[x+w, y]]);
    println!("hi");
    println!("r is {:?}", r);
    println!(
        "{} {} {} {}",
        img[[r.xmax, r.ymax]],
        img[[r.xmin, r.ymin]],
        img[[r.xmin, r.ymax]],
        img[[r.xmax, r.ymin]]
    );
    img[[r.xmax, r.ymax]] + img[[r.xmin, r.ymin]] - img[[r.xmin, r.ymax]] - img[[r.xmax, r.ymin]]
    // img[[x+w, y+h]] + img[[x, y]] - img[[x, y+h]] - img[[x+w, y]]
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

        let r = Rectangle::new((0, 0), (1, 1));
        println!("{}", compute_area(&img, &r));
        assert!(false);
        // assert!(compute_area(&img, &Rectangle::new(0, 0, 2, 2)) == 10);
    }
}
