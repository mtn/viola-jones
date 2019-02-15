extern crate viola_jones;

fn main() {
    viola_jones::Learner::test_cascade("data/test_img.jpg", "saved_cascade.json");

    // let mut learner = viola_jones::Learner::new("data/faces", "data/background", 4);
    // learner.train();
}
