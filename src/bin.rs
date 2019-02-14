extern crate viola_jones;

fn main() {
    let mut learner = viola_jones::Learner::new("data/faces", "data/background", 20, 7);
    learner.train();
}
