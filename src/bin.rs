extern crate viola_jones;

fn main() {
    let mut learner = viola_jones::Learner::new("data/faces", "data/background", 10);
    learner.train();
}
