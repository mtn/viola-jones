extern crate viola_jones;

fn main() {
    let (mut faces, mut backgrounds) =
        viola_jones::preprocess::load_and_preprocess_data("data/faces", "data/background");
}

#[cfg(test)]
mod tests {}
