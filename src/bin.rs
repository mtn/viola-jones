extern crate viola_jones;

fn main() {
    let (faces, backgrounds) = viola_jones::load_data("data/faces", "data/background");
}

#[cfg(test)]
mod tests {}
