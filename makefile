release:
	cargo build --release
	./target/release/detector

btrace:
	cargo build --release
	RUST_BACKTRACE=1 ./target/release/detector
