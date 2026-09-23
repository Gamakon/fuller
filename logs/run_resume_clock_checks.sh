#!/bin/zsh
# Full gate for fix/resume-clock: zero-warning test run + clippy, both with gpu.
cd /Users/andrewmorgan/Dev/gamakon/fuller
echo "=== RUSTFLAGS=-D warnings cargo test --features gpu ==="
RUSTFLAGS="-D warnings" cargo test --features gpu 2>&1
echo "=== EXIT test: $? ==="
echo "=== cargo clippy --all-targets --features gpu ==="
cargo clippy --all-targets --features gpu 2>&1
echo "=== EXIT clippy: $? ==="
