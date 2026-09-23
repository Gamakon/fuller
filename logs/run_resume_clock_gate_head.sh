#!/bin/zsh
# Gate for fix/resume-clock at HEAD d675797, in a worktree beside the repo (so
# the ../hff path dependency resolves) and isolated from a concurrent agent's
# edits to the main tree.
cd /Users/andrewmorgan/Dev/gamakon/fuller-gate-wt
echo "=== HEAD ==="; git rev-parse HEAD
echo "=== RUSTFLAGS=-D warnings cargo test --features gpu ==="
RUSTFLAGS="-D warnings" cargo test --features gpu 2>&1
echo "=== EXIT test: $? ==="
echo "=== cargo clippy --all-targets --features gpu ==="
cargo clippy --all-targets --features gpu 2>&1
echo "=== EXIT clippy: $? ==="
