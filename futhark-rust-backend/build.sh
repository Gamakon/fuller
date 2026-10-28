#!/bin/bash
# Make the Futhark fork with the rust-wgpu backend from scratch: fetch Futhark's webgpu branch at the pinned commit,
# apply patches/ in order, build with cabal (GHC 9.14.1), install bin/futhark.
#
#   build.sh [dir]     (default: /private/tmp/claude-501/futhark-rust, the path fuller's FUTHARK_RUST_WGPU default
#                       points at: <dir>/bin/futhark)
#
# Nothing is pushed anywhere: the clone has its remote removed once fetched.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=${1:-/private/tmp/claude-501/futhark-rust}
base=14e9ccbf3a47c06ed585d15e6827d186dd74d8a0
if [ ! -d "$dir/.git" ]; then
  git clone --branch webgpu --single-branch https://github.com/diku-dk/futhark.git "$dir"
  git -C "$dir" remote remove origin
  git -C "$dir" checkout -q -b rust-wgpu "$base"
  git -C "$dir" -c user.name=futhark-rust-backend -c user.email=none am -q "$here"/patches/*.patch
fi
if [ "$(git -C "$dir" merge-base HEAD "$base")" != "$base" ]; then
  echo "build.sh: $dir is not on the pinned base $base" >&2
  exit 1
fi
"$dir/build_rust_wgpu.sh"
