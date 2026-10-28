#!/bin/bash
# Build the runtime ONCE as the dylib a run's in-process host and every generation's module share, and copy it, with
# what rustc needs to compile a generation against it, into <dir> (a run's own copy: a later rebuild never changes the
# runtime under a running fit):
#
#   prebuild_rt.sh <dir>
#
# <dir>/libfuthark_rust_rt.dylib   the runtime, its install name @rpath/libfuthark_rust_rt.dylib, an rpath to the
#                                  toolchain's libstd (built with -C prefer-dynamic: the runtime and every generation
#                                  share the toolchain's libstd dylib, so std, wgpu and the runtime are one copy each)
# <dir>/deps/                      the dependencies' rlibs and proc-macro dylibs (rustc's -L dependency=)
# <dir>/rustc_flags                the flags a generation is compiled with
#
# The build runs under /private/tmp/claude-501/locks/cargo, held only while cargo runs (CARGO_LOCK_HELD as in the
# shim's scripts), with -D warnings.
set -euo pipefail
# Xcode's install_name_tool and codesign, ahead of any other on the PATH.
export PATH=/usr/bin:$PATH
here=$(cd "$(dirname "$0")" && pwd)
if [ $# -ne 1 ]; then
  sed -n '2,15p' "$0" >&2
  exit 2
fi
dir=$1
mkdir -p "$dir/deps"
lock=/private/tmp/claude-501/locks/cargo
if [ "${CARGO_LOCK_HELD:-}" != "$lock" ]; then
  until mkdir "$lock" 2>/dev/null; do sleep 2; done
  trap 'rmdir "$lock" 2>/dev/null || true' EXIT
fi
(cd "$here" && RUSTFLAGS="-D warnings -C prefer-dynamic" CARGO_TARGET_DIR="$here/target-dylib" nice -n 10 cargo build --release --lib)
if [ "${CARGO_LOCK_HELD:-}" != "$lock" ]; then
  rmdir "$lock"
  trap - EXIT
fi
deps="$here/target-dylib/release/deps"
sys=$(rustc --print sysroot)/lib
cp "$deps"/*.rlib "$dir/deps/"
for f in "$deps"/*.dylib; do
  [ "$(basename "$f")" = libfuthark_rust_rt.dylib ] || cp "$f" "$dir/deps/"
done
cp "$deps/libfuthark_rust_rt.dylib" "$dir/libfuthark_rust_rt.dylib"
install_name_tool -id @rpath/libfuthark_rust_rt.dylib -add_rpath "$sys" "$dir/libfuthark_rust_rt.dylib"
codesign -f -s - "$dir/libfuthark_rust_rt.dylib" 2>/dev/null
printf '%s\n' "-C prefer-dynamic" "-L dependency=$dir/deps" "--extern futhark_rust_rt=$dir/libfuthark_rust_rt.dylib" "-C link-args=-Wl,-rpath,$dir" "-C link-args=-Wl,-rpath,$sys" > "$dir/rustc_flags"
echo "prebuilt the runtime into $dir"
