#!/bin/bash
# Build the shim ONCE and copy its static library and generated header into <dir>, for links that must not run cargo:
#
#   prebuild_shim.sh <dir>
#   FUTHARK_SHIM_PREBUILT=<dir> build_futhark.sh <generated.c> <exe>     # plain cc, no cargo, no lock
#
# The build runs under /private/tmp/claude-501/locks/cargo (held only while cargo runs; CARGO_LOCK_HELD as in
# build_futhark.sh) and the header is checked against the committed include/webgpu/webgpu.h, as build_futhark.sh does.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [ $# -ne 1 ]; then
  sed -n '2,8p' "$0" >&2
  exit 2
fi
dir=$1
mkdir -p "$dir/include/webgpu"
lock=/private/tmp/claude-501/locks/cargo
json=$(mktemp)
if [ "${CARGO_LOCK_HELD:-}" != "$lock" ]; then
  until mkdir "$lock" 2>/dev/null; do sleep 2; done
  trap 'rmdir "$lock" 2>/dev/null || true; rm -f "$json"' EXIT
fi
(cd "$here" && RUSTFLAGS="-D warnings" cargo build --release --message-format=json-render-diagnostics) >"$json"
if [ "${CARGO_LOCK_HELD:-}" != "$lock" ]; then
  rmdir "$lock"
  trap 'rm -f "$json"' EXIT
fi
read -r out_dir lib < <(python3 -I -c '
import json, sys
out_dir = lib = ""
for line in open(sys.argv[1]):
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if "futhark-wgpu-shim" not in m.get("package_id", ""):
        continue
    if m.get("reason") == "build-script-executed":
        out_dir = m["out_dir"]
    if m.get("reason") == "compiler-artifact":
        for f in m.get("filenames", []):
            if f.endswith(".a"):
                lib = f
print(out_dir, lib)
' "$json")
if [ -z "${out_dir:-}" ] || [ -z "${lib:-}" ]; then
  echo "prebuild_shim.sh: cargo reported no header directory or static library for the shim" >&2
  exit 1
fi
if ! cmp -s "$out_dir/include/webgpu/webgpu.h" "$here/include/webgpu/webgpu.h"; then
  echo "prebuild_shim.sh: include/webgpu/webgpu.h is not the generated header" >&2
  exit 1
fi
cp "$lib" "$dir/libfuthark_wgpu_shim.a"
cp "$out_dir/include/webgpu/webgpu.h" "$dir/include/webgpu/webgpu.h"
echo "prebuilt the shim into $dir"
