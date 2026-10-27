#!/bin/sh
# S1 of docs/PLAN_futhark_wgpu_host.md: the shim's surface, taken from the
# GENERATED artefact, not from the runtime's source.
#
# The branch compiler writes probe.c (Futhark's host runtime with the
# program's kernels embedded). The runtime has two wait paths, chosen by the
# preprocessor: the default one (Emscripten's: callbacks may fire on their own,
# and the runtime spins on emscripten_sleep(0)) and the USE_DAWN one (futures
# and wgpuInstanceWaitAny). unifdef resolves each path, and the identifiers
# left in the code are the surface of that path.
#
# Writes, next to this script:
#   wgpu_functions.default.txt   the wgpu* functions of the default path
#   wgpu_functions.dawn.txt      the wgpu* functions of the USE_DAWN path
#   wgpu_functions.txt           their union: the shim's exported functions
#   wgpu_types.txt               every WGPU* type and constant the code names
#   emscripten_symbols.txt       what the default path needs from emscripten.h
#
# Usage: extract_surface.sh [path/to/futhark]   (default: the pinned artefact)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
futhark=${1:-/tmp/branch_artifact/futhark-nightly-macos-arm64/bin/futhark}
work=$(mktemp -d)
trap 'rm -r "$work"' EXIT

cp "$here/probe.fut" "$work/probe.fut"
# The compile writes probe.c and then fails at the emcc step (no Emscripten
# here, by design). The C file is what we want; its absence is the failure.
"$futhark" webgpu "$work/probe.fut" -o "$work/probe" >"$work/compile.log" 2>&1 || true
if [ ! -s "$work/probe.c" ]; then
  echo "the branch compiler wrote no probe.c:" >&2
  cat "$work/compile.log" >&2
  exit 1
fi

# unifdef exits 1 when it changed something; only 2 is an error.
resolve() { unifdef "$1" "$work/probe.c" >"$2" || [ $? -eq 1 ]; }
resolve -UUSE_DAWN "$work/default.c"
resolve -DUSE_DAWN "$work/dawn.c"

# Comments name functions too ("based on looping with emscripten_sleep"), so
# they are stripped first, and string literals are blanked: the embedded WGSL
# is C strings, and it is shader code, not host surface.
strip() { python3 -I "$here/strip_c.py" "$1"; }
strip "$work/default.c" >"$work/default.nc.c"
strip "$work/dawn.c" >"$work/dawn.nc.c"

fns() { grep -oE '\bwgpu[A-Z][A-Za-z0-9]*' "$1" | sort -u; }
fns "$work/default.nc.c" >"$here/wgpu_functions.default.txt"
fns "$work/dawn.nc.c" >"$here/wgpu_functions.dawn.txt"
sort -u "$here/wgpu_functions.default.txt" "$here/wgpu_functions.dawn.txt" >"$here/wgpu_functions.txt"
cat "$work/default.nc.c" "$work/dawn.nc.c" | grep -oE '\bWGPU[A-Za-z0-9_]*' | sort -u >"$here/wgpu_types.txt"
grep -oE '\bemscripten_[a-z_]+|\bEM_[A-Z_]+\b|\bEMSCRIPTEN_[A-Z_]+' "$work/default.nc.c" | sort -u >"$here/emscripten_symbols.txt" || true

printf 'default path: %s functions\n' "$(wc -l <"$here/wgpu_functions.default.txt" | tr -d ' ')"
printf 'dawn path:    %s functions\n' "$(wc -l <"$here/wgpu_functions.dawn.txt" | tr -d ' ')"
printf 'union:        %s functions\n' "$(wc -l <"$here/wgpu_functions.txt" | tr -d ' ')"
printf 'types:        %s WGPU* names\n' "$(wc -l <"$here/wgpu_types.txt" | tr -d ' ')"
printf 'emscripten:   %s\n' "$(tr '\n' ' ' <"$here/emscripten_symbols.txt")"
