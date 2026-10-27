#!/bin/bash
# S5 of docs/PLAN_futhark_wgpu_host.md: the same program through Futhark's C
# backend and through the webgpu backend + this shim, on the same inputs,
# compared bit for bit — on both of the runtime's wait paths.
#
#   tests/s5/run.sh <workdir> [program.fut ...]   (default: every *.fut here)
#
# Each program's first line names its inputs as `futhark dataset` generators
# (`-- input: -g [100000]f32 ...`); the data are seeded, so a rerun sees the
# same inputs. The GPU runs are guarded by a 120 s alarm each and take the
# GPU lock (/private/tmp/claude-501/locks/gpu) for the duration of the runs.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
shim=$(cd "$here/../.." && pwd)
futhark=${FUTHARK_BRANCH:-/tmp/branch_artifact/futhark-nightly-macos-arm64/bin/futhark}
work=$1
shift
mkdir -p "$work"
progs=("$@")
[ ${#progs[@]} -eq 0 ] && progs=("$here"/*.fut)
guard() { perl -e 'alarm shift; exec @ARGV' "$@"; }
gpu=/private/tmp/claude-501/locks/gpu

for p in "${progs[@]}"; do
  name=$(basename "${p%.fut}")
  d="$work/$name"
  mkdir -p "$d"
  cp "$p" "$d/prog.fut"
  gens=$(head -1 "$p" | sed -n 's/^-- input: //p')
  echo "== $name ($(date -u +%H:%M:%SZ))"
  (cd "$d" && "$futhark" dataset -b -s 20261027 --f32-bounds=-100:100 --i32-bounds=-100000:100000 $gens >in.bin) || { echo "  dataset: failed"; continue; }
  if ! (cd "$d" && guard 300 "$futhark" c prog.fut -o prog_c >c_compile.log 2>&1); then
    echo "  C backend: the compile failed:"; sed 's/^/    /' "$d/c_compile.log" | head -5; continue
  fi
  (cd "$d" && guard 300 "$futhark" webgpu prog.fut -o prog_w >w_compile.log 2>&1)
  if [ ! -s "$d/prog_w.c" ]; then
    echo "  webgpu backend: no C written:"; sed 's/^/    /' "$d/w_compile.log" | head -5; continue
  fi
  ok=1
  "$shim/build_futhark.sh" "$d/prog_w.c" "$d/prog_wgpu" >"$d/build.log" 2>&1 || ok=0
  "$shim/build_futhark.sh" --dawn "$d/prog_w.c" "$d/prog_wgpu_dawn" >>"$d/build.log" 2>&1 || ok=0
  if [ $ok -eq 0 ]; then echo "  link: failed"; sed 's/^/    /' "$d/build.log" | tail -5; continue; fi
  echo "  port: $(tail -1 "$d/prog_wgpu.port.log" | sed 's/.*: //')"
  guard 120 "$d/prog_c" -b <"$d/in.bin" >"$d/c.out" 2>"$d/c.err" || { echo "  C run: failed"; continue; }
  until mkdir "$gpu" 2>/dev/null; do sleep 2; done
  for v in prog_wgpu prog_wgpu_dawn; do
    FUTHARK_WGPU_SHIM_TIMESTAMPS=1 FUTHARK_WGPU_SHIM_REPORT="$d/$v.report.json" \
      guard 120 "$d/$v" -b <"$d/in.bin" >"$d/$v.out" 2>"$d/$v.err"
    echo "    $v exit $?"
  done
  rmdir "$gpu"
  for v in prog_wgpu prog_wgpu_dawn; do
    if [ -s "$d/$v.out" ]; then
      printf '  %-15s vs C: %s\n' "$v" "$(python3 -I "$shim/tools/compare_futhark_values.py" "$d/c.out" "$d/$v.out")"
      python3 -I -c '
import json, sys
r = json.load(open(sys.argv[1]))
p = r["passes"]
good = sum(1 for x in p if x["end_tick"] > x["begin_tick"] > 0)
print("    %d dispatches, %d submits; %d of %d passes timed on the GPU, %.1f us; emscripten_sleep %d, WaitAny %d" % (r["dispatches"], r["submits"], good, len(p), r["gpu_ns_total"] / 1e3, r["emscripten_sleep_calls"], r["wait_any_calls"]))
' "$d/$v.report.json"
    else
      echo "  $v: no output:"; sed 's/^/    /' "$d/$v.err" | head -8
    fi
  done
done
