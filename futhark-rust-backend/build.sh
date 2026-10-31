#!/bin/bash
# Make the Futhark fork with the rust-wgpu backend: fetch Futhark's webgpu branch at the pinned commit, apply
# patches/ in order, build with cabal (GHC 9.14.1), install bin/futhark.
#
#   build.sh [dir]     (default: $HOME/.local/share/fuller/futhark-rust, the path fuller's default compiler points at:
#                       <dir>/bin/futhark, fuller::futhark_rust::compiler)
#
# An existing clone is brought up to the series: its commits since the pinned base must be the first patches of
# patches/ (by patch id, in order), and the rest are applied. A clone that went another way is a hard stop.
# The installed compiler's series is copied to <dir>/bin/patches/, so fuller's tests can check that the default
# compiler is built from exactly the committed series (a stale build fails them).
#
# Nothing is pushed anywhere: the clone has its remote removed once fetched.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=${1:-$HOME/.local/share/fuller/futhark-rust}
base=14e9ccbf3a47c06ed585d15e6827d186dd74d8a0
if [ ! -d "$dir/.git" ]; then
  mkdir -p "$(dirname "$dir")"
  git clone --branch webgpu --single-branch https://github.com/diku-dk/futhark.git "$dir"
  git -C "$dir" remote remove origin
  git -C "$dir" checkout -q -b rust-wgpu "$base"
fi
if [ "$(git -C "$dir" merge-base HEAD "$base")" != "$base" ]; then
  echo "build.sh: $dir is not on the pinned base $base" >&2
  exit 1
fi
if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
  echo "build.sh: $dir has uncommitted changes" >&2
  exit 1
fi
patches=("$here"/patches/*.patch)
have=$(git -C "$dir" rev-list --count "$base"..HEAD)
if [ "$have" -gt "${#patches[@]}" ]; then
  echo "build.sh: $dir has $have commits on the base, the series ${#patches[@]}" >&2
  exit 1
fi
# The clone's commits must BE the series' first patches: compared by git's patch id (the diff, not the message).
i=0
while read -r commit; do
  got=$(git -C "$dir" show "$commit" | git patch-id --stable | cut -d' ' -f1)
  want=$(git patch-id --stable < "${patches[$i]}" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "build.sh: $dir commit $((i + 1)) ($commit) is not ${patches[$i]}" >&2
    exit 1
  fi
  i=$((i + 1))
done < <(git -C "$dir" rev-list --reverse "$base"..HEAD)
if [ "$have" -lt "${#patches[@]}" ]; then
  echo "build.sh: applying patches $((have + 1))..${#patches[@]} to $dir"
  git -C "$dir" -c user.name=futhark-rust-backend -c user.email=none am -q "${patches[@]:$have}"
fi
"$dir/build_rust_wgpu.sh"
rm -f "$dir"/bin/patches/*.patch
mkdir -p "$dir/bin/patches"
cp "${patches[@]}" "$dir/bin/patches/"
echo "build.sh: $dir/bin/futhark is the series of ${#patches[@]} patches"
