#!/bin/sh
# Re-fetch the pinned Futhark webgpu compiler (a CI artefact, not a release)
# and check it is the binary this shim was built and tested against.
#
#   ./fetch_futhark_branch.sh [destination]   (default: /tmp/branch_artifact)
#
# Needs the GitHub CLI (`gh`), logged in. See README.md for the pin.
set -eu
dest=${1:-/tmp/branch_artifact}
run=39795721627
name=futhark-nightly-macos-arm64
want=3d54dcbe3d79aecce49fb7cbcc64c14c8bf026a3dfba6507533b2294388ea4e6

mkdir -p "$dest"
cd "$dest"
if [ ! -f "$name.tar.xz" ]; then
  gh run download "$run" -R diku-dk/futhark -n "$name" -D .
fi
tar -xJf "$name.tar.xz"
got=$(shasum -a 256 "$name/bin/futhark" | cut -d' ' -f1)
if [ "$got" != "$want" ]; then
  echo "the fetched compiler is not the pinned one: sha256 $got, want $want" >&2
  exit 1
fi
"$name/bin/futhark" --version | head -1
echo "ok: $dest/$name/bin/futhark"
