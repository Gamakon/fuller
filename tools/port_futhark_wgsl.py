#!/usr/bin/env python3
"""Port Futhark's WebGPU WGSL runtime to naga/wgpu.

Futhark's `rts/wgsl/*.wgsl` (diku-dk/futhark PR #2140) does not validate under
naga as emitted. One construct is responsible: `bitcast` between a scalar and a
vector.

    fn futrts_from_bits16(a: i16) -> f16 { return bitcast<vec2<f16>>(a)[0]; }

naga lowers `bitcast<T>(e)` to `Expression::As { convert: None }`, which needs the
source and the destination to have the same number of components; the WGSL spec
only requires the same total bit width. So `bitcast<vec2<f16>>(a: i32)` -- legal
WGSL -- is rejected by naga.

The affected code is only the f16 bit-level helpers (the `to_bits16`/`from_bits16`
pair, the f16 infinity/NaN helpers, and the f16 atomics). The first version of
the kingdom uses f32/i32/u32 only, so this script DELETES the functions that use
the construct and leaves the rest of the runtime untouched. It prints exactly what
it removed, so the deletion is auditable.

Usage:  port_futhark_wgsl.py <out.wgsl> <file.wgsl> ...
"""
import re
import sys

# A function whose body mentions one of these is removed.
F16_BITCAST = re.compile(
    r"bitcast<\s*vec2\s*<\s*f16\s*>\s*>|bitcast<\s*(?:i32|u32)\s*>\s*\(\s*vec2\s*<\s*f16\s*>",
    re.IGNORECASE,
)

# naga allows a pointer ARGUMENT only in the `private` and `function` address
# spaces (naga/src/valid/function.rs:1843). Futhark's runtime helpers take
# `ptr<storage, ...>` and `ptr<workgroup, ...>` arguments -- the shape OpenCL C
# allows and WGSL does not -- so they are removed as well.
STORAGE_POINTER_ARG = re.compile(
    r"ptr\s*<\s*(?:storage|workgroup)\b",
    re.IGNORECASE,
)

# `enable f16;` is harmless on a device with SHADER_F16, and the f16 arithmetic
# (sin/cos of f16, f16 min/max) is fine in naga. Only the bit-level helpers go.


def split_functions(lines):
    """Yield (start, end) line spans of top-level `fn` definitions.

    Braces are counted from the first `{`, so a one-line function ends on its
    own line and a signature spread over several lines is not cut in half.
    """
    spans = []
    start = None
    depth = 0
    seen_brace = False
    for i, line in enumerate(lines):
        if start is None:
            if line.startswith("fn "):
                start = i
                depth = 0
                seen_brace = False
                depth += line.count("{") - line.count("}")
                seen_brace = seen_brace or "{" in line
                if seen_brace and depth <= 0:
                    spans.append((start, i + 1))
                    start = None
            continue
        depth += line.count("{") - line.count("}")
        seen_brace = seen_brace or "{" in line
        if seen_brace and depth <= 0:
            spans.append((start, i + 1))
            start = None
    if start is not None:
        spans.append((start, len(lines)))
    return spans


def port(files):
    out = []
    removed = []
    for path in files:
        lines = open(path).read().split("\n")
        drop = set()
        for start, end in split_functions(lines):
            body = "\n".join(lines[start:end])
            reason = None
            if F16_BITCAST.search(body):
                reason = "f16 scalar<->vector bitcast"
            elif STORAGE_POINTER_ARG.search(body.split("{", 1)[0]):
                reason = "pointer argument outside the function/private address space"
            if reason:
                removed.append((lines[start].strip(), reason))
                drop.update(range(start, end))
        kept = [l for i, l in enumerate(lines) if i not in drop]
        out.append("\n".join(kept))
    return "\n".join(out), removed


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    out_path, files = sys.argv[1], sys.argv[2:]
    text, removed = port(files)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path} ({len(text)} bytes)")
    print(f"removed {len(removed)} function(s):")
    for r, why in removed:
        print(f"  - {r[:110]}\n      ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
