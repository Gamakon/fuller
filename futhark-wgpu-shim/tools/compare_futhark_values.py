#!/usr/bin/env python3
"""Compare two Futhark binary value streams (the `-b` output of a compiled
Futhark program) bit for bit.

    compare_futhark_values.py <a.out> <b.out>

Prints one line: the number of values, how many are bit-identical, and for
floats the largest absolute difference and the largest distance in units in
the last place. Exit status 0 when every bit agrees, 1 otherwise.
"""
import struct
import sys

TYPES = {b" f32": ("f", 4), b" f64": ("d", 8), b" i32": ("i", 4), b" i64": ("q", 8),
         b"  i8": ("b", 1), b" i16": ("h", 2), b" u32": ("I", 4), b"bool": ("?", 1)}


def values(path):
    d = open(path, "rb").read()
    out = []
    i = 0
    while i < len(d):
        if d[i:i + 1].isspace():
            i += 1
            continue
        if d[i:i + 1] != b"b" or d[i + 1] != 2:
            raise SystemExit(f"{path}: not a Futhark binary value at byte {i}")
        rank = d[i + 2]
        ty = d[i + 3:i + 7]
        code, size = TYPES[ty]
        i += 7
        dims = struct.unpack_from(f"<{rank}q", d, i)
        i += 8 * rank
        n = 1
        for x in dims:
            n *= x
        raw = d[i:i + n * size]
        i += n * size
        out.append((ty, dims, raw, code, size))
    return out


def ulps(a, b):
    ia = struct.unpack("<i", struct.pack("<f", a))[0]
    ib = struct.unpack("<i", struct.pack("<f", b))[0]
    ia = ia if ia >= 0 else -2147483648 - ia
    ib = ib if ib >= 0 else -2147483648 - ib
    return abs(ia - ib)


def main(argv):
    a, b = values(argv[1]), values(argv[2])
    if [(t, d) for t, d, *_ in a] != [(t, d) for t, d, *_ in b]:
        print(f"different shapes or types: {[(t, d) for t, d, *_ in a]} vs {[(t, d) for t, d, *_ in b]}")
        return 1
    total = same = 0
    max_abs = 0.0
    max_ulp = 0
    for (ty, dims, ra, code, size), (_, _, rb, _, _) in zip(a, b):
        n = len(ra) // size
        va = struct.unpack(f"<{n}{code}", ra)
        vb = struct.unpack(f"<{n}{code}", rb)
        for k in range(n):
            total += 1
            if ra[k * size:(k + 1) * size] == rb[k * size:(k + 1) * size]:
                same += 1
            elif code == "f":
                max_abs = max(max_abs, abs(va[k] - vb[k]))
                max_ulp = max(max_ulp, ulps(va[k], vb[k]))
            elif code in "dfq" or code in "iIhb":
                max_abs = max(max_abs, abs(va[k] - vb[k]))
    line = f"{total} values, {same} bit-identical, {total - same} differ"
    if same != total:
        line += f"; largest |difference| {max_abs:.9g}, largest ulp distance {max_ulp}"
    print(line)
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
