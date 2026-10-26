#!/usr/bin/env python3
"""Extract the WGSL modules Futhark's WebGPU backend embeds in its generated C.

`futhark webgpu prog.fut --library -o out` writes `out.c`, `out.json` and
`out.wrapper.js`.  The WGSL is not a file of its own: it is C string literals
inside `out.c`, one array per module (the runtime prelude, then one per
generated kernel).  This reads them back out so they can be handed to a
validator (naga) or a driver.

The PR's `futhark dev --compile-imp-webgpu` writes the same WGSL to stdout,
but it wants a GPUMem IR file rather than a `.fut` source, so the extraction
has to come from the C for the ordinary path.

    python3 tools/extract_futhark_wgsl.py out.c outdir

Writes `outdir/<name>.wgsl` for every module it finds, and prints a summary.
"""

import os
import re
import sys

# `static const char *name[] = {` ... `};`
ARRAY_RE = re.compile(r"static\s+const\s+char\s*\*\s*(\w+)\s*\[\s*\]\s*=\s*\{", re.MULTILINE)

SIMPLE_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\",
    '"': '"', "'": "'", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "?": "?",
}


def unescape(literal: str) -> str:
    """The text of a C string literal, with the escapes resolved."""
    out = []
    i = 0
    n = len(literal)
    while i < n:
        c = literal[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        e = literal[i]
        if e in SIMPLE_ESCAPES:
            out.append(SIMPLE_ESCAPES[e])
            i += 1
        elif e == "x":
            j = i + 1
            digits = ""
            while j < n and len(digits) < 2 and literal[j] in "0123456789abcdefABCDEF":
                digits += literal[j]
                j += 1
            if digits:
                out.append(chr(int(digits, 16)))
                i = j
            else:
                out.append("x")
                i += 1
        elif e in "01234567":
            j = i
            digits = ""
            while j < n and len(digits) < 3 and literal[j] in "01234567":
                digits += literal[j]
                j += 1
            out.append(chr(int(digits, 8)))
            i = j
        else:
            out.append(e)
            i += 1
    return "".join(out)


def string_literals(text: str):
    """Every C string literal in `text`, in order, with its quotes removed."""
    i = 0
    n = len(text)
    while i < n:
        if text[i] != '"':
            i += 1
            continue
        start = i + 1
        i = start
        buf = []
        while i < n:
            c = text[i]
            if c == "\\" and i + 1 < n:
                buf.append(c)
                buf.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                break
            buf.append(c)
            i += 1
        yield unescape("".join(buf))
        i += 1


def array_body(text: str, start: int) -> str:
    """The body of the C array that starts at `start`, up to its `NULL};`.

    The naive "find the next `};`" is wrong: the WGSL itself contains `};`
    (Futhark's codegen emits a stray empty statement after a block), so the
    scan has to skip over the string literals and stop at the array's own
    terminator.
    """
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if text.startswith("NULL", i):
            j = text.find("};", i)
            if j >= 0:
                return text[start:i]
        i += 1
    return text[start:]


def modules(text: str):
    """(name, wgsl) for every `static const char *name[] = {...}` block."""
    for m in ARRAY_RE.finditer(text):
        name = m.group(1)
        body = array_body(text, m.end())
        wgsl = "".join(string_literals(body))
        if wgsl:
            yield name, wgsl


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, outdir = argv[1], argv[2]
    text = open(src, encoding="utf-8", errors="replace").read()
    os.makedirs(outdir, exist_ok=True)
    total = 0
    for name, wgsl in modules(text):
        path = os.path.join(outdir, f"{name}.wgsl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(wgsl)
        kernels = wgsl.count("@compute")
        total += 1
        print(f"{path}: {len(wgsl)} bytes, {kernels} compute entry point(s)")
    print(f"{total} module(s) from {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
