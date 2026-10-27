#!/usr/bin/env python3
"""Port the WGSL embedded in Futhark's generated webgpu C so naga accepts it.

    port_futhark_c.py <in.c> <out.c>

Futhark's webgpu backend (diku-dk/futhark PR #2140) embeds every WGSL module
in the generated C as an array of string literals (`gpu_program`, and one
`<builtin>_gpu_program` per built-in kernel: the transposes and the copies).
Two constructs in them stop naga, the front end wgpu uses:

1. **Pointer arguments in the storage and workgroup address spaces.** The
   runtime's helpers (`read_i32`, `write_i8`, the atomics, ...) take
   `ptr<storage, array<T>, read_write>`. That is WGSL under the
   `unrestricted_pointer_parameters` language extension, which naga has not
   implemented (naga 30: `UnimplementedLanguageExtension::
   UnrestrictedPointerParameters`; its validator allows pointer arguments only
   in the function and private spaces). The built-in kernels call these
   helpers as `read_i32(&src_mem, i)`, with the address of a module-scope
   binding.

   The port **specialises** each such call: `read_i32(&src_mem, i)` becomes
   `read_i32__src_mem(i)`, a copy of the helper with the pointer argument
   removed and `*buffer` replaced by `src_mem`. Taking the address of a
   module-scope variable and dereferencing it inside the callee is the same as
   naming the variable, so the meaning is unchanged; a helper that passes its
   pointer on (`read_bool` calls `read_i8(buffer, i)`) is specialised in turn.
   The generated kernels also pass the address of one ELEMENT, as in
   `atomic_add_i32_global((&counters_mem[i]), 0, 1)`; that call becomes
   `atomic_add_i32_global__counters_mem__elem(i, 0, 1)`, a copy whose first
   argument is the index and which names `counters_mem[index]`. The index is
   evaluated once, at the call, as the address was.

   A call whose pointer argument is anything but `&name` or `&name[i]` is NOT
   ported: it is printed, and naga will then reject the module by name.

2. **Scalar-vector `bitcast` on f16** — a naga strictness, not invalid WGSL
   (`tools/port_futhark_wgsl.py`). The functions that use it are deleted, as
   that tool deletes them; a kernel that needs one fails in naga by name.

Every pointer helper left unspecialised is then deleted (no caller needs it).
The port prints, per module, every specialisation and every deletion, so what
it changed is auditable. Only the WGSL arrays are touched; the host C is
copied byte for byte.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

from extract_futhark_wgsl import ARRAY_RE, array_body, string_literals  # noqa: E402
from port_futhark_wgsl import F16_BITCAST, STORAGE_POINTER_ARG, split_functions  # noqa: E402

FN_NAME = re.compile(r"^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def signature(text):
    """The text of a function up to its body's opening brace."""
    return text.split("{", 1)[0]


def params(sig):
    """[(name, type)] of a function signature."""
    inner = sig[sig.index("(") + 1 : sig.rindex(")")]
    out = []
    depth = 0
    cur = ""
    for c in inner:
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    result = []
    for p in out:
        name, ty = p.split(":", 1)
        result.append((name.strip(), ty.strip()))
    return result


class Helper:
    def __init__(self, name, text):
        self.name = name
        self.text = text
        ps = params(signature(text))
        self.ptr_param = ps[0][0] if ps and STORAGE_POINTER_ARG.search(ps[0][1]) else None
        self.others = [p for p in ps[1:] if STORAGE_POINTER_ARG.search(p[1])]

    def specialise(self, target, element):
        """This helper with its pointer argument fixed.

        `element` False: the argument was `&target`, and the copy names
        `target` where the original dereferenced the pointer.
        `element` True: the argument was `&target[i]`; the copy takes `i` as
        its first argument (`INDEX`) and names `target[INDEX]`. The index is
        evaluated once, at the call, exactly as the address was.
        """
        p = self.ptr_param
        sig, body = self.text.split("{", 1)
        kept = params(sig)[1:]
        if element:
            kept = [(INDEX, "i32")] + kept
            place = f"{target}[{INDEX}]"
            name = f"{self.name}__{target}__elem"
        else:
            place = target
            name = f"{self.name}__{target}"
        tail = sig[sig.rindex(")") + 1 :]
        new_sig = f"fn {name}(" + ", ".join(f"{n}: {t}" for n, t in kept) + ")" + tail
        new_body = re.sub(r"\(\s*\*\s*" + re.escape(p) + r"\s*\)", place, body)
        new_body = re.sub(r"\*\s*" + re.escape(p) + r"\b", place, new_body)
        # The pointer itself — passed on to another helper (`read_i8(buffer,
        # i)`) or to a builtin (`atomicLoad(p)`).
        new_body = re.sub(r"(?<![\w.&*])" + re.escape(p) + r"\b", "&" + place, new_body)
        return new_sig + "{" + new_body


# The name of the index argument an element specialisation takes.
INDEX = "futhark_port_index"
IDENT = re.compile(r"^&\s*([A-Za-z_][A-Za-z0-9_]*)$")
ELEMENT = re.compile(r"^&\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]$", re.S)


def first_argument(text, i):
    """(the first argument's text, the index of the `,` or `)` ending it)."""
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c in "([":
            depth += 1
        elif c in ")]":
            if depth == 0:
                return text[i:j], j
            depth -= 1
        elif c == "," and depth == 0:
            return text[i:j], j
        j += 1
    raise SystemExit("an unterminated call in the WGSL")


def strip_parens(arg):
    """`((x))` -> `x`, only when the outer parentheses enclose the whole."""
    arg = arg.strip()
    while arg.startswith("(") and arg.endswith(")"):
        depth = 0
        whole = True
        for k, c in enumerate(arg):
            depth += c == "("
            depth -= c == ")"
            if depth == 0 and k != len(arg) - 1:
                whole = False
                break
        if not whole:
            break
        arg = arg[1:-1].strip()
    return arg


def balanced_index(arg):
    """For `&x[e]`: (`x`, `e`) when the final `]` closes the first `[`."""
    m = ELEMENT.match(arg)
    if not m:
        return None
    inner = m.group(2)
    depth = 0
    for c in inner:
        depth += c in "(["
        depth -= c in ")]"
        if depth < 0:
            return None
    return (m.group(1), inner) if depth == 0 else None


def port_module(wgsl):
    lines = wgsl.split("\n")
    spans = split_functions(lines)
    helpers = {}
    f16 = set()
    for start, end in spans:
        text = "\n".join(lines[start:end])
        m = FN_NAME.match(lines[start].strip())
        if not m:
            continue
        name = m.group(1)
        if F16_BITCAST.search(text):
            f16.add(name)
        elif STORAGE_POINTER_ARG.search(signature(text)):
            h = Helper(name, text)
            if h.ptr_param is None or h.others:
                raise SystemExit(f"{name}: a pointer argument that is not the first, or more than one; not ported")
            helpers[name] = h

    removed_spans = []
    for start, end in spans:
        m = FN_NAME.match(lines[start].strip())
        if m and (m.group(1) in helpers or m.group(1) in f16):
            removed_spans.append((start, end, m.group(1)))
    drop = set()
    for start, end, _ in removed_spans:
        drop.update(range(start, end))
    rest = "\n".join(l for i, l in enumerate(lines) if i not in drop)

    call = re.compile(r"\b(" + "|".join(map(re.escape, helpers)) + r")\(") if helpers else None
    made = {}
    order = []
    unported = []

    def rewrite(text):
        if call is None:
            return text
        out = []
        pos = 0
        while True:
            m = call.search(text, pos)
            if not m:
                out.append(text[pos:])
                return "".join(out)
            name = m.group(1)
            arg, end = first_argument(text, m.end())
            core = strip_parens(arg)
            ident = IDENT.match(core)
            elem = None if ident else balanced_index(core)
            if not ident and not elem:
                unported.append(text[m.start() : m.start() + 100].split("\n")[0])
                out.append(text[pos : m.end()])
                pos = m.end()
                continue
            target = ident.group(1) if ident else elem[0]
            key = (name, target, elem is not None)
            if key not in made:
                made[key] = None
                made[key] = rewrite(helpers[name].specialise(target, elem is not None))
                order.append(key)
            new_name = f"{name}__{target}" + ("__elem" if elem else "")
            out.append(text[pos : m.start()])
            if elem:
                out.append(f"{new_name}({rewrite(elem[1])}" + (", " if text[end] == "," else ")"))
            else:
                out.append(f"{new_name}(" + ("" if text[end] == "," else ")"))
            pos = end + 1

    rest = rewrite(rest)
    specs = "\n".join(made[k] for k in order)
    # WGSL's module-scope declarations may come in any order, so the
    # specialised helpers simply follow the rest of the module.
    ported = rest + ("\n" + specs + "\n" if specs else "")
    return ported, [n for _, _, n in removed_spans], order, unported


def c_literal(text):
    """`text` as C string literals, one per line of the WGSL."""
    out = []
    for line in text.split("\n"):
        s = ""
        for ch in line:
            o = ord(ch)
            if ch == "\\":
                s += "\\\\"
            elif ch == '"':
                s += '\\"'
            elif ch == "?":
                s += "\\?"
            elif ch == "\t":
                s += "\\t"
            elif 32 <= o < 127:
                s += ch
            else:
                s += "".join(f"\\{b:03o}" for b in ch.encode("utf-8"))
        out.append('"' + s + '\\n"')
    return "\n".join(out)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = argv[1], argv[2]
    text = open(src, encoding="utf-8").read()
    pieces = []
    pos = 0
    total_spec = 0
    total_unported = 0
    for m in ARRAY_RE.finditer(text):
        name = m.group(1)
        if not name.endswith("gpu_program"):
            continue
        body_start = m.end()
        body = array_body(text, body_start)
        wgsl = "".join(string_literals(body))
        if not wgsl:
            # `{NULL}`: a generated kernel whose code lives in `gpu_program`.
            # The runtime tests `gpu_program[0]`, so the array stays empty.
            continue
        ported, removed, specs, unported = port_module(wgsl)
        print(f"{name}: {len(specs)} specialised, {len(removed)} deleted")
        for helper, target, element in specs:
            if element:
                print(f"    specialised {helper}(&{target}[i], ...) -> {helper}__{target}__elem(i, ...)")
            else:
                print(f"    specialised {helper}(&{target}, ...) -> {helper}__{target}(...)")
        for r in removed:
            print(f"    deleted {r}")
        for u in unported:
            print(f"    NOT PORTED (the pointer argument is neither &name nor &name[i]): {u}")
        total_spec += len(specs)
        total_unported += len(unported)
        pieces.append(text[pos:body_start])
        pieces.append("\n" + c_literal(ported) + ",\n")
        pos = body_start + len(body)
    pieces.append(text[pos:])
    with open(dst, "w", encoding="utf-8") as f:
        f.write("".join(pieces))
    print(f"wrote {dst}: {total_spec} specialisation(s), {total_unported} call(s) not ported")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
