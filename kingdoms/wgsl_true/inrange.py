"""In-range indices as a type (docs/DESIGN_inrange_index_types.md), for an interface that declares `inrange_types`.

`u32.in_<B>` is a value below `arrayLength(&B)`; a load or store of B takes it and spells no clamp (values.py, arena.py).
`u32.len_<B>` is a value at most `arrayLength(&B)`. The rows that give `u32.in_<B>`:

* `safe.inrange_<B>`: the clamp `min({0}, arrayLength(&B) - 1u)`, in range by its own text (every index stays legal);
* `own.u32_in_<B>`: `lit_u32(0u)`, in range because WebGPU forbids a zero-size binding;
* `inrange.<B>_g<d>`: the guard register `gi_<B><d>`, declared `0u` and assigned only by a guard row of depth d, inside
  `if (gv < L)` with `L: u32.len_<B>`, so it is below `arrayLength(&B)` at every point of the kernel.

`u32.len_<B>`: `arrayLength(&B)`; the interface's `len_bounds` of B (texts over the uniform's fields, each proved at most
the contract's length of B by `witness`); and `len.<B>_of_in`, `{0}` over `u32.in_<B>` (`x < len` gives `x <= len`).

`witness` re-reads every emitted spelling and proves the rules above from the text (gen_tables refuses to write a table
that breaks one).
"""
import os
import re

from lib import *

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GUARD_DEPTHS = range(0, 6)


def buffers(iface):
    """The storage bindings whose plain loads and stores take `u32.in_<B>`: the inputs (not an atomic one) and the
    non-atomic arenas."""
    if not iface.get("inrange_types"):
        return []
    out = [i["name"] for i in iface.get("inputs", []) if i.get("kind") != "atomic_u32"]
    out += [a["name"] for a in iface.get("arenas", []) if not a.get("atomic")]
    return out


def in_t(b):
    return f"u32.in_{b}"


def len_t(b):
    return f"u32.len_{b}"


def reg(b, d):
    return f"gi_{b}{d}"


def declare(iface):
    for b in buffers(iface):
        T(in_t(b), f"a position below arrayLength(&{b}): a load or store of {b} spells no clamp")
        register_twin(in_t(b))
        T(len_t(b), f"a value at most arrayLength(&{b}) (the dispatch contract's length of {b})")
        register_twin(len_t(b))
        for d in GUARD_DEPTHS:
            decl(f"var {reg(b, d)}: u32 = 0u;")


# ---------------------------------------------------------------- offset types (design section 6)
# A RUN of binding B is a region [base, base + L) the dispatch contract places inside B (the interface's `runs`: a base
# polynomial over the uniform's fields, nb = ceil(x / 256), and a length L that is a uniform field). `u32.upto_<L>` is
# a value at most `<u>.<L>`; `inrange.<B>_run_<r>(u32.upto_<L>)`, spelled `(<base> + {0})`, is in range of B because
# the witness proves base + L + 1 <= the contract's length of B. A run loop `loop.run_<L><c>_<X><d>_f<k>` counts in its
# own register kr_<L><d> (with bound register nr_<L><d> = min(<u>.<L>, 2^c)); only run loops assign them, so kr_<L><d>
# is at most <u>.<L> everywhere: 0 before any run loop, below nr_<L><d> inside one, nr_<L><d> after it.

def run_lengths(iface):
    return sorted({r["len"] for b in buffers(iface) for r in iface.get("runs", {}).get(b, [])})


def upto_t(f):
    return f"u32.upto_{f}"


def runlen_t(f):
    return f"u32.runlen_{f}"


def wgsl_of(poly, u):
    """The WGSL text of a run's base polynomial (deterministic, so the witness re-derives it)."""
    def tok(m):
        t = m.group(0)
        if t.isdigit():
            return t + "u"
        if t == "nb":
            return f"(({u}.x + 255u) / 256u)"
        return f"{u}.{t}"
    return "(" + re.sub(r"\bnb\b|\b[a-z]\b|\b\d+\b", tok, poly) + ")"


def run_loop_text(f, d1, cap, u):
    kr, nr = f"kr_{f}{d1}", f"nr_{f}{d1}"
    return f"{nr} = min({{0}}, {cap}u); {kr} = 0u; loop {{ if ({kr} >= {nr}) {{ break; }} {{1}} {kr} = {kr} + 1u; }}"


def declare_runs(iface):
    for f in run_lengths(iface):
        T(upto_t(f), f"a value at most the uniform field {f}: a run loop's counter, 0, or a clamp")
        T(runlen_t(f), f"the uniform field {f} itself: a run loop's trip count (an argument, so the work bound reads its value)")
        for d in range(1, 6):
            decl(f"var kr_{f}{d}: u32 = 0u;")
            decl(f"var nr_{f}{d}: u32 = 0u;")


def build_runs(iface):
    u = iface.get("uniform", "cfg")
    for f in run_lengths(iface):
        R(f"own.u32_upto_{f}", upto_t(f), [], "lit_u32(0u)", "0 is at most any length")
        R(f"run.len_{f}", runlen_t(f), [], f"{u}.{f}", "the run length's field")
        R(f"run.upto_{f}_clamp", upto_t(f), ["u32.index"], f"min({{0}}, {u}.{f})", "any value restricted to the length by the row's text")
        R(f"index.of_upto_{f}", "u32.index", [upto_t(f)], "{0}", "a run offset as a value", twin=True)
        for d in range(1, 6):
            R(f"run.kr_{f}{d}", upto_t(f), [], f"kr_{f}{d}", f"the counter of the run loops of depth {d}: at most {u}.{f} in every state")
        for c in (6, 12, 18, 24):
            cls = c // 6
            for x, d, k in rungs("LC"):
                if d + 1 > 5 or k + cls > K:
                    continue
                R(f"loop.run_{f}{c}_{x}{d}_f{k}", S(x, d, k + cls), [runlen_t(f), B("C", d + 1, k)], run_loop_text(f, d + 1, 2 ** c, u),
                  f"a loop over a run's length: its counter kr_{f}{d + 1} offsets the runs of length {u}.{f}")
    for b in buffers(iface):
        for r in iface.get("runs", {}).get(b, []):
            R(f"inrange.{b}_run_{r['name']}", in_t(b), [upto_t(r["len"])], f"({wgsl_of(r['base'], u)} + {{0}})",
              f"the run {r['name']} of {b}: base {r['base']} plus at most {r['len']}, below the contract's length (the witness)")


def guard_texts(b, d):
    g = reg(b, d)
    lt = "{ let gv = {0}; if (gv < {1}) { " + g + " = gv; {2} } }"
    lt_else = "{ let gv = {0}; if (gv < {1}) { " + g + " = gv; {2} } else { {3} } }"
    ge = "{ let gv = {0}; if (gv >= {1}) { {2} } else { " + g + " = gv; {3} } }"
    return lt, lt_else, ge


def break_guard_text(b, d):
    """The break guard of a loop body at depth d (design section 7): `{0}`, then the index `{1}` checked against the
    length `{2}`; the loop breaks unless it is below, so the register is assigned only below `{2}`, and `{3}`, the rest
    of the body, runs only after that assignment."""
    return "{0} { let gv = {1}; if (gv >= {2}) { break; } " + reg(b, d) + " = gv; } {3}"


def build(iface):
    u = iface.get("uniform", "cfg")
    for b in buffers(iface):
        R(f"safe.inrange_{b}", in_t(b), ["u32.index"], f"min({{0}}, arrayLength(&{b}) - 1u)", f"any position restricted to {b}'s last element by the row's own text", twin=True)
        R(f"own.u32_in_{b}", in_t(b), [], "lit_u32(0u)", "the first element: a binding is never empty", twin=True)
        for d in GUARD_DEPTHS:
            R(f"inrange.{b}_g{d}", in_t(b), [], reg(b, d), f"the guard register of depth {d}: 0u, or a value a guard row proved below a length of {b}")
        R(f"len.{b}_len", len_t(b), [], f"arrayLength(&{b})", twin=True)
        for f in iface.get("len_bounds", {}).get(b, []):
            R(f"len.{b}_{f}", len_t(b), [], f"{u}.{f}", f"the contract: {u}.{f} is at most the length of {b}", twin=True)
        R(f"len.{b}_of_in", len_t(b), [in_t(b)], "{0}", "below the length is at most the length", twin=True)
        R(f"len.{b}_min", len_t(b), ["u32.index", len_t(b)], "min({0}, {1})", "at most a value at most the length", twin=True)
        for d in range(1, 6):
            for k in STRATA:
                R(f"block.brk_guard_{b}_B{d}_f{k}", f"block.B{d}_f{k}", [B("C", d, k), "u32.index", len_t(b), B("C", d, k)], break_guard_text(b, d),
                  f"if (e >= L) {{ break; }} with L at most {b}'s length: e is in range of {b} in the rest of the body (the register)")
        for x, d, k in rungs("LC"):
            lt, lt_else, ge = guard_texts(b, d)
            body = B("C", d, k)
            sfx = f"_{b}_{x}{d}_f{k}"
            R("stmt.guard_lt" + sfx, S(x, d, k), ["u32.index", len_t(b), body], lt, f"if (e < L) with L at most {b}'s length: e is in range of {b} in the body (the register)")
            R("stmt.guard_lt_else" + sfx, S(x, d, k), ["u32.index", len_t(b), body, body], lt_else)
            R("stmt.guard_ge" + sfx, S(x, d, k), ["u32.index", len_t(b), body, body], ge, "if (e >= L) {..} else {..}: e is in range of the else branch")


# ---------------------------------------------------------------- the witness

def _poly_parse(text):
    """A polynomial over named non-negative variables from `+`, `*`, integers, names and parentheses: {monomial: coeff},
    a monomial a sorted tuple of names."""
    toks = re.findall(r"\d+|[A-Za-z_]\w*|[+*()]", text)
    assert "".join(toks) == re.sub(r"\s+", "", text), f"the length text {text!r} holds more than + * ( ) names and integers"
    pos = [0]

    def add(p, q):
        r = dict(p)
        for m, c in q.items():
            r[m] = r.get(m, 0) + c
        return {m: c for m, c in r.items() if c != 0}

    def mul(p, q):
        r = {}
        for m1, c1 in p.items():
            for m2, c2 in q.items():
                m = tuple(sorted(m1 + m2))
                r[m] = r.get(m, 0) + c1 * c2
        return {m: c for m, c in r.items() if c != 0}

    def atom():
        t = toks[pos[0]]
        pos[0] += 1
        if t == "(":
            p = expr()
            assert toks[pos[0]] == ")", text
            pos[0] += 1
            return p
        if t.isdigit():
            return {(): int(t)} if int(t) else {}
        return {(t,): 1}

    def term():
        p = atom()
        while pos[0] < len(toks) and toks[pos[0]] == "*":
            pos[0] += 1
            p = mul(p, atom())
        return p

    def expr():
        p = term()
        while pos[0] < len(toks) and toks[pos[0]] == "+":
            pos[0] += 1
            p = add(p, term())
        return p

    p = expr()
    assert pos[0] == len(toks), text
    return p, add, mul


def contract_ranges(iface):
    """{field: (lo, hi)} from the problem's safety.contract (`param <uniform>.<f> lo hi`), plus nb = ceil(x / 256)."""
    path = os.path.join(ROOT, "problems", iface["name"], "safety.contract")
    u = iface.get("uniform", "cfg")
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) == 4 and p[0] == "param" and p[1].startswith(u + "."):
            out[p[1][len(u) + 1:]] = (int(p[2]), int(p[3]))
    if "x" in out:
        out["nb"] = ((out["x"][0] + 255) // 256, (out["x"][1] + 255) // 256)
    return out


def dominated(bound, length, ranges):
    """True when `bound <= length` for every value in the ranges: each variable v is lo_v + v' with v' >= 0, and every
    coefficient of (length - bound) over the v' is non-negative (a sum of non-negative terms)."""
    p, add, mul = _poly_parse(f"({length})")
    q, _, _ = _poly_parse(f"({bound})")
    diff = add(p, {m: -c for m, c in q.items()})
    sub = {}
    for m, c in diff.items():
        term = {(): c}
        for v in m:
            lo = ranges[v][0]
            term = mul(term, add({(v + "'",): 1}, {(): lo} if lo else {}))
        sub = add(sub, term)
    return all(c >= 0 for c in sub.values())


def value_at_hi(text, ranges):
    p, _, _ = _poly_parse(f"({text})")
    total = 0
    for m, c in p.items():
        v = c
        for n in m:
            v *= ranges[n][1]
        total += v
    return total


def witness(iface):
    """Every rule of the design proved from the emitted spellings. Returns the problems (empty when every rule holds)."""
    bs = buffers(iface)
    if not bs:
        return []
    problems = []
    ranges = contract_ranges(iface)
    lengths = iface["lengths"]
    for b in bs:
        if b not in lengths:
            problems.append(f"the interface declares no contract length of {b}")
            continue
        for f in iface.get("len_bounds", {}).get(b, []):
            if not dominated(f, lengths[b], ranges):
                problems.append(f"len_bounds {f} of {b} is not proved at most its length {lengths[b]} over {ranges}")
            if value_at_hi(f, ranges) >= 2 ** 32:
                problems.append(f"len_bounds {f} of {b} wraps at the contract's largest parameters")
    subscript = {b: re.compile(r"\b" + re.escape(b) + r"\[") for b in bs}
    clamp = {b: f"min({{}}, arrayLength(&{b}) - 1u)" for b in bs}
    for name, r in ROWS.items():
        sp = r["spell"]
        if name.startswith("cs."):
            # a counting-sort scaffold (countsort.py): a reference kernel's own entry text, in range by the contract's
            # lengths as the kernel is (its subscripts are checked by the CPU executor on every case, a Rust test), not
            # by these rules; countsort's witness proves it is a compiler variant byte for byte
            continue
        for b in bs:
            # rule 1: every subscript of B is {i} of type in_B, or a clamp text
            for m in subscript[b].finditer(sp):
                depth, j = 1, m.end()
                while depth:
                    depth += {"[": 1, "]": -1}.get(sp[j], 0)
                    j += 1
                inner = sp[m.end():j - 1]
                hole = re.fullmatch(r"\{(\d+)\}", inner)
                if hole:
                    a = r["args"][int(hole.group(1))]
                    if a not in (in_t(b), w_of(in_t(b)), a_of(in_t(b))):
                        problems.append(f"row {name} indexes {b} with argument type {a}, not {in_t(b)}")
                elif not (inner.startswith("min(") and inner.endswith(f", arrayLength(&{b}) - 1u)")):
                    problems.append(f"row {name} indexes {b} by {inner!r}: neither an in_{b} argument nor the clamp")
            # rule 2: the rows that give in_B are the design's four forms
            if r["out"] in (in_t(b), w_of(in_t(b)), a_of(in_t(b))):
                ok = (r["args"] == ["u32.index"] or r["args"] == [w_of("u32.index")] or r["args"] == [a_of("u32.index")]) and sp == clamp[b].format("{0}")
                ok = ok or (not r["args"] and sp == "lit_u32(0u)")
                ok = ok or (r["out"] == in_t(b) and not r["args"] and re.fullmatch(re.escape(f"gi_{b}") + r"\d", sp) is not None)
                # section 6: (base + {0}) over u32.upto_L, base a run of B whose base + L + 1 is proved at most B's length
                runs = {f"({wgsl_of(q['base'], iface.get('uniform', 'cfg'))} + {{0}})": q for q in iface.get("runs", {}).get(b, [])}
                q = runs.get(sp)
                if r["out"] == in_t(b) and q is not None and r["args"] == [upto_t(q["len"])]:
                    bound = f"{q['base']} + {q['len']} + 1"
                    if not dominated(bound, lengths[b], ranges):
                        problems.append(f"run {q['name']} of {b}: {bound} is not proved at most its length {lengths[b]}")
                    elif value_at_hi(bound, ranges) >= 2 ** 32:
                        problems.append(f"run {q['name']} of {b} wraps at the contract's largest parameters")
                    else:
                        ok = True
                if not ok:
                    problems.append(f"row {name} gives {r['out']} by {sp!r}, which is none of the design's minting forms")
            # rule 2: every assignment of a guard register is a guard row's, under its own guard of a length of B
            for m in re.finditer(re.escape(f"gi_{b}") + r"(\d) = ", sp):
                d = m.group(1)
                lt, lt_else, ge = guard_texts(b, int(d))
                guard = sp in (lt, lt_else, ge) and len(r["args"]) >= 2 and r["args"][1] == len_t(b)
                guard = guard or (sp == break_guard_text(b, int(d)) and len(r["args"]) == 4 and r["args"][2] == len_t(b) and r["out"].startswith(f"block.B{d}_f")
                                  and r["args"][0] == r["args"][3] and r["args"][0].startswith(f"block.C{d}_f"))
                if not guard:
                    problems.append(f"row {name} assigns gi_{b}{d} outside a guard of a length of {b}: {sp!r}")
            # rule 3 (the len rows): arrayLength, a proved contract bound, or the in_B identity
            if r["out"] in (len_t(b), w_of(len_t(b)), a_of(len_t(b))):
                u = iface.get("uniform", "cfg")
                ok = (not r["args"] and sp == f"arrayLength(&{b})")
                ok = ok or (not r["args"] and sp.startswith(u + ".") and sp[len(u) + 1:] in iface.get("len_bounds", {}).get(b, []))
                ok = ok or (sp == "{0}" and len(r["args"]) == 1 and r["args"][0] in (in_t(b), w_of(in_t(b)), a_of(in_t(b))))
                ok = ok or (sp == "min({0}, {1})" and len(r["args"]) == 2 and r["args"][1] in (len_t(b), w_of(len_t(b)), a_of(len_t(b))))
                if not ok:
                    problems.append(f"row {name} gives {r['out']} by {sp!r}, which is not proved at most the length of {b}")
    # section 6: the rows giving u32.upto_L are a run counter, 0 or the clamp; only run loops assign kr_L<d> / nr_L<d>
    u = iface.get("uniform", "cfg")
    for f in run_lengths(iface):
        for name, r in ROWS.items():
            sp = r["spell"]
            if r["out"] == runlen_t(f) and (r["args"] or sp != f"{u}.{f}"):
                problems.append(f"row {name} gives {r['out']} by {sp!r}, not the field {u}.{f}")
            if r["out"] == upto_t(f):
                ok = (not r["args"] and (sp == "lit_u32(0u)" or re.fullmatch(f"kr_{f}" + r"[1-5]", sp) is not None))
                ok = ok or (r["args"] == ["u32.index"] and sp == f"min({{0}}, {u}.{f})")
                if not ok:
                    problems.append(f"row {name} gives {r['out']} by {sp!r}, none of the counter, 0 or the clamp")
            for m in re.finditer(f"\\b([kn]r_{f})([1-5]) = ", sp):
                d1 = int(m.group(2))
                if not any(sp == run_loop_text(f, d1, 2 ** c, u) for c in (6, 12, 18, 24)) or len(r["args"]) != 2 or r["args"][0] != runlen_t(f) or not r["args"][1].startswith(f"block.C{d1}_f"):
                    problems.append(f"row {name} assigns {m.group(1)}{d1} and is not a run loop of depth {d1}")
    return problems


def witness_committed(table_dir):
    """The witness over a COMMITTED table (`functions.tsv` of `table_dir`, its interface named by `interface.tsv`), so a
    hand-edited table fails as a generated one would: `python3 inrange.py mod/kmeans_update_segmented`."""
    from interface import INTERFACES
    meta = dict(line.rstrip("\n").split("\t", 1) for line in open(os.path.join(table_dir, "interface.tsv")) if "\t" in line)
    iface = INTERFACES[meta["interface"]]
    unesc = lambda s: s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    ROWS.clear()
    lines = open(os.path.join(table_dir, "functions.tsv")).read().split("\n")[1:]
    for line in lines:
        if not line:
            continue
        p = line.split("\t")
        ROWS[p[1]] = dict(out=p[3], args=p[4].split(";") if p[4] else [], spell=unesc(p[5]))
    for b in buffers(iface):
        for t in (in_t(b), len_t(b)):
            TWIN[t] = t + "w"
            TWIN_A[t] = t + "a"
    TWIN["u32.index"], TWIN_A["u32.index"] = "u32.indexw", "u32.indexa"
    return witness(iface)


if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    probs = witness_committed(d if os.path.isabs(d) else os.path.join(HERE, d))
    for p in probs:
        print("PROBLEM:", p)
    print(f"inrange witness: {len(probs)} problems")
    sys.exit(1 if probs else 0)
