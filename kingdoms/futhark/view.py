"""THE VIEW: a Futhark kingdom as a query over the master table.

`master/rows.tsv` holds every row of every Futhark kingdom as a TEMPLATE over
size variables (`{D}`), the element (`{E}`) and the context (`C`). A problem's
view specification (`view.tsv`, written by `sniff.py`) is the WHERE clause:
the element, the inputs and output over named dimensions, the families, the
depth, the master version. The view

  1. FILTERS the master rows the specification admits (family, version,
     element);
  2. BINDS them: each row's scope is a list of quantifiers, outermost first;
     consecutive rows sharing their first quantifier are one block, the block
     iterates it and its rows recurse on their remaining quantifiers -- the
     master's row order IS the expansion order;
  3. keeps the types REACHABLE from the root (the closure over the rows'
     arguments), which is what makes the gene layout's per-type heads;
  4. ORDERS and writes the exact table: types.tsv, functions.tsv and the
     e-graph artefacts.

`src/futhark_view.rs` is the second implementation (the one the engine runs);
a Rust test requires the two to write the same bytes.
"""

import itertools
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "master")

VAR_RE = re.compile(r"\{([A-Z][A-Z0-9]*(?:\.[a-z0-9]+)?)\}")
QUANT_RE = re.compile(r"^([A-Z][A-Z0-9]*):([a-z0-9]+)(?:\((.*)\))?$")
ROW_COLUMNS = ["ref", "since", "family", "kind", "elems", "scope", "when", "name", "out", "args", "spelling", "note"]
FRAME_COLUMNS = ["family", "kind", "code", "scope", "binds"]
SIGNATURE_FAMILIES = ("scan", "hist", "mod")


def read_tsv(path, columns):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    head = lines[0].split("\t")
    if head != columns:
        raise ValueError(f"{path}: the header is {head}, expected {columns}")
    out = []
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) != len(columns):
            raise ValueError(f"{path}:{i}: {len(f)} columns, expected {len(columns)}")
        out.append(dict(zip(columns, f)))
    return out


def master(path=MASTER):
    """The master: its version, rows, frames and literal pools."""
    with open(os.path.join(path, "VERSION")) as f:
        version = int(f.read().strip())
    pools = {}
    for p in read_tsv(os.path.join(path, "pools.tsv"), ["element", "literals"]):
        pools[p["element"]] = p["literals"].split(",")
    return {
        "version": version,
        "rows": read_tsv(os.path.join(path, "rows.tsv"), ROW_COLUMNS),
        "frames": read_tsv(os.path.join(path, "frames.tsv"), FRAME_COLUMNS),
        "rules": read_tsv(os.path.join(path, "rules.tsv"), ["on", "elems", "needs", "rule"]),
        "pools": pools,
    }


# ---- the specification --------------------------------------------------------------------------------------------

def read_spec(text):
    """A view specification: four-column lines `kind name element dims`; `#` lines are comments."""
    spec = {"inputs": [], "sizes": [], "families": [], "shape": "signature"}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 4:
            raise ValueError(f"the specification line {line!r} has {len(f)} fields, wants 4")
        kind, name, elem, dims = f
        lst = [x for x in dims.split(",") if x]
        if kind == "input":
            spec["inputs"].append((name, elem, tuple(lst)))
        elif kind == "size":
            spec["inputs"].append((name, "size", ()))
            spec["sizes"].append(name)
        elif kind == "output":
            spec["output"] = (elem, tuple(lst))
        elif kind in ("master", "depth"):
            spec[kind] = int(name)
        elif kind in ("element", "shape"):
            spec[kind] = name
        elif kind in ("families", "literals", "dims", "roles", "root"):
            spec[kind] = [x for x in name.split(",") if x]
        else:
            raise ValueError(f"a specification line of kind {kind}")
    return spec


# ---- binding ------------------------------------------------------------------------------------------------------

def class_dims(cls):
    """`a.rows.cols` -> ("rows", "cols"); a scalar or an index -> ()."""
    return tuple(cls.split(".")[1:]) if cls.startswith("a.") else ()


def array_class(dims):
    return "a." + ".".join(dims)


def signature_sets(spec):
    """The sets a signature view's quantifiers range over, derived from the specification as the signature family
    has always derived them: the dims in order of first appearance, the classes (each input's, its rows', the
    output's, each dim's rank-1 class), the rank-1 classes, the states a loop may carry, the context-free classes."""
    elem = spec["element"]
    dims = []
    for name, e, ds in spec["inputs"]:
        if e == "size":
            dims.append(name)
        for d in ds:
            if d not in dims:
                dims.append(d)
    if any(d not in dims for d in spec["output"][1]):
        raise ValueError(f"the output's dims {spec['output'][1]} are not all dims of an input {dims}")
    classes = []

    def add(c):
        if c not in classes:
            classes.append(c)
            ds = class_dims(c)
            if len(ds) >= 2:
                add(array_class(ds[1:]))

    for _, e, ds in spec["inputs"]:
        if e != "size":
            add(array_class(ds) if ds else elem)
    out_dims = spec["output"][1]
    add(array_class(out_dims) if out_dims else elem)
    for d in dims:
        add(array_class((d,)))
    rank1 = [c for c in classes if len(class_dims(c)) == 1]
    # A class no row can produce is context-free: no row template's output is an array of rank 2 or more.
    free = [c for c in classes if len(class_dims(c)) >= 2]
    return {"elem": elem, "dims": dims, "classes": classes, "rank1": rank1, "states": [elem] + rank1, "free": free,
            "inputs": [(n, array_class(ds) if ds else elem) for n, e, ds in spec["inputs"] if e != "size"]}


def futhark_type(cls, elem):
    return "".join(f"[{d}]" for d in class_dims(cls)) + elem


def subst(text, env):
    """Every `{VAR}` of the text replaced by its binding; `{0}`, `{1}`, ... (the children) are left alone."""
    def one(m):
        if m.group(1) not in env:
            raise ValueError(f"the variable {{{m.group(1)}}} of {text!r} is not bound")
        return env[m.group(1)]
    return VAR_RE.sub(one, text)


def class_env(var, cls, elem):
    """The bindings of a class variable: the class, its dims, and its Futhark type."""
    ds = class_dims(cls)
    return {var: cls, f"{var}.type": futhark_type(cls, elem), f"{var}.dims": ".".join(ds), f"{var}.br": "][".join(ds),
            f"{var}.d0": ds[0] if ds else ""}


class Binder:
    """The quantifiers of one view."""

    def __init__(self, spec, mst):
        self.spec = spec
        if spec["shape"] == "signature":
            self.sets = signature_sets(spec)
            self.depth = spec["depth"]
            self.frames = self.frame_list(mst)
            self.contexts = self.context_list()
        else:
            roles = spec["roles"]
            self.sets = {"elem": spec["element"], "roles": roles, "pairs": [(a, b) for a in roles for b in roles]}
            self.depth = 0
            self.frames = {}
            self.contexts = [()]

    def frame_list(self, mst):
        """Every frame of the view, in order: a frame template expanded over its scope, with its code and the
        variables it binds (by frame number `{K}`, bound when a context places it)."""
        out = []
        for f in mst["frames"]:
            for env in self.expand_scope(f["scope"].split(";"), {"E": self.sets["elem"]}):
                code = subst(f["code"], env)
                binds = [b.split(":", 1) for b in f["binds"].split(",")]
                out.append((code, [(name, cls, env) for name, cls in binds]))
        return out

    def context_list(self):
        ctxs, frontier = [()], [()]
        for _ in range(self.depth):
            nxt = [c + (f[0],) for c in frontier for f in self.frames]
            ctxs += nxt
            frontier = nxt
        return ctxs

    def frame_vars(self, ctx):
        """The variables a context binds: (name, class), outermost first."""
        by_code = dict(self.frames)
        out = []
        for k, code in enumerate(ctx, start=1):
            for name, cls, env in by_code[code]:
                e = dict(env, K=str(k))
                out.append((subst(name, e), subst(cls, e)))
        return out

    def expand_scope(self, scope, env):
        """Every binding of a list of quantifiers, in order."""
        if not scope or scope == [""]:
            yield env
            return
        for b in self.quantifier(scope[0], env):
            yield from self.expand_scope(scope[1:], dict(env, **b))

    def quantifier(self, q, env):
        """The bindings of one quantifier, in order."""
        if q == "C":
            for ctx in self.contexts:
                yield {"C": ctx, "K": str(len(ctx) + 1), "CTX": "/".join(("c",) + ctx), "shallow": len(ctx) < self.depth}
            return
        m = QUANT_RE.match(q)
        if not m:
            raise ValueError(f"the quantifier {q!r} does not parse")
        var, kind, arg = m.group(1), m.group(2), subst(m.group(3), env) if m.group(3) is not None else None
        s, elem = self.sets, self.sets["elem"]
        if kind == "dims":
            yield from ({var: d} for d in s["dims"])
        elif kind == "rank1":
            yield from ({var: class_dims(c)[0]} for c in s["rank1"])
        elif kind == "states":
            yield from (class_env(var, c, elem) for c in s["states"])
        elif kind == "free":
            yield from (class_env(var, c, elem) for c in s["free"])
        elif kind == "vars":
            yield from ({var: v} for v, c in self.frame_vars(env["C"]) if c == arg)
        elif kind == "inputs":
            yield from ({var: n} for n, c in s["inputs"] if c == arg)
        elif kind == "rows":
            for y in s["classes"]:
                ys = class_dims(y)
                if len(ys) >= 2 and array_class(ys[1:]) == arg:
                    yield class_env(var, y, elem)
        elif kind == "pairs":
            yield from ({var: f"{a}.{b}", f"{var}.a": a, f"{var}.b": b} for a, b in s["pairs"])
        elif kind == "swap":
            a, b = arg.split(".")
            if (b, a) in s["pairs"]:
                yield {var: f"{b}.{a}"}
        elif kind == "triples":
            for a, b, c in itertools.product(s["roles"], repeat=3):
                if (a, b) in s["pairs"] and (b, c) in s["pairs"] and (a, c) in s["pairs"]:
                    yield {var: f"{a}.{b}.{c}", f"{var}.ab": f"{a}.{b}", f"{var}.bc": f"{b}.{c}", f"{var}.ac": f"{a}.{c}"}
        else:
            raise ValueError(f"the quantifier kind {kind!r} is not one of the master's")

    def dual(self, ty, env):
        """A type template in a row's context as the dual: `class:context`, or the bare class when the row is
        outside any context or the class is context-free. `T+<frame>` is T in the context plus that frame."""
        ty = subst(ty, env)
        cls, plus, frame = ty.partition("+")
        if "C" not in env or cls in self.sets.get("free", ()):
            if plus:
                raise ValueError(f"the type {ty} pushes a frame outside a context")
            return cls
        ctx = env["C"] + ((frame,) if plus else ())
        if plus and frame not in dict(self.frames):
            raise ValueError(f"the type {ty} pushes the frame {frame}, which the view does not have")
        return f"{cls}:{'/'.join(('c',) + ctx)}"


def admitted(row, spec, mst):
    """Whether the specification admits a master row: its family, its version, its element."""
    fams = set(spec["families"]) | ({"core"} if spec["shape"] == "signature" else set())
    elems = row["elems"].split(",")
    return (row["family"] in fams and int(row["since"]) <= spec.get("master", mst["version"])
            and (row["elems"] == "*" or spec["element"] in elems))


def expand(rows, binder, env):
    """The rows of a block, expanded: rows with no quantifier left are instantiated in place; consecutive rows that
    share their next quantifier form one block that iterates it."""
    out, i = [], 0
    while i < len(rows):
        row, scope = rows[i]
        if not scope:
            if row["when"] == "shallow" and not env.get("shallow", False):
                pass
            elif row["when"] not in ("", "shallow"):
                raise ValueError(f"the row {row['ref']} has the condition {row['when']!r}")
            else:
                out.append(instantiate(row, binder, env))
            i += 1
            continue
        q, j = scope[0], i
        while j < len(rows) and rows[j][1] and rows[j][1][0] == q:
            j += 1
        block = [(r, s[1:]) for r, s in rows[i:j]]
        for b in binder.quantifier(q, env):
            out += expand(block, binder, dict(env, **b))
        i = j
    return out


def instantiate(row, binder, env):
    out = binder.dual(row["out"], env)
    args = [binder.dual(a, env) for a in row["args"].split(";") if a]
    e = dict(env, OUT=out)
    return {"ref": row["ref"], "name": subst(row["name"], e), "kind": row["kind"], "out": out, "args": args,
            "spelling": subst(row["spelling"], e), "note": subst(row["note"], e)}


def view_rows(spec, mst):
    """Steps 1 to 3: the view's rows, filtered, bound, and closed under reachability from the root."""
    binder = Binder(spec, mst)
    picked = [(r, [q for q in r["scope"].split(";") if q]) for r in mst["rows"] if admitted(r, spec, mst)]
    rows = expand(picked, binder, {"E": spec["element"]})
    if spec["shape"] == "signature":
        elem, dims = spec["output"]
        root = binder.dual(array_class(dims) if dims else elem, {"C": ()})
    else:
        root = "mat." + ".".join(spec["root"])
    keep, changed = {root}, True
    while changed:
        changed = False
        for r in rows:
            if r["out"] in keep:
                for a in r["args"]:
                    if a not in keep:
                        keep.add(a)
                        changed = True
    return [r for r in rows if r["out"] in keep], root, binder


# ---- the written tables ---------------------------------------------------------------------------------------------

def signature_types(spec, rows, root):
    """The signature view's types: every dual a row names, the root first, then by context depth and name."""
    elem = spec["element"]
    literals = [f"{l}{elem}" for l in spec["literals"]]
    named = []
    for r in rows:
        for d in [r["out"]] + r["args"]:
            if d not in named:
                named.append(d)
    named.sort(key=lambda d: (d != root, d.count("/"), d))
    out = []
    for d in named:
        cls, _, ctx = d.partition(":")
        slot = "ix" if cls.startswith("ix.") else ("arr" if cls.startswith("a.") else elem)
        meaning = {"ix": f"an index in range for [{cls[3:]}]", "arr": f"a [{']['.join(class_dims(cls))}]{elem}",
                   elem: f"an {elem}"}[slot] + (f", in the context {ctx}" if ctx else ", the interface's input class")
        out.append({"dual": d, "slot": slot, "form": cls, "literals": literals if slot == elem else [],
                    "root": d == root, "meaning": meaning})
    return out


def ordered(rows):
    """Functions first, then terminals and constants, each in expansion order."""
    return [r for r in rows if r["kind"] == "function"] + [r for r in rows if r["kind"] != "function"]


def types_text(types):
    out = ["ref\tdual\tslot\tform\tliterals\troot\tmeaning"]
    for i, t in enumerate(types, start=1):
        out.append(f"T{i:04d}\t{t['dual']}\t{t['slot']}\t{t['form']}\t{','.join(t['literals'])}\t"
                   f"{'yes' if t['root'] else 'no'}\t{t['meaning']}")
    return "\n".join(out) + "\n"


def functions_text(rows):
    out = ["ref\tname\tkind\tout\targs\tspelling\tnote"]
    for i, r in enumerate(rows, start=1):
        out.append(f"F{i:05d}\t{r['name']}\t{r['kind']}\t{r['out']}\t{';'.join(r['args'])}\t{r['spelling']}\t{r['note']}")
    return "\n".join(out) + "\n"


def sig_sort(dual):
    return "S" + re.sub(r"[^A-Za-z0-9]", "_", dual)


def sig_ctor(name, kind, out):
    if kind == "const":
        return "C" + sig_sort(out)[1:]
    op = name.split("@", 1)[0]
    return "X" + re.sub(r"[^A-Za-z0-9]", "_", op) + "__" + sig_sort(out)[1:]


RULE_CTOR_RE = re.compile(r"\{([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)\}")


def signature_egraph(types, rows, mst, elem):
    """The signature view's e-graph: one sort a dual, one constructor a row (a constant takes its pool index), and
    the master's rules instantiated on every type they apply to (`elem`: the element types; `arr`: the rank-1 array
    types in a context that hold the row the rule `needs`)."""
    data = ["; The signature family's e-graph datatype, generated by gen_tables.py.",
            "; One sort per dual: the sort IS the type (class and context).", ""]
    for t in types:
        data.append(f"(sort {sig_sort(t['dual'])})")
    data.append("")
    for r in rows:
        if r["kind"] == "const":
            data.append(f"(constructor {sig_ctor(r['name'], r['kind'], r['out'])} (i64) {sig_sort(r['out'])})")
        else:
            data.append(f"(constructor {sig_ctor(r['name'], r['kind'], r['out'])} ({' '.join(sig_sort(a) for a in r['args'])}) {sig_sort(r['out'])})")
    names = {r["name"] for r in rows}
    rules = ["; The signature family's e-graph ruleset, generated by gen_tables.py.", "(ruleset futhark_sig)", ""]
    for on in ("elem", "arr"):
        for t in types:
            d = t["dual"]
            if on == "elem" and t["slot"] != elem:
                continue
            if on == "arr" and (t["slot"] != "arr" or t["form"].count(".") != 1 or ":" not in d):
                continue
            mine = [r for r in mst["rules"] if r["on"] == on and elem in r["elems"].split(",")
                    and (not r["needs"] or f"{r['needs']}@{d}" in names)]
            if not mine:
                continue
            rules.append(f"; {d}")
            for r in mine:
                rules.append(RULE_CTOR_RE.sub(lambda m: sig_ctor(f"{m.group(1)}@{d}", "function", d), r["rule"]))
            rules.append("")
    rtext = ["# row name\tconstructor\tsort"] + [f"{r['name']}\t{sig_ctor(r['name'], r['kind'], r['out'])}\t{sig_sort(r['out'])}" for r in rows]
    stext = ["# dual\tsort"] + [f"{t['dual']}\t{sig_sort(t['dual'])}" for t in types]
    return {
        "egraph.datatype": "\n".join(data) + "\n",
        "egraph.rules": "\n".join(rules) + "\n",
        "egraph.rows": "\n".join(rtext) + "\n",
        "egraph.sorts": "\n".join(stext) + "\n",
    }


def signature_view(spec_text, mst=None):
    """Every written artefact of a signature view, by file name."""
    mst = mst or master()
    spec = read_spec(spec_text)
    rows, root, _ = view_rows(spec, mst)
    types = signature_types(spec, rows, root)
    rows = ordered(rows)
    out = {"types.tsv": types_text(types), "functions.tsv": functions_text(rows)}
    out.update(signature_egraph(types, rows, mst, spec["element"]))
    return out


def matrix_view(spec_text, mst=None):
    """The matrix family's view (the A7 kingdom): its types, declared in the master's template order, and its rows.
    Its cost table and e-graph are derived from these rows by the generator (`gen_tables.py`)."""
    mst = mst or master()
    spec = read_spec(spec_text)
    rows, root, binder = view_rows(spec, mst)
    types = []
    for t in read_tsv(os.path.join(MASTER, "types.tsv"), ["family", "scope", "dual", "slot", "form", "meaning"]):
        if t["family"] not in spec["families"]:
            continue
        for env in binder.expand_scope([q for q in t["scope"].split(";") if q], {"E": spec["element"]}):
            dual = subst(t["dual"], env)
            types.append({"dual": dual, "slot": t["slot"], "form": subst(t["form"], env), "literals": [],
                          "root": dual == root, "meaning": subst(t["meaning"], env)})
    named = {d for r in rows for d in [r["out"]] + r["args"]}
    types = [t for t in types if t["dual"] in named]
    return {"types.tsv": types_text(types), "functions.tsv": functions_text(ordered(rows))}, rows
