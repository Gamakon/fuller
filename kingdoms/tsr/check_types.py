#!/usr/bin/env python3
"""Checks of the TSR kingdom tables against the engine's own definitions.

    python3 -I check_types.py            # the whole suite, no GPU, no cargo
    python3 -I check_types.py --quick    # fewer trees (smoke test only; the report says so)

The tables are READ FROM THE TSVs (and regenerated into a scratch directory for
the ceiling-2 and the extra combinations); the legality of a tree is computed by
a literal port of the engine's definitions, not from the profile the types were
derived from:

  stream order   phylu src/evolve/order.rs `stream_order` (arms of a sum chain or
                 a product chain flattened, greatest arm order plus 1 on a tie)
  depth          phylu src/evolve/engine.rs `t_depth` (raisers on a path)
  windows        order.rs `RULES` upper bounds of the argument order

Plan checks (docs/PLAN_tsr_typed_gep.md section 6): #3 types against the rules
(2.6), #8 redundancy counts, #9 alphabet, and the widths of section 3 computed
from the TSVs by code that shares nothing with the generator.
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FULLER = os.path.abspath(os.path.join(HERE, "..", ".."))
PHYLU = os.path.abspath(os.path.join(FULLER, "..", "phylu"))

# ------------------------------------------------- the engine's definitions ----
# Arity per Math constructor of the 18-symbol alphabet.
ARITY = {c: 2 for c in ("Add", "Sub", "Mul", "ProtectedDiv")}
ARITY.update({c: 1 for c in (
    "Sin", "Cos", "Tanh", "ProtectedSqrt", "ProtectedLog", "ProtectedExp", "Neg", "Abs",
    "Pow2", "Pow3", "ProtectedInv", "Tan", "ProtectedAsin", "ProtectedAcos")})
FAMILY = {"Add": 1, "Sub": 1, "Mul": 2, "ProtectedDiv": 2}  # order.rs `family`
# engine.rs DEPTH_RAISING_OPS restricted to the alphabet.
RAISERS = {"Abs", "ProtectedSqrt", "ProtectedLog", "ProtectedExp", "Sin", "Cos", "Tan", "Tanh",
           "ProtectedAsin", "ProtectedAcos"}
# order.rs RULES upper bounds (arg order at most 2); ProtectedInv is `Inv` after canon.
WINDOWED = {"Pow2", "Pow3", "ProtectedInv", "Sin", "Cos", "Tan", "Tanh"}
WIDE_CTORS = ["Add", "Sub", "Mul", "ProtectedDiv", "Sin", "Cos", "Tanh", "ProtectedSqrt", "ProtectedLog",
              "ProtectedExp", "Neg", "Abs", "Pow2", "Pow3", "ProtectedInv", "Tan", "ProtectedAsin",
              "ProtectedAcos"]
UNARY = [c for c in WIDE_CTORS if ARITY[c] == 1]
BINARY = [c for c in WIDE_CTORS if ARITY[c] == 2]

# Tree: ("L",) | (ctor, child) | (ctor, left, right).
LEAF = ("L",)


def flatten(tree):
    """Breadth-first node array [(ctor, first_child)], children contiguous (math_to_nodes)."""
    nodes = [None]
    queue = [(tree, 0)]
    qi = 0
    while qi < len(queue):
        t, slot = queue[qi]
        qi += 1
        if t[0] == "L":
            nodes[slot] = ("L", 0)
        else:
            first = len(nodes)
            nodes[slot] = (t[0], first)
            for ch in t[1:]:
                nodes.append(None)
                queue.append((ch, len(nodes) - 1))
    return nodes


def arity(ctor):
    return 0 if ctor == "L" else ARITY[ctor]


def engine_stream_order(nodes):
    """Literal port of order.rs `stream_order`."""
    n = len(nodes)
    best = [1] * n
    ties = [0] * n
    order = [1] * n
    for i in range(n - 1, -1, -1):
        ctor, first = nodes[i]
        a = arity(ctor)
        if a == 0:
            order[i] = 1
            continue
        fam = FAMILY.get(ctor)
        m = c = 0
        for j in range(a):
            k = first + j
            if fam is not None and FAMILY.get(nodes[k][0]) == fam:
                o, kk = best[k], ties[k]
            else:
                o, kk = order[k], 1
            if o > m:
                m, c = o, kk
            elif o == m:
                c += kk
        best[i], ties[i] = m, c
        order[i] = m + (1 if c >= 2 else 0)
    return order


def engine_t_depth(nodes):
    """Literal port of engine.rs `t_depth`."""
    def counts(i):
        return 1 if nodes[i][0] in RAISERS else 0
    depth = [0] * len(nodes)
    depth[0] = counts(0)
    for i, (ctor, first) in enumerate(nodes):
        for j in range(arity(ctor)):
            k = first + j
            depth[k] = depth[i] + counts(k)
    return max(depth)


def engine_facts(tree):
    """(root order, t_depth, windows hold) by the literal ports."""
    nodes = flatten(tree)
    order = engine_stream_order(nodes)
    ok = all(order[first] <= 2 for ctor, first in nodes if ctor in WINDOWED)
    return order[0], engine_t_depth(nodes), ok


def legal(facts, ceiling, depth, hi):
    o, d, w = facts
    return (ceiling is None or o <= ceiling) and (not depth or d <= 2) and (not hi or w)


# ----------------------------------------------------------- the tables ----
class Tables:
    def __init__(self, directory):
        self.dir = directory
        self.types = self._read("types.tsv")
        self.functions = self._read("functions.tsv")
        self.duals = [t["dual"] for t in self.types]
        self.root = [t["dual"] for t in self.types if t["root"] == "yes"]
        if len(self.root) != 1:
            raise AssertionError(f"{directory}: exactly one root dual expected, found {self.root}")
        self.root = self.root[0]
        self.has_root_twins = self.root == "ROOT"
        self.fn = [f for f in self.functions if f["class"] == "function"]
        self.twin = [f for f in self.functions if f["class"] == "root"]
        self.term = [f for f in self.functions if f["class"] == "terminal"]
        self.absent = [f for f in self.functions if f["class"] == "absent"]
        self.canon = [f for f in self.functions if f["class"] == "canon"]
        self.body = [d for d in self.duals if d != "ROOT"]
        # Typing tables: (ctor, argument duals) -> output.
        self.rows = defaultdict(set)   # ctor -> {(ins, out)}
        for f in self.fn:
            self.rows[f["constructor"]].add((tuple(f["in"].split(",")), f["out"]))
        self.twins = defaultdict(set)  # ctor -> {ins}
        for f in self.twin:
            self.twins[f["constructor"]].add(tuple(f["in"].split(",")))

    def _read(self, name):
        with open(os.path.join(self.dir, name)) as fh:
            lines = fh.read().rstrip("\n").split("\n")
        head = lines[0].split("\t")
        return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]

    # --- typing by sets of achievable duals (a leaf fits every dual) ---
    def typer(self):
        return Typer(self)

    def all_rows_with_twins(self):
        """[(ctor, ins, out)] including the root twins."""
        out = [(f["constructor"], tuple(f["in"].split(",")), f["out"]) for f in self.fn]
        out += [(f["constructor"], tuple(f["in"].split(",")), f["out"]) for f in self.twin]
        return out


class Typer:
    """Bottom-up typing. A node's value is an interned id of the frozenset of
    duals its subtree can be given; transitions are memoised by (ctor, ids)."""

    def __init__(self, t):
        self.t = t
        self.sets = []
        self.ids = {}
        self.memo = {}
        self.leaf = self.intern(frozenset(t.body if t.has_root_twins else t.duals))
        self.by_ctor = defaultdict(list)
        for ctor, rows in t.rows.items():
            for ins, out in rows:
                self.by_ctor[ctor].append((ins, out))

    def intern(self, s):
        i = self.ids.get(s)
        if i is None:
            i = len(self.sets)
            self.ids[s] = i
            self.sets.append(s)
        return i

    def step(self, ctor, *kids):
        key = (ctor,) + kids
        r = self.memo.get(key)
        if r is None:
            ks = [self.sets[k] for k in kids]
            r = self.intern(frozenset(
                out for ins, out in self.by_ctor[ctor] if all(a in s for a, s in zip(ins, ks))))
            self.memo[key] = r
        return r

    def root_ok(self, ctor, *kids):
        """Does the whole tree rooted at (ctor, kids) have a typing at the root dual?"""
        key = ("root", ctor) + kids
        r = self.memo.get(key)
        if r is None:
            ks = [self.sets[k] for k in kids]
            if self.t.has_root_twins:
                r = any(all(a in s for a, s in zip(ins, ks)) for ins in self.t.twins[ctor])
            else:
                r = bool(self.step(ctor, *kids) != self.intern(frozenset()))
            self.memo[key] = r
        return r

    def of_tree(self, tree):
        if tree[0] == "L":
            return self.leaf
        return self.step(tree[0], *[self.of_tree(c) for c in tree[1:]])

    def typable(self, tree):
        if tree[0] == "L":
            return True  # both terminals exist at the root dual: asserted by the contract check
        return self.root_ok(tree[0], *[self.of_tree(c) for c in tree[1:]])


# ------------------------------------------------------------ tree sources ----
def all_trees(n, ops_un, ops_bin, cache=None):
    """Every tree of exactly n nodes over the operator lists (leaves are one symbol)."""
    cache = {} if cache is None else cache
    if n in cache:
        return cache[n]
    if n == 1:
        out = [LEAF]
    else:
        out = [(u, c) for u in ops_un for c in all_trees(n - 1, ops_un, ops_bin, cache)]
        for i in range(1, n - 1):
            ls = all_trees(i, ops_un, ops_bin, cache)
            rs = all_trees(n - 1 - i, ops_un, ops_bin, cache)
            out += [(b, l, r) for b in ops_bin for l in ls for r in rs]
    cache[n] = out
    return out


def random_tree(rng, n):
    """A random tree of n nodes; splits are often balanced so order 4 is reached."""
    if n == 1:
        return LEAF
    if n == 2 or rng.random() < 0.3:
        return (rng.choice(UNARY), random_tree(rng, n - 1))
    rest = n - 1
    if rng.random() < 0.5:
        left = max(1, min(rest - 1, rest // 2 + rng.randint(-1, 1)))
    else:
        left = rng.randint(1, rest - 1)
    return (rng.choice(BINARY), random_tree(rng, left), random_tree(rng, rest - left))


def size_of(t):
    return 1 if t[0] == "L" else 1 + sum(size_of(c) for c in t[1:])


# ----------------------------------------------------------------- checks ----
class Report:
    def __init__(self):
        self.lines = []
        self.failures = []

    def ok(self, msg):
        self.lines.append("ok    " + msg)
        print("ok    " + msg, flush=True)

    def info(self, msg):
        self.lines.append("      " + msg)
        print("      " + msg, flush=True)

    def fail(self, msg):
        self.failures.append(msg)
        print("FAIL  " + msg, flush=True)

    def expect(self, cond, msg, detail=""):
        if cond:
            self.ok(msg)
        else:
            self.fail(msg + (" :: " + detail if detail else ""))


def gen_into(tmp, name, ceiling, depth, hi):
    out = os.path.join(tmp, name)
    cmd = [sys.executable, "-I", os.path.join(HERE, "gen_tables.py"), "--variant", "custom",
           "--ceiling", str(ceiling) if ceiling else "none", "--depth", "on" if depth else "off",
           "--windows", "hi" if hi else "none", "--out", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return Tables(out)


def check_contract(rep, name, t, n_inputs=4):
    """The checks of kingdom_contract.rs `KingdomRelations::check` that hold at the table level."""
    known = set(t.duals)
    bad = [f["ref"] for f in t.functions if f["class"] in ("function", "root")
           and (f["out"] not in known or any(a not in known for a in f["in"].split(",")))]
    rep.expect(not bad, f"{name}: every row's output and argument is a known dual", str(bad[:5]))
    refs = [x["ref"] for x in t.types] + [f["ref"] for f in t.functions]
    rep.expect(len(set(refs)) == len(refs), f"{name}: every row has a unique stable ref")
    sig = [(f["constructor"], f["in"], f["out"], f["class"]) for f in t.fn + t.twin]
    rep.expect(len(set(sig)) == len(sig), f"{name}: no duplicate signature")
    rep.expect(all(len(f["in"].split(",")) == int(f["arity"]) for f in t.fn + t.twin),
               f"{name}: arity column matches the signature")
    root_in = [f for f in t.fn + t.twin if "ROOT" in f["in"].split(",")]
    rep.expect(not root_in, f"{name}: nothing takes ROOT as an argument")
    if t.has_root_twins:
        fn_sigs = sorted((f["constructor"], f["in"]) for f in t.fn)
        tw_sigs = sorted((f["constructor"], f["in"]) for f in t.twin)
        rep.expect(fn_sigs == tw_sigs and all(f["out"] == "ROOT" for f in t.twin),
                   f"{name}: ROOT twins are exactly the function rows with output ROOT")
    var = {f["out"] for f in t.term if f["instance"] == "var"}
    const = {f["out"] for f in t.term if f["instance"] == "const"}
    rep.expect(var == known and const == known,
               f"{name}: every dual (ROOT included) has a variable terminal and a ? ({len(known)} duals, "
               f"{len(known) * (n_inputs + 1)} terminals at {n_inputs} inputs)")


def check_alphabet(rep, name, t):
    inst = []
    for f in t.fn:
        if f["constructor"] not in inst:
            inst.append(f["constructor"])
    rep.expect(inst == WIDE_CTORS, f"{name}: one row set per symbol of wide() (18, in wide() order)", str(inst))
    # Every function has at least one row.
    absent_ctors = {f["constructor"] for f in t.absent}
    canon_ctors = {f["constructor"] for f in t.canon}
    return absent_ctors, canon_ctors


def engine_cross_checks(rep, absent_ctors, canon_ctors):
    """Alphabet and absent set against the sources, when the sources are there."""
    gpu = os.path.join(FULLER, "src", "gpu_eval.rs")
    with open(gpu) as fh:
        src = fh.read()
    body = src[src.index("pub fn from_math"):]
    body = body[:body.index("_ => return None")]
    read = set(re.findall(r'"(\w+)" => Op::', body))
    must_absent = read - {"Var", "Num"} - set(WIDE_CTORS)
    rep.expect(must_absent == absent_ctors,
               f"every other constructor math_to_nodes reads is an explicit absent row ({len(must_absent)}: "
               f"{', '.join(sorted(must_absent))})",
               f"missing {sorted(must_absent - absent_ctors)} extra {sorted(absent_ctors - must_absent)}")
    rep.expect(canon_ctors == {"Pow"}, "Pow(x,2) and Pow(x,3) canonicalise (pow2, pow3)")
    eng = os.path.join(PHYLU, "src", "evolve", "engine.rs")
    if not os.path.exists(eng):
        rep.info(f"wide() NOT cross-checked against the engine source ({eng} is absent)")
        return
    with open(eng) as fh:
        e = fh.read()
    i = e.index("pub fn wide(n_inputs")
    seg = e[i:e.index("symbols.extend", i)]
    ops = re.findall(r"Op::(\w+)", seg)
    ops = [o for o in ops if o != "Function"]
    rep.expect(ops == WIDE_CTORS, "alphabet equals SymbolTable::wide() read from phylu engine.rs", str(ops))
    # DEPTH_RAISING_OPS intersected with the alphabet equals RAISERS.
    j = e.index("pub const DEPTH_RAISING_OPS")
    raising = set(re.findall(r"Op::(\w+)", e[j:e.index("];", j)]))
    rep.expect(raising & set(WIDE_CTORS) == RAISERS,
               "raisers in the alphabet equal DEPTH_RAISING_OPS read from engine.rs")
    ordsrc = os.path.join(PHYLU, "src", "evolve", "order.rs")
    with open(ordsrc) as fh:
        o = fh.read()
    k = o.index("pub const RULES")
    rules = re.findall(r"\(Op::(\w+), (\d+), ([\w:]+)\)", o[k:o.index("];", k)])
    hi2 = {("ProtectedInv" if r == "Inv" else r) for r, lo, hi in rules if hi == "2"}
    rep.expect(hi2 & set(WIDE_CTORS) == WINDOWED,
               "windows (argument order at most 2) equal RULES read from order.rs")


def check_redundancy(rep, name, t, expect):
    n = sum(1 for f in t.twin if f["constructor"] == "Add") if t.has_root_twins else \
        sum(1 for f in t.fn if f["constructor"] == "Add")
    rep.expect(n == expect, f"{name}: row choices that decode to x + y = {n} (expected {expect})")


def rev4(t, N):
    """Sizing from the TSVs (separate code from gen_tables.bounds)."""
    root = t.root
    rows = t.all_rows_with_twins()
    mult = defaultdict(int)             # (argument dual, output dual) -> max multiplicity
    max_in = defaultdict(lambda: 1)     # output dual -> max arity of a producing row
    binary_out = set()
    for ctor, ins, out in rows:
        for a in set(ins):
            mult[(a, out)] = max(mult[(a, out)], ins.count(a))
        max_in[out] = max(max_in[out], len(ins))
        if len(ins) > 1:
            binary_out.add(out)
    reach, frontier = {root}, [root]
    while frontier:
        u = frontier.pop()
        for (a, o), m in list(mult.items()):
            if o == u and a not in reach:
                reach.add(a)
                frontier.append(a)
    per = {}
    for a in reach:
        per[a] = max(mult[(a, o)] for o in reach)
    h = {a: (N - 1) * per[a] + (1 if a == root else 0) for a in reach}
    L = 1 + sum(h[o] * (max_in[o] - 1) for o in reach)
    P = {a: (1 if a == root else 0) + sum(h[o] * mult[(a, o)] for o in reach) for a in reach}
    tail = {a: max(0, min(P[a], L + h[a]) - h[a]) for a in reach}
    dc = {a: min(P[a], L) for a in reach}
    width = sum(h[a] + tail[a] + dc[a] for a in reach)
    unary_only = [a for a in reach if a not in binary_out and a != root]
    return {"rows": len(rows), "duals": len(reach), "heads": sum(h.values()), "L": L, "tails": sum(tail.values()),
            "dc": sum(dc.values()), "width": width, "unary_only": len(unary_only),
            "unary_only_heads": sum(h[a] for a in unary_only), "root": (h[root], P[root], tail[root], dc[root])}


# rows with twins, duals + ROOT, heads, L, tails, Dc, width, unary-only (heads), ROOT (h,P,t,d)
PINNED = {
    ("order", 8): (1148, 12, 134, 100, 719, 769, 1622, 3, 35, (1, 1, 0, 1)),
    ("order", 17): (1148, 12, 305, 226, 1619, 1732, 3656, 3, 80, (1, 1, 0, 1)),
    ("order_depth", 8): (8264, 34, 400, 296, 4243, 4601, 9244, 9, 105, (1, 1, 0, 1)),
    ("order_depth", 17): (8264, 34, 913, 674, 9625, 10442, 20980, 9, 240, (1, 1, 0, 1)),
    ("depth", 17): (136, 4, 97, 98, 198, 263, 558, 0, 0, (1, 1, 0, 1)),
    ("order_depth_windows", 17): (8094, 34, 913, 674, 9625, 10442, 20980, 9, 240, (1, 1, 0, 1)),
}
WIDTHS = {
    ("single", 8): 47, ("single", 17): 101, ("single", 32): 191,
    ("depth", 8): 252, ("depth", 17): 558, ("depth", 32): 1068,
    ("order", 8): 1622, ("order", 17): 3656, ("order", 32): 7046,
    ("order_depth", 8): 9244, ("order_depth", 17): 20980, ("order_depth", 32): 40540,
    ("order_depth_windows", 8): 9244, ("order_depth_windows", 17): 20980, ("order_depth_windows", 32): 40540,
}
ROW_COUNTS = {"single": (18, 18), "order": (574, 1148), "depth": (68, 136), "order_depth": (4132, 8264),
              "order_depth_windows": (4047, 8094)}
VARIANT_PARAMS = {"single": (None, False, False), "order": (3, False, False), "depth": (None, True, False),
                  "order_depth": (3, True, False), "order_depth_windows": (3, True, True)}


def check_sizing(rep, variants):
    for (name, N), want in sorted(WIDTHS.items()):
        t = variants[name]
        got = rev4(t, N)["width"]
        rep.expect(got == want, f"width {name} N={N}: {got} words (plan {want})")
    for (name, N), want in sorted(PINNED.items()):
        r = rev4(variants[name], N)
        got = (r["rows"], r["duals"], r["heads"], r["L"], r["tails"], r["dc"], r["width"], r["unary_only"],
               r["unary_only_heads"], r["root"])
        rep.expect(got == want, f"pinned {name} N={N}: rows {r['rows']} duals {r['duals']} heads {r['heads']} "
                                f"L {r['L']} tails {r['tails']} Dc {r['dc']} width {r['width']} "
                                f"unary-only {r['unary_only']} ({r['unary_only_heads']}) root {r['root']}",
                   f"expected {want}")


def literal_exhaustive(rep, label, tables, params, max_n, ops_un, ops_bin):
    """All trees of up to max_n nodes: literal engine ports against the typing, per table."""
    cache = {}
    typers = [t.typer() for t in tables]
    counts = [[0, 0, 0] for _ in tables]   # total, illegal, disagreements
    firsts = []
    t0 = time.time()
    for n in range(1, max_n + 1):
        for tree in all_trees(n, ops_un, ops_bin, cache):
            facts = engine_facts(tree)
            for i, (typer, (ceiling, depth, hi)) in enumerate(zip(typers, params)):
                want = legal(facts, ceiling, depth, hi)
                got = typer.typable(tree)
                c = counts[i]
                c[0] += 1
                c[1] += not want
                if want != got:
                    c[2] += 1
                    if len(firsts) < 5:
                        firsts.append((label, i, tree, want, got))
    for i, c in enumerate(counts):
        rep.expect(c[2] == 0, f"{label} [{i}] literal ports, {c[0]:,} trees up to {max_n} nodes "
                              f"({c[1]:,} illegal): {c[2]} disagreements", str(firsts))
    rep.info(f"{label}: {time.time() - t0:.0f} s")


def incremental_exhaustive(rep, label, tables, params, max_n, ops_un, ops_bin):
    """All trees of up to max_n nodes enumerated without being stored (the last size is streamed).
    A node record is (fam, best, ties, order, depth, windows_bad, typing set ids...), updated by the
    engine's recurrences; cross-validated against the literal ports in `literal_exhaustive` and the
    random run."""
    typers = [t.typer() for t in tables]
    K = len(typers)

    def leaf():
        return (0, 1, 0, 1, 0, False) + tuple(tp.leaf for tp in typers)

    def un(ctor, r):
        fam, best, ties, order, dep, bad = r[:6]
        # unary: its own arm is the child's order, once (family none).
        nbad = bad or (ctor in WINDOWED and order > 2)
        return (0, order, 1, order, dep + (1 if ctor in RAISERS else 0), nbad) + tuple(
            tp.step(ctor, r[6 + i]) for i, tp in enumerate(typers))

    def bi(ctor, a, b):
        f = FAMILY[ctor]
        m = c = 0
        for r in (a, b):
            if r[0] == f:
                o, kk = r[1], r[2]
            else:
                o, kk = r[3], 1
            if o > m:
                m, c = o, kk
            elif o == m:
                c += kk
        order = m + (1 if c >= 2 else 0)
        dep = max(a[4], b[4]) + (1 if ctor in RAISERS else 0)
        return (f, m, c, order, dep, a[5] or b[5]) + tuple(
            tp.step(ctor, a[6 + i], b[6 + i]) for i, tp in enumerate(typers))

    by_size = {1: [leaf()]}
    counts = [[0, 0, 0] for _ in range(K)]
    firsts = []
    t0 = time.time()

    def finish(ctor, kids, rec):
        facts = (rec[3], rec[4], not rec[5])
        for i, (typer, (ceiling, depth, hi)) in enumerate(zip(typers, params)):
            want = legal(facts, ceiling, depth, hi)
            got = typer.root_ok(ctor, *[k[6 + i] for k in kids])
            c = counts[i]
            c[0] += 1
            c[1] += not want
            if want != got:
                c[2] += 1
                if len(firsts) < 5:
                    firsts.append((label, i, ctor, facts))

    for i in range(K):
        counts[i][0] += 1  # the single-leaf tree is legal and typable (terminals at the root exist)
    for n in range(2, max_n + 1):
        store = n < max_n
        out = []
        for ctor in ops_un:
            for r in by_size[n - 1]:
                rec = un(ctor, r)
                if store:
                    out.append(rec)
                finish(ctor, (r,), rec)
        for i in range(1, n - 1):
            ls, rs = by_size[i], by_size[n - 1 - i]
            for ctor in ops_bin:
                for a in ls:
                    for b in rs:
                        rec = bi(ctor, a, b)
                        if store:
                            out.append(rec)
                        finish(ctor, (a, b), rec)
        if store:
            by_size[n] = out
    for i, c in enumerate(counts):
        rep.expect(c[2] == 0, f"{label} [{i}] {c[0]:,} trees up to {max_n} nodes ({c[1]:,} illegal): "
                              f"{c[2]} disagreements", str(firsts))
    rep.info(f"{label}: {time.time() - t0:.0f} s")
    return counts


def random_run(rep, label, tables, params, count, max_nodes, seed):
    rng = random.Random(seed)
    typers = [t.typer() for t in tables]
    tot = [0] * len(tables)
    ill = [0] * len(tables)
    bad = [0] * len(tables)
    firsts = []
    nodes_max = 0
    t0 = time.time()
    for _ in range(count):
        n = rng.randint(1, max_nodes)
        tree = random_tree(rng, n)
        facts = engine_facts(tree)
        nodes_max = max(nodes_max, facts[0])
        for i, (typer, (ceiling, depth, hi)) in enumerate(zip(typers, params)):
            want = legal(facts, ceiling, depth, hi)
            got = typer.typable(tree)
            tot[i] += 1
            ill[i] += not want
            if want != got:
                bad[i] += 1
                if len(firsts) < 5:
                    firsts.append((i, tree, facts, want, got))
    for i in range(len(tables)):
        unlimited = params[i] == (None, False, False)
        rep.expect(bad[i] == 0 and (ill[i] == 0 if unlimited else 0 < ill[i] < tot[i]),
                   f"{label} [{i}] random: {tot[i]:,} trees up to {max_nodes} nodes ({ill[i]:,} illegal, "
                   f"highest order reached {nodes_max}): {bad[i]} disagreements", str(firsts))
    rep.info(f"{label}: {time.time() - t0:.0f} s")


def validate_recurrence(rep, count, seed):
    """The incremental recurrence used by the 9-node run equals the literal ports."""
    rng = random.Random(seed)
    for _ in range(count):
        tree = random_tree(rng, rng.randint(1, 40))

        def rec(t):
            if t[0] == "L":
                return (0, 1, 0, 1, 0, False)
            if len(t) == 2:
                r = rec(t[1])
                return (0, r[3], 1, r[3], r[4] + (t[0] in RAISERS), r[5] or (t[0] in WINDOWED and r[3] > 2))
            a, b = rec(t[1]), rec(t[2])
            f = FAMILY[t[0]]
            m = c = 0
            for r in (a, b):
                o, kk = (r[1], r[2]) if r[0] == f else (r[3], 1)
                if o > m:
                    m, c = o, kk
                elif o == m:
                    c += kk
            return (f, m, c, m + (c >= 2), max(a[4], b[4]) + (t[0] in RAISERS), a[5] or b[5])

        r = rec(tree)
        facts = engine_facts(tree)
        if (r[3], r[4], not r[5]) != facts:
            rep.fail(f"recurrence differs from the literal ports on {tree}: {r} vs {facts}")
            return
    rep.ok(f"incremental recurrence equals the literal stream_order / t_depth / windows on {count:,} random trees")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke test: 7 nodes exhaustive, 20k random")
    a = ap.parse_args()
    rep = Report()
    t0 = time.time()
    print(f"check_types: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
          f"{' (QUICK, not the plan numbers)' if a.quick else ''}", flush=True)

    # ---- the committed tables
    variants = {n: Tables(os.path.join(HERE, n)) for n in VARIANT_PARAMS}
    for name, t in variants.items():
        check_contract(rep, name, t)
        absent, canon = check_alphabet(rep, name, t)
        if name == "single":
            engine_cross_checks(rep, absent, canon)
        got = (len(t.fn), len(t.fn) + len(t.twin) if t.has_root_twins else len(t.fn))
        rep.expect(got == ROW_COUNTS[name], f"{name}: {got[0]} function rows, {got[1]} with root twins "
                                            f"(plan {ROW_COUNTS[name]})")
    check_redundancy(rep, "single", variants["single"], 1)
    check_redundancy(rep, "order", variants["order"], 105)
    check_redundancy(rep, "order_depth", variants["order_depth"], 945)
    check_sizing(rep, variants)

    # ---- the committed tables equal a fresh generation
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "gen_tables.py"), "--out", tmp], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # --out with no custom parameters puts all variants under the same path; regenerate one by one.
        for name, (c, d, h) in VARIANT_PARAMS.items():
            out = os.path.join(tmp, name)
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "gen_tables.py"), "--variant", name,
                            "--out", out], check=True, stdout=subprocess.DEVNULL)
            same = all(open(os.path.join(out, f)).read() == open(os.path.join(HERE, name, f)).read()
                       for f in ("types.tsv", "functions.tsv", "TYPES.md", "symbols.md"))
            rep.expect(same, f"{name}: committed tables equal a fresh generation")

        # ---- types against the rules (plan 2.6 / test #3)
        exh_n = 5 if a.quick else 6
        nine_n = 7 if a.quick else 9
        n_rand = 5000 if a.quick else 60000
        combos = [(False, False), (True, False), (False, True), (True, True)]   # (depth, hi)
        validate_recurrence(rep, 2000 if a.quick else 20000, 11)

        t2 = [gen_into(tmp, f"r2_{int(d)}{int(h)}", 2, d, h) for d, h in combos]
        p2 = [(2, d, h) for d, h in combos]
        literal_exhaustive(rep, "ceiling 2, all 18 operators", t2, p2, exh_n, UNARY, BINARY)

        five = ["Sin", "ProtectedSqrt", "Neg", "Pow2", "ProtectedExp"]
        order_only, depth_win = t2[0], t2[3]
        counts = incremental_exhaustive(rep, "ceiling 2, 5 unary + 4 binary", [order_only, t2[1], depth_win],
                                        [(2, False, False), (2, True, False), (2, True, True)], nine_n, five, BINARY)
        if not a.quick:
            rep.info(f"plan 2.6 totals: 4,335,881 trees; illegal {counts[0][1]:,} (order only; plan 18,448), "
                     f"{counts[2][1]:,} (depth and windows; plan 2,324,248)")
            rep.expect(counts[0][0] == 4335881, "exhaustive 9-node tree count is 4,335,881")

        t3 = [gen_into(tmp, f"r3_{int(d)}{int(h)}", 3, d, h) for d, h in combos]
        p3 = [(3, d, h) for d, h in combos]
        random_run(rep, "ceiling 3, up to 40 nodes", t3, p3, n_rand, 40, 20261020)
        random_run(rep, "ceiling 2, up to 30 nodes, depth and windows", [depth_win], [(2, True, True)],
                   n_rand, 30, 20261021)
        # The committed variants, as shipped, against the same rules.
        names = ["order", "order_depth", "order_depth_windows", "depth", "single"]
        random_run(rep, "committed variants, up to 40 nodes", [variants[n] for n in names],
                   [VARIANT_PARAMS[n] for n in names], n_rand, 40, 20261022)

    dt = time.time() - t0
    print(f"\n{len(rep.failures)} failures, {dt:.0f} s", flush=True)
    if rep.failures:
        for f in rep.failures:
            print("  FAIL " + f)
        sys.exit(1)


if __name__ == "__main__":
    main()
