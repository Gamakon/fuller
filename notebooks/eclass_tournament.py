"""Equivalence-class tournament chart (paper figure, one HOF).

Pipeline (matches the Method section):
  1. Load a HOF pickle, take gene[0]'s karva tokens.
  2. denoise_karva_candidates -> the equivalence class (goal-equivalent forms).
  3. For each form's Math s-expr, walk the AST and collect the pattern-dependent
     measures from the Appendix library. Each measure is bounded to [0,1], 0=best.
     node_count (parsimony) always fires, so the vector is never all-zero.
  4. Augment (HFF-TrueNorth): e = max(0, 1 - ||x||^2 / k); theta = arccos(e).
  5. CDF-correct each angle in its own dimension k via the regularised
     incomplete beta function -> percentile, cross-comparable.
  6. Scatter: x = D (patterns fired), y = CDF-corrected angle. Original gene
     marked as an open circle; the chosen best (min percentile) as a star.

Writes images/eclass_tournament_<problem>.png. Jupytext conversion later.
"""
from __future__ import annotations
import sys, os, re, pickle, math, glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betainc

# --- engine modules on path (to unpickle Individuals) ----------------------
_HFF = "/Users/andrewmorgan/Dev/kaito/hff/notebooks"
_SRB = os.path.join(_HFF, "..", "srbench_submission", "algorithms", "hff-sr")
for _p in (os.path.abspath(_SRB), _HFF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _denoise_op import _token_tuple, SEMANTIC_ID_MAP  # noqa: E402
from fuller import eclass_extract_hff, master_pset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)

TRANSC = {"Sin", "Cos", "Tan", "Exp", "Log", "Tanh"}


# --- tiny Math s-expr parser -> nested tuple (op, [children]) | ("Var",n) | ("Num",v)
def parse(s: str):
    toks = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', s)
    pos = 0

    def walk():
        nonlocal pos
        assert toks[pos] == "("
        pos += 1
        head = toks[pos]; pos += 1
        if head == "Num":
            v = float(toks[pos]); pos += 1; pos += 1  # value, ")"
            return ("Num", v)
        if head == "Var":
            n = toks[pos].strip('"'); pos += 1; pos += 1
            return ("Var", n)
        kids = []
        while toks[pos] != ")":
            kids.append(walk())
        pos += 1
        return (head, kids)

    return walk()


def children(node):
    return node[1] if (isinstance(node[1], list)) else []


def walk_nodes(node):
    yield node
    if isinstance(node[1], list):
        for c in node[1]:
            yield from walk_nodes(c)


# --- the bounded measures (subset of the Appendix table; 0=best, in [0,1]) ---
def sat(c, s=1.0):
    return 1.0 - 1.0 / (1.0 + c / s)


def is_op(node, names):
    return isinstance(node[1], list) and node[0] in names


def measures(expr: str) -> dict:
    """Return {name: value in [0,1]} for every pattern that FIRES on this form."""
    root = parse(expr)
    nodes = list(walk_nodes(root))
    out = {}

    # node_count — ALWAYS fires (parsimony), keeps the vector non-zero.
    out["node_count"] = sat(len(nodes), s=12.0)

    # max_tree_depth — always.
    def depth(n):
        ch = children(n)
        return 1 + (max((depth(c) for c in ch), default=0))
    out["max_tree_depth"] = sat(depth(root) - 1, s=6.0)

    # transcendental_count
    tc = sum(1 for n in nodes if n[0] in TRANSC)
    if tc:
        out["transcendental_count"] = sat(tc, s=2.0)

    # transc_nesting_depth — longest transc-inside-transc chain
    def transc_chain(n):
        ch = children(n)
        below = max((transc_chain(c) for c in ch), default=0)
        return below + 1 if n[0] in TRANSC else below

    def transc_nest(n):
        # depth of nesting where transc contains transc
        ch = children(n)
        best = max((transc_nest(c) for c in ch), default=0)
        if n[0] in TRANSC:
            inner = max((transc_chain(c) for c in ch), default=0)
            best = max(best, inner)  # +0: count transc-within-transc levels
        return best
    tn = transc_nest(root)
    if tn:
        out["transc_nesting_depth"] = sat(tn, s=1.0)

    # self_nested_transc_count — same transc fn directly inside itself
    def self_nest(n):
        cnt = 0
        for c in children(n):
            if n[0] in TRANSC and c[0] == n[0]:
                cnt += 1
            cnt += self_nest(c)
        return cnt
    sn = self_nest(root)
    if sn:
        out["self_nested_transc_count"] = sat(sn, s=1.0)

    # distinct_constant_count
    consts = {round(n[1], 9) for n in nodes if n[0] == "Num"}
    if consts:
        out["distinct_constant_count"] = sat(len(consts), s=4.0)

    # sign_op_count
    so = sum(1 for n in nodes if n[0] in ("Neg", "Sub"))
    if so:
        out["sign_op_count"] = sat(so, s=3.0)

    # constant_to_variable_ratio — always (ratio, already [0,1])
    n_c = sum(1 for n in nodes if n[0] == "Num")
    n_v = sum(1 for n in nodes if n[0] == "Var")
    if n_c + n_v > 0:
        out["const_to_var_ratio"] = n_c / (n_c + n_v)

    return out


# --- augmentation + CDF correction ------------------------------------------
def angle(vec: np.ndarray) -> float:
    k = len(vec)
    nrm2 = float(np.dot(vec, vec))
    e = max(0.0, 1.0 - nrm2 / k)
    return math.acos(min(1.0, e))  # arccos(e); 0 iff all measures 0


def cdf_correct(theta: float, k: int) -> float:
    # P(theta <= t) for uniform points on S^{k-1}
    if k < 2:
        return theta / math.pi
    return float(betainc((k - 1) / 2.0, 0.5, math.sin(theta) ** 2))


# --- run one HOF -------------------------------------------------------------
def run(pkl_path: str):
    """Decode gene[0]'s LIVE kexpression, enumerate + HFF-rank the e-class in
    Rust (single source of truth for the percentile), and derive D (patterns
    fired) per form in Python for the x-axis."""
    d = pickle.load(open(pkl_path, "rb"))
    name = os.path.basename(pkl_path).replace("hff_hof_", "").replace(".pkl", "")
    gene = d["hof"][0][0]

    # Live k-expression only — the dead tail carries '?'-named RNC placeholders.
    ktoks = [_token_tuple(t) for t in gene.kexpression]
    variables = sorted({v for k, v in ktoks if k == "var"})
    rnc = sorted({v for k, v in ktoks if k == "num"})
    sem = {n: a for n, a in master_pset()}
    functions = {
        v_: (SEMANTIC_ID_MAP.get(v_, v_), sem.get(SEMANTIC_ID_MAP.get(v_, v_), 2))
        for k_, v_ in ktoks if k_ == "func"
    }
    tail = [("var", variables[0])] if variables else [("num", 1.0)]

    # Rust HFF extractor: [(angle_percentile, expr)] best-first. The percentile
    # is authoritative (computed by the same hff_core the GA uses).
    # "wide" = algebra+powers+distribute+comm/assoc: the form-generating family
    # that populates the e-class so the angular tournament has members to rank.
    # Bounded iters keeps the combinatorial growth in check.
    ranked = eclass_extract_hff(
        ktoks, tail, variables, functions, rnc,
        family="wide", k=128, iters=6,
    )
    print(f"{name}: {len(ranked)} equivalent forms")

    # The original form is the LAST (worst-percentile) — or, more robustly, the
    # one whose expr matches the input decode. Mark by largest node-count proxy.
    pts = []
    for pc, expr in ranked:
        m = measures(expr)           # Python re-derivation, ONLY for D (x-axis)
        dim = len(m)                 # number of patterns that fired on this form
        pts.append((dim, pc, expr))
    return name, pts


def plot(name, pts):
    # pts: [(dim, percentile, expr)] from the Rust HFF ranking (best-first).
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    best = int(np.argmin(ys))   # lowest CDF-corrected angle = tournament winner
    worst = int(np.argmax(ys))  # the noisiest equivalent form (proxy: original)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(xs, ys, c="tab:blue", s=30, alpha=0.6, label="equivalent form")
    ax.scatter(xs[worst], ys[worst], facecolors="none", edgecolors="black",
               s=130, linewidths=1.6, label="noisiest form")
    ax.scatter(xs[best], ys[best], marker="*", c="tab:red", s=260,
               edgecolors="black", linewidths=0.6, label="HFF-selected best", zorder=5)
    ax.set_xlabel("D — number of pattern rules fired")
    ax.set_ylabel("CDF-corrected HFF angle (percentile, 0=best)")
    ax.set_title(f"Equivalence-class tournament — {name}")
    # integer x ticks (D is a count)
    if len(set(xs)) > 1:
        ax.set_xticks(range(int(xs.min()), int(xs.max()) + 1))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(IMG, f"eclass_tournament_{name}.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    pkl = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/andrewmorgan/Dev/kaito/fuller/hof_pickles/hff_hof_lean_I_9_18.pkl"
    name, pts = run(pkl)
    if pts:
        plot(name, pts)
    else:
        print("no scorable forms")
