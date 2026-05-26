#!/usr/bin/env python3
"""Parity corpus generator — runs SymPy ONCE, offline, to produce ground-truth
(input, target) pairs as gamakAST `Math` s-expressions.

SymPy is used here and ONLY here: to generate targets. It never appears in the
scorer (parity/score.py), which uses gamakAST's own tools. This keeps the
parity claim honest — SymPy sets the homework; we grade it ourselves.

Output: parity/corpus/<module>.jsonl, each line {"input": <math>, "target": <math>}.

Usage: python parity/gen_corpus.py            # all modules
       python parity/gen_corpus.py trigsimp   # one module
"""
from __future__ import annotations

import json
import os
import random
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")

# Variables available to generated expressions (real-valued, matching the
# real-domain evaluator gamakAST uses).
X, Y, Z = sp.symbols("x y z", real=True)
VARS = [X, Y, Z]
NUM_LEAVES = [sp.Integer(-2), sp.Integer(-1), sp.Integer(2), sp.Integer(3), sp.Rational(1, 2)]

# Per-module operator sets + the SymPy simplifier that defines the target.
MODULES = {
    "powsimp":  dict(target=sp.powsimp,  bins=[sp.Add, sp.Mul], uns=[],
                     leaves=VARS + NUM_LEAVES, pow_exps=[2, 3, -1]),
    "radsimp":  dict(target=sp.radsimp,  bins=[sp.Add, sp.Mul], uns=[sp.sqrt],
                     leaves=VARS + NUM_LEAVES, pow_exps=[2]),
    "trigsimp": dict(target=sp.trigsimp, bins=[sp.Add, sp.Mul], uns=[sp.sin, sp.cos, sp.tan],
                     leaves=VARS + NUM_LEAVES, pow_exps=[2]),
    "ratsimp":  dict(target=sp.ratsimp,  bins=[sp.Add, sp.Mul], uns=[],
                     leaves=VARS + NUM_LEAVES, pow_exps=[2, -1]),
    "simplify": dict(target=sp.simplify, bins=[sp.Add, sp.Mul], uns=[sp.sin, sp.cos, sp.sqrt, sp.exp, sp.log],
                     leaves=VARS + NUM_LEAVES, pow_exps=[2, 3, -1]),
}

# sympy func -> (Math constructor, arity). Real-domain only.
SYMPY_TO_MATH = {
    sp.Add: ("Add", 2), sp.Mul: ("Mul", 2),
    sp.sin: ("Sin", 1), sp.cos: ("Cos", 1), sp.tan: ("Tan", 1),
    sp.exp: ("Exp", 1), sp.log: ("Log", 1), sp.tanh: ("Tanh", 1),
    sp.Abs: ("Abs", 1),
}


def gen_expr(depth, cfg):
    if depth <= 0 or random.random() < 0.3:
        return random.choice(cfg["leaves"])
    roll = random.random()
    if cfg["uns"] and roll < 0.4:
        return random.choice(cfg["uns"])(gen_expr(depth - 1, cfg))
    if roll < 0.6 and cfg["pow_exps"]:
        return gen_expr(depth - 1, cfg) ** random.choice(cfg["pow_exps"])
    op = random.choice(cfg["bins"])
    return op(gen_expr(depth - 1, cfg), gen_expr(depth - 1, cfg))


def to_math(expr):
    """sympy expression -> gamakAST Math s-expression, or None if unconvertible."""
    if expr.is_Number:
        return f"(Num {float(expr)})"
    if expr.is_Symbol:
        return f'(Var "{expr.name}")'
    if expr.is_Pow:
        base, exp = expr.args
        b = to_math(base)
        if b is None:
            return None
        if exp == 2:
            return f"(Pow2 {b})"
        if exp == 3:
            return f"(Pow3 {b})"
        if exp == -1:
            return f"(Inv {b})"
        if exp == sp.Rational(1, 2):
            return f"(Sqrt {b})"
        e = to_math(exp)
        return None if e is None else f"(Pow {b} {e})"
    spec = SYMPY_TO_MATH.get(expr.func)
    if spec is None:
        return None
    ctor, arity = spec
    args = list(expr.args)
    # binarise n-ary Add/Mul left-leaning
    if ctor in ("Add", "Mul") and len(args) > 2:
        acc = to_math(args[0])
        for a in args[1:]:
            am = to_math(a)
            if acc is None or am is None:
                return None
            acc = f"({ctor} {acc} {am})"
        return acc
    if len(args) != arity:
        return None
    parts = [to_math(a) for a in args]
    if any(p is None for p in parts):
        return None
    return f"({ctor} {' '.join(parts)})"


def build(module, n=200, seed=0, depth=4):
    cfg = MODULES[module]
    random.seed(seed)
    pairs, tries = [], 0
    while len(pairs) < n and tries < n * 200:
        tries += 1
        e = gen_expr(depth, cfg)
        try:
            tgt = cfg["target"](e)
        except Exception:
            continue
        if sp.srepr(e) == sp.srepr(tgt):
            continue  # nothing simplified — no parity signal
        im, tm = to_math(e), to_math(tgt)
        if im is None or tm is None or im == tm:
            continue
        pairs.append({"input": im, "target": tm})
    return pairs


def main():
    os.makedirs(CORPUS, exist_ok=True)
    mods = sys.argv[1:] or list(MODULES)
    for m in mods:
        pairs = build(m)
        path = os.path.join(CORPUS, f"{m}.jsonl")
        with open(path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"{m}: {len(pairs)} pairs -> {path}")


if __name__ == "__main__":
    main()
