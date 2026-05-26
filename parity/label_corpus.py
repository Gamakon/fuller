#!/usr/bin/env python3
"""Offline labeler for the before->after simplification corpus.

Input:  /tmp/gamak_simplify_corpus.jsonl (emitted by the HFF sweep's
        instrumented visit_subtree — env GAMAK_SIMPLIFY_CORPUS).
        Each line: {before, after_simplify, after_snap, changed, ...} as srepr.

Output: parity/corpus/labeled_simplify.jsonl, each line adds:
        - family: which sympy sub-simplifier reproduces `after_simplify`
                  from `before` (the classifier LABEL). One of
                  {powsimp, radsimp, trigsimp, ratsimp, cancel, expand,
                   none, multi, full_only}.
        - features: cheap structural features of `before` (the classifier X).

This runs OFFLINE and IS allowed to use sympy (it is analysis of sympy's own
output, not part of the gamakAST crate). The crate never imports sympy.

Usage: python parity/label_corpus.py [in.jsonl] [out.jsonl]
"""
from __future__ import annotations

import json
import os
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = "/tmp/gamak_simplify_corpus.jsonl"
DEFAULT_OUT = os.path.join(HERE, "corpus", "labeled_simplify.jsonl")

# Candidate sub-simplifiers, in rough cheap->expensive order. The first whose
# output (srepr) matches `after_simplify` is the label. If several match, label
# = "multi" (the cheapest is recorded separately as `first_family`).
SUBSIMPLIFIERS = [
    ("powsimp", lambda e: sp.powsimp(e, force=False)),
    ("expand", sp.expand),
    ("cancel", sp.cancel),
    ("radsimp", sp.radsimp),
    ("ratsimp", sp.ratsimp),
    ("trigsimp", sp.trigsimp),
]


def structural_features(expr) -> dict:
    """Cheap, first-inspection structural features for the classifier X."""
    ops = sp.count_ops(expr, visual=True)
    counts = {}
    # count_ops(visual=True) returns a symbolic sum like 2*MUL + ADD; pull terms
    for term in sp.Add.make_args(ops):
        c, sym = term.as_coeff_Mul()
        counts[str(sym)] = int(c) if c.is_Integer else 1
    atoms = expr.atoms(sp.Function)
    fnames = {type(f).__name__ for f in atoms}
    return {
        "n_ops": int(sp.count_ops(expr)),
        "has_trig": int(bool({"sin", "cos", "tan"} & fnames)),
        "has_exp_log": int(bool({"exp", "log"} & fnames)),
        "has_sqrt": int(expr.has(sp.sqrt) or any(
            getattr(p, "exp", None) == sp.Rational(1, 2) for p in expr.atoms(sp.Pow))),
        "has_pow": int(bool(expr.atoms(sp.Pow))),
        "has_div": int(any(getattr(p, "exp", 0) is not None and
                           getattr(p, "exp", 0).is_negative
                           for p in expr.atoms(sp.Pow) if p.is_Pow)),
        "n_add": counts.get("ADD", 0),
        "n_mul": counts.get("MUL", 0),
        "n_pow": counts.get("POW", 0),
        "n_symbols": len(expr.free_symbols),
        "depth": _depth(expr),
    }


def _depth(expr) -> int:
    if not expr.args:
        return 1
    return 1 + max(_depth(a) for a in expr.args)


def label_one(before_srepr: str, after_srepr: str) -> tuple[str, list[str]]:
    """Return (label, all_matching_families) for one pair."""
    try:
        before = sp.sympify(before_srepr)
        after = sp.sympify(after_srepr)
    except Exception:
        return "parse_error", []
    if sp.srepr(before) == after_srepr:
        return "none", []  # nothing changed
    matches = []
    for name, fn in SUBSIMPLIFIERS:
        try:
            if sp.srepr(fn(before)) == after_srepr:
                matches.append(name)
        except Exception:
            continue
    if not matches:
        # full simplify reproduced it but no single sub-pass did
        return "full_only", []
    if len(matches) == 1:
        return matches[0], matches
    return "multi", matches


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    if not os.path.exists(in_path):
        print(f"no corpus at {in_path} yet — nothing to label")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n, labeled, hist = 0, 0, {}
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            before = rec.get("before")
            after = rec.get("after_simplify")
            if not before or not after or not rec.get("changed", False):
                continue  # only learn from pairs that actually simplified
            label, matches = label_one(before, after)
            try:
                feats = structural_features(sp.sympify(before))
            except Exception:
                feats = {}
            # first_family = the single cheapest sub-pass that reproduces the
            # result (SUBSIMPLIFIERS is cheap->expensive ordered). Gives the
            # classifier one clean primary label even when several match.
            first_family = matches[0] if matches else label
            hist[label] = hist.get(label, 0) + 1
            labeled += 1
            fout.write(json.dumps({
                "before": before, "after": after,
                "family": label, "first_family": first_family,
                "all_families": matches, "features": feats,
            }) + "\n")
    print(f"read {n} records, labeled {labeled} simplifying pairs -> {out_path}")
    print("family histogram:")
    for k in sorted(hist, key=lambda k: -hist[k]):
        print(f"  {k:<12} {hist[k]}")


if __name__ == "__main__":
    main()
