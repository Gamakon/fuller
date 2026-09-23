"""Patch the recovery notebook: the linter offers EVERY option, HFF picks.

Apply only when no sweep is running — each problem run reads the notebook
source at start-up.

  * HFF_LINT_EXACTNESS (default "finite"): the weakest rule level the linter may
    use. Exactness is a label on a candidate, not a gate: every candidate is
    scored on data, and HFF (performance + parsimony) ranks the original
    against all of them.
  * the linter is handed the e-class rows, so it also offers the data-justified
    prunes (label "prune").
  * grafts are counted by the level of the form that was grafted.
"""
import sys

p = "/Users/andrewmorgan/Dev/gamakon/hff/notebooks/v1.0.4_Multidemic_SymbolicEquationRecovery.py"
s = open(p).read()


def swap(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"anchor not found exactly once:\n{old[:120]}")
    s = s.replace(old, new)


swap('''if FULLER_MODE not in ("lint", "egglog", "off"):
    raise ValueError(f"HFF_FULLER must be lint, egglog or off, not {FULLER_MODE!r}")
''', '''if FULLER_MODE not in ("lint", "egglog", "off"):
    raise ValueError(f"HFF_FULLER must be lint, egglog or off, not {FULLER_MODE!r}")
# How far the linter may go. A variant is a MUTATION: it is scored on the data
# like any other candidate, so fidelity to its parent — a guess — is not the
# test; performance and parsimony are, and HFF applies it. Exactness is kept as
# a label on each candidate (bit / rounding / finite / prune), not as a gate.
#   finite (default): every rule, plus the data-justified prunes.
#   rounding, bit:    only rules at least that faithful, and no prunes.
LINT_EXACTNESS = os.environ.get("HFF_LINT_EXACTNESS", "finite")
if LINT_EXACTNESS not in ("bit", "rounding", "finite"):
    raise ValueError(f"HFF_LINT_EXACTNESS must be bit, rounding or finite, not {LINT_EXACTNESS!r}")
_NB_GRAFTS_BY_LEVEL: dict = {}
_NB_OFFERED_BY_LEVEL: dict = {}
''')

swap('''            results = _f._fuller.lint_karva_candidates_batch(
                list(todo.values()), variables, _build_functions_dict(pset),
                k_variants=ECLASS_K, rng_seed=0, target_head_length=None)
''', '''            results = _f._fuller.lint_karva_candidates_batch(
                list(todo.values()), variables, _build_functions_dict(pset),
                k_variants=ECLASS_K, rng_seed=0, target_head_length=None,
                exactness=LINT_EXACTNESS,
                rows=_NB_ECLASS_ROWS if LINT_EXACTNESS == "finite" else [])
''')

swap('''            props = [(c["head"], c["tail"], int(c["cost"]), (), False)
                     for c in sorted(res["candidates"], key=lambda c: c["cost"])
''', '''            for c in res["candidates"]:
                _lv = c.get("level", "egglog")
                _NB_OFFERED_BY_LEVEL[_lv] = _NB_OFFERED_BY_LEVEL.get(_lv, 0) + 1
            props = [(c["head"], c["tail"], int(c["cost"]), (), False,
                      c.get("level", "egglog"))
                     for c in sorted(res["candidates"], key=lambda c: c["cost"])
''')

swap('''        for head, tail, cost, consts, is_snap in props:
            if consts:
''', '''        for head, tail, cost, consts, is_snap, *_rest in props:
            _level = _rest[0] if _rest else ("snap" if is_snap else "egglog")
            if consts:
''')

swap('''            dev = hgh._resolve_rnc(new_gene, finalTerminals)
            if dev is None:
                raise RuntimeError(
                    f"rebuilt variant of {okey} does not resolve for the "
                    "device although its original did")
''', '''            new_gene.fuller_level = _level      # how this form was obtained
            dev = hgh._resolve_rnc(new_gene, finalTerminals)
            if dev is None:
                raise RuntimeError(
                    f"rebuilt variant of {okey} does not resolve for the "
                    "device although its original did")
''')

swap('''        _NB_GPU_STATS["grafts"] += 1
''', '''        _NB_GPU_STATS["grafts"] += 1
        _glv = getattr(new_gene, "fuller_level", "?")
        _NB_GRAFTS_BY_LEVEL[_glv] = _NB_GRAFTS_BY_LEVEL.get(_glv, 0) + 1
''')

swap('''experiment["graft_mode"] = GRAFT_MODE
''', '''experiment["graft_mode"] = GRAFT_MODE
experiment["lint_exactness"] = LINT_EXACTNESS if FULLER_MODE == "lint" else "n/a"
experiment["offered_by_level"] = dict(sorted(_NB_OFFERED_BY_LEVEL.items()))
experiment["grafts_by_level"] = dict(sorted(_NB_GRAFTS_BY_LEVEL.items()))
''')

open(p, "w").write(s)
print("patched")
