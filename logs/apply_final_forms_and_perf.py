"""Patch the recovery notebook (apply only when no sweep is running):

 1. per-stage timers, printed as a [perf] table and stored in the record;
 2. the end-of-run simplification done the join's way — generate candidate
    forms with fuller, EXECUTE each on train/val/extrap, let HFF (performance +
    parsimony) rank them — and three model strings for the same best
    individual, so the uplift of the sympy step can be measured:
        model_no_sympy          fuller only, printed by fuller, sympy never called
        model_lint_then_sympy   the HFF-picked fuller form, then capped sympy
        model_sympy             today's path (what `discovered_expr` already is)
"""
import sys

p = "/Users/andrewmorgan/Dev/gamakon/hff/notebooks/v1.0.4_Multidemic_SymbolicEquationRecovery.py"
s = open(p).read()


def swap(old, new, count=1):
    global s
    if s.count(old) != count:
        sys.exit(f"anchor found {s.count(old)}x, wanted {count}:\n{old[:160]}")
    s = s.replace(old, new)


# ---- 1. timers -----------------------------------------------------------
swap('''_NB_ECLASS_CACHE: dict = {}
''', '''_NB_ECLASS_CACHE: dict = {}
# Wall seconds per stage of a run. Everything that takes time has a name here,
# so the time engineering wins back can be SEEN and spent on a larger search.
_NB_PERF: dict = {}


def _perf_add(stage: str, t0: float) -> None:
    _NB_PERF[stage] = _NB_PERF.get(stage, 0.0) + (time.perf_counter() - t0)
''')

swap('''        for idx, deme in enumerate(demes):
            if idx in HALTED_DEMES:
                continue      # wrapper-culled: frozen, no select/mutate/cross
            _ts = _island_tournsize(idx)
''', '''        _pt = time.perf_counter()
        for idx, deme in enumerate(demes):
            if idx in HALTED_DEMES:
                continue      # wrapper-culled: frozen, no select/mutate/cross
            _ts = _island_tournsize(idx)
''')
swap('''        # Phase 2 — ONE evaluation for every island's unevaluated individuals.
        _nb_evaluate_islands([[ind for ind in deme if not ind.fitness.valid]
                              for deme in demes])

        # Phase 3 — record. A halted island logs its frozen state, evals=0.
''', '''        _perf_add("evolve: select + mutate + crossover", _pt)
        # Phase 2 — ONE evaluation for every island's unevaluated individuals.
        _pt = time.perf_counter()
        _nb_evaluate_islands([[ind for ind in deme if not ind.fitness.valid]
                              for deme in demes])
        _perf_add("evaluate: total (tokens, simplifier, device, scoring, HFF, polish)", _pt)

        # Phase 3 — record. A halted island logs its frozen state, evals=0.
        _pt = time.perf_counter()
''')
swap('''        _nb_f64_polish_hof()
        _NB_GPU_STATS["gen_seconds"] += time.perf_counter() - _gen_t0
''', '''        _perf_add("record: log + hall of fame", _pt)
        _pt = time.perf_counter()
        _nb_f64_polish_hof()
        _perf_add("hall of fame: f64 re-fit", _pt)
        _NB_GPU_STATS["gen_seconds"] += time.perf_counter() - _gen_t0
''')
swap('''            assign_fitness_batch(g, raw[at:at + len(g)])
            if os.environ.get("HFF_GPU") == "1":
                _nb_f64_polish(g)
''', '''            _pt = time.perf_counter()
            assign_fitness_batch(g, raw[at:at + len(g)])
            _perf_add("  of evaluate: HFF rank + graft", _pt)
            if os.environ.get("HFF_GPU") == "1":
                _pt = time.perf_counter()
                _nb_f64_polish(g)
                _perf_add("  of evaluate: f64 re-fit of leaders", _pt)
''')

# ---- 2. end-of-run: three model strings, HFF picks the fuller form -------
swap('''# %% [markdown]
# ## 4.2 Equation-recovery scoring (structural + numerical)
''', '''# %% [markdown]
# ## 4.1b The reported form, the join's way: generate, EXECUTE, let HFF rank
#
# Three model strings for the SAME best individual, so the uplift of the sympy
# completion step is a measurement, not an opinion. SRBench runs its own sympy
# `simplify` on whatever string it is handed, so ours may add nothing.

# %%
def _nb_chromosome_math(ind, wrapper_name, a, b):
    """The whole reported model as one fuller `Math` expression:
    a * WRAPPER(LINKER(genes)) + b. None if a gene does not decode."""
    import fuller._fuller as _ff
    variables = [t.name for t in pset.terminals
                 if (isinstance(t, SymbolTerminal) or t.value is None) and t.name != "?"]
    genes = []
    for g in ind:
        toks = _nb_geppy_tokens(g)
        if toks is None:
            return None
        genes.append(_ff.karva_to_math(toks[0], toks[1], variables,
                                       _build_functions_dict(pset), []))
    linker = ind.linker.__name__.replace("round_", "")
    if len(genes) == 1:
        body = genes[0]
    elif linker == "mulval":
        body = genes[0]
        for g in genes[1:]:
            body = f"(Mul {body} {g})"
    else:
        body = genes[0]
        for g in genes[1:]:
            body = f"(Add {body} {g})"
        if linker == "avgval":
            body = f"(Div {body} (Num {float(len(genes))!r}))"
        elif linker != "addval":
            return None
    if ind.linker.__name__.startswith("round_"):
        return None                      # rounding has no Math constructor
    body = {"identity": body,
            "log_abs": f"(Log (Abs {body}))",
            "sqrt_abs": f"(Sqrt (Abs {body}))",
            "exp": f"(Exp {body})",
            "square": f"(Pow2 {body})"}.get(wrapper_name)
    if body is None:
        return None
    return f"(Add (Mul (Num {float(a)!r}) {body}) (Num {float(b)!r}))"


def _nb_final_forms(ind, wrapper_name):
    """Candidate forms of the reported model, each executed on train / val /
    extrap; HFF ranks them on the six error terms plus size. Returns
    (winner, table) or (None, reason)."""
    import fuller._fuller as _ff
    math = _nb_chromosome_math(ind, wrapper_name, ind.a, ind.b)
    if math is None:
        return None, "the model has no Math form (undecodable gene, or a rounding linker)"
    cols = list(finalTerminals)
    rows = train[cols].iloc[:256].to_dict(orient="records")
    cands = _ff.lint_forms(math, cols, "finite", 8, rows)
    var_tr, var_va = float(np.var(Y)), float(np.var(Y_val))
    frames = {"tr": (train, Y), "va": (validation, Y_val), "ex": (extrapolation, Y_extrap)}
    table = []
    for c in cands:
        pred = {k: np.asarray(_ff.eval_math(c["math"], {n: f[n].astype(float).tolist() for n in cols}))
                for k, (f, _) in frames.items()}
        mse = {k: hgh.safe_mse(frames[k][1], pred[k]) for k in frames}
        vec = [mse["tr"], mse["va"], float(np.max(np.abs(Y_val - pred["va"]))), mse["ex"],
               mse["tr"] / var_tr if var_tr > 0 else float("inf"),
               mse["va"] / var_va if var_va > 0 else float("inf")]
        if all(np.isfinite(vec)):
            table.append({**c, "vec": vec})
    if not table:
        return None, "no candidate form evaluates finitely on the data"
    # HFF, normalised by hand: error columns min-max over the candidates, size
    # as nodes / the input's nodes. normalize=False, as in the join.
    F = _nb_minmax_columns(np.array([t["vec"] for t in table], dtype=np.float64))
    size = np.array([t["nodes"] for t in table], dtype=np.float64) / float(cands[0]["nodes"])
    fit = hff.calculate_fitness_hf1_enhanced(np.hstack([F, size.reshape(-1, 1)]), normalize=False,
                                             north_pole_method=settings.north_pole_method)
    for t, f in zip(table, fit):
        t["hff"] = float(f)
    table.sort(key=lambda t: (t["hff"], t["nodes"], t["infix"]))
    return table[0], table


_pt = time.perf_counter()
model_no_sympy = model_lint_then_sympy = None
_final_note = ""
if _won_via_rule:
    _final_note = "won by a static rule: the model is not a gene, nothing to simplify"
else:
    _winner, _forms = _nb_final_forms(best_ind, _best_wrapper_name)
    if _winner is None:
        _final_note = _forms
    else:
        model_no_sympy = _winner["infix"]
        print("Candidate forms of the reported model (executed on data, ranked by HFF):")
        for _t in _forms:
            print(f"  hff={_t['hff']:.6g} nodes={_t['nodes']:>3} {_t['level']:<8} "
                  f"val_mse={_t['vec'][1]:.4g}  {_t['infix'][:110]}")
_perf_add("final form: fuller candidates + execute + HFF (no sympy)", _pt)
if model_no_sympy is not None:
    _pt = time.perf_counter()
    from hff.sr.bounded_simplify import capped_simplify as _capped
    _local = {v: sp.Symbol(v, real=True) for v in finalTerminals}
    model_lint_then_sympy = str(_capped(sp.sympify(model_no_sympy, locals=_local)))
    _perf_add("final form: capped sympy on the fuller form", _pt)
if _final_note:
    print(f"No fuller form for the reported model: {_final_note}")
print(f"model_no_sympy        : {model_no_sympy}")
print(f"model_lint_then_sympy : {model_lint_then_sympy}")
print(f"model_sympy (reported): {snapped}")

# %% [markdown]
# ## 4.2 Equation-recovery scoring (structural + numerical)
''')

swap('''print(json.dumps(experiment, sort_keys=False, indent=4, default=str))''',
     '''experiment["model_no_sympy"] = model_no_sympy
experiment["model_lint_then_sympy"] = model_lint_then_sympy
experiment["model_sympy"] = str(snapped)
experiment["final_form_note"] = _final_note

# ---- performance: where the run's time went ------------------------------
_run_total = sum(v for k, v in _NB_PERF.items() if not k.startswith("  of "))
_NB_PERF["  of evaluate: device dispatch + host scoring"] = _NB_GPU_STATS.get("seconds", 0.0)
_NB_PERF["  of evaluate: simplifier (fuller)"] = _NB_GPU_STATS.get("expand_seconds", 0.0)
print("\\n[perf] stage                                                        seconds   share")
for _k, _v in sorted(_NB_PERF.items(), key=lambda kv: (kv[0].startswith("  of "), -kv[1])):
    print(f"[perf] {_k:<62}{_v:>9.2f}  {100.0 * _v / _run_total if _run_total else 0.0:>5.1f}%")
_gens = max(1, int(gen) - 1)
_evald = int(_NB_GPU_STATS.get("chromosomes", 0))
print(f"[perf] generations {_gens} | individuals evaluated {_evald:,} | candidates scored "
      f"{int(_NB_GPU_STATS.get('candidates', 0)):,} | {1e3 * _run_total / _gens:.1f} ms per generation | "
      f"{_evald / _run_total if _run_total else 0.0:,.0f} individuals per second")
experiment["perf_seconds"] = {k.strip(): round(v, 4) for k, v in _NB_PERF.items()}
experiment["perf_generations"] = _gens
experiment["perf_individuals_evaluated"] = _evald
experiment["perf_candidates_scored"] = int(_NB_GPU_STATS.get("candidates", 0))

print(json.dumps(experiment, sort_keys=False, indent=4, default=str))''')

open(p, "w").write(s)
print("patched")
