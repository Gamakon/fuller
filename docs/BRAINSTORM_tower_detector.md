# BRAINSTORM — a tower detector for evolved models

## Context page

| | |
|---|---|
| **Objective** | Beat SRBench ground-truth (133 problems). The search keeps returning FAKES: train R² = test R² = 1.0000, not the law, built from long towers of nested functions. Every error objective in HFF says they are perfect, so they win tournaments. We want a function that IDENTIFIES and MEASURES a tower. |
| **Approach** | Take the labelled models we already have (13 race logs, 1,230 fits), compute structural features per model, measure which features separate real laws from fakes, then show how the useful ones are computed straight off the Karva token array in one pass. |
| **Unique feature** | The measure is *transcendental nesting depth* (`t_depth`), not size. All 133 true laws have `t_depth` ≤ 2. A long, flat, all-binary law scores 0. That is what separates it from parsimony, which Andrew's HFF deliberately works without. |
| **Status** | Brainstorm + offline evidence only. No engine code changed, nothing run on the GPU, nothing committed. Scripts are in the session scratchpad (`.../scratchpad/towers/`: `features.py`, `analyse.py`, `extra.py`, `karva_pass.py`). |
| **Why this** | Leave-one-part-out already showed fakes carry dead parts (median R² loss 0.003 vs 0.49 for laws). That costs re-scoring. A tower measure is integer arithmetic on the tokens: no data rows, no evaluation. The two are complementary. |

**The no-cheating line.** The true laws (`true_model` in the `.json.updated` files, from `metadata.yaml`) are used here ONLY to label and to check an offline detector for false positives. Nothing in this design reads a problem's law during search; the in-loop measure sees only token ids and an arity table. The one number taken from the laws collectively is a ceiling (`t_depth` ≤ 2 over all 133) — a property of the benchmark's whole family, not of any one problem. Whether even that is acceptable is Andrew's call; see §5.

---

## 0. Data, and what it limits

| | count |
|---|---|
| Races read | 13 (`ab1_base`, `ab1_cleanse`, `ab2_base_s11/s12`, `ab2_harvest_s11/s12`, `ab3_rnc5_s11`, `ab3_restart3_s11`, `big_pop_redundancy_g35_s11`, `smogd_run_s11`, `smogd_smote_2k_s11` (3 JSONs only), `hfflog_1000g_s11`, `allfixes_s12`) |
| Fits with a JSON + a log row | 1,230 |
| Parsed by sympy | 1,222 (8 rows / 6 distinct strings failed: 4 `Invalid NaN comparison`, 1 integer overflow, 1 30-s timeout — reported, not dropped silently) |
| Raw classes | law (`sol=Y`) 418 · fake (`sol=n`, r2_test ≥ 0.999) 224 · not-converged 580 |
| **Deduplicated on (problem, model string)** — used for every table below | **law 144 (61 problems) · fake 187 (49 problems) · not-converged 476 (86 problems)** |
| True laws | 133 of 133 parsed |
| Problems with both a law and a fake | 16. Problems with a fake and never solved in these races: 33 |

Limits, stated once:

- The strings are **sympy-tidied infix, not Karva**. Sympy flattens Add/Mul to n-ary, prints division as `x**-1`, and cancels some nestings. On the 10 fits where both strings exist (`allfixes_s12/side_by_side.tsv`), fuller's own string vs the tidied one: `t_depth` identical in 8 of 10, one lower by 1 in the other 2; nodes differ by ≤ 3. Both are post-lint strings, so neither is the literal gene tree. Tower measures here are a lower bound on the Karva's.
- Protected ops print as `Piecewise((v, v < 1.797e308), (0, True))`. I kept `v` and counted the Piecewise separately.
- The found-law class is biased to the easy problems (median 9 nodes; true laws overall median 13, max 37). So every "found law" false-positive figure is paired with the same figure on all 133 true laws. The second is the one that matters.
- **The tidy removes `Abs` where it can prove positivity.** `hff/notebooks/_rust_race.py` (l. 187) applies `_with_positive_columns` before the tidy, so `Abs(x)` and `sqrt(x**2)` collapse when the columns are all-positive. Found laws are the models where that succeeds; fakes are not. So `n_abs` = 0 on all 144 found laws is partly a tidy artefact, and a found law's `t_depth` may be 1 lower here than on its Karva. On the Karva every model under the SqrtAbs or LogAbs wrapper contains an `Abs`. The `n_abs` ceiling of 0, and rule (b″) in §2, should not be expected to survive the move to Karva.
- Label noise exists in the fake class: e.g. `feynman_I_47_23` `sqrt(x_0)*sqrt(x_1/x_2)` and `feynman_test_5` `6.283185*sqrt(x_0**3/(x_1*(x_2 + x_3)))` are `sol=n` with R² ≥ 0.999 and are structurally flat. They are counted as fakes below, which costs every detector some recall.

Feature definitions (sympy side):

| feature | definition |
|---|---|
| `nodes`, `ops`, `depth` | sympy tree nodes; internal nodes; n-ary depth |
| **`t_depth`** | T = {exp, log, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, Abs, any `Pow` with a non-integer exponent (sqrt, `x**(1/4)`, `x**y`)}. Integer powers are NOT in T. `t_depth` = max over root-to-leaf paths of the number of T nodes on the path. Immune to Add/Mul flattening. |
| `chain` | longest run of consecutive one-child nodes (1-arg function or numeric power). `a*x`, `x+c`, and `x**-1` pass through without adding. The literal "tower height". |
| `sat_depth` | same as `t_depth` but only {tanh, sin, cos, atan} |
| `n_T`, `T_frac` | number of T nodes; / ops |
| `tower_mass` | Σ over T nodes of (number of T ancestors). 0 when no T node sits inside another. |
| `tt_pairs` | T nodes whose direct child (through pass-throughs) is a T node |
| `exp_exp`, `tanh_tanh` | counts of those direct patterns |
| `unary_frac` | one-child nodes / ops |
| `n_vars`, `var_frac` | distinct variables; / variables in the true law |
| `n_float`, `n_const`, `log10_max_const`, `cancel_pair` | floats; floats + integers > 3; log10 of the largest \|constant\|; 1 if two constants > 100 agree within 1 % |
| `n_abs`, `n_piecewise`, `repeat_subtrees` | counts |

---

## 1. What a tower is, measured

Quartiles are 25 / 50 / 75 %. AUC = P(feature is larger on the second class). n: law 144, fake 187, not-converged 476, true laws 133.

| feature | law | fake | not-converged | true laws | AUC fake vs law | AUC notconv vs law | AUC fake vs true laws |
|---|---|---|---|---|---|---|---|
| nodes | 7 / 9 / 11 | 32 / 43 / 54 | 37 / 48 / 59 | 8 / 13 / 18 | 0.984 | 0.999 | 0.937 |
| ops | 2 / 3 / 4 | 15 / 21 / 29 | 18 / 24 / 31 | 3 / 5 / 7 | 0.986 | 1.000 | 0.944 |
| depth | 3 / 3 / 3 | 9 / 13 / 16 | 10 / 13 / 17 | 3 / 5 / 6 | 0.986 | 0.999 | 0.954 |
| **t_depth** | 0 / 0 / 0 | 3 / 4 / 7 | 3 / 5 / 7 | 0 / 0 / 1 | 0.973 | 0.992 | 0.953 |
| n_T | 0 / 0 / 0 | 4 / 7 / 12 | 5.75 / 9 / 13 | 0 / 0 / 1 | 0.974 | 0.993 | 0.956 |
| n_const | 0 / 1 / 1 | 3 / 5 / 6 | 3 / 4 / 6 | 0 / 0 / 1 | 0.967 | 0.996 | 0.972 |
| n_float | 0 / 1 / 1 | 3 / 4 / 5 | 2 / 4 / 5 | 0 / 0 / 1 | 0.966 | 0.995 | 0.973 |
| tower_mass | 0 / 0 / 0 | 3 / 10 / 30.5 | 5 / 15 / 33 | 0 / 0 / 0 | 0.947 | 0.975 | 0.943 |
| chain | 0 / 1 / 1 | 2 / 3 / 4 | 2 / 3 / 4 | 0 / 1 / 1 | 0.945 | 0.972 | 0.925 |
| T_frac | 0 / 0 / 0 | 0.25 / 0.33 / 0.42 | 0.28 / 0.36 / 0.43 | 0 / 0 / 0.18 | 0.922 | 0.943 | 0.873 |
| tt_pairs | 0 / 0 / 0 | 1 / 2 / 5 | 1 / 3 / 5 | 0 / 0 / 0 | 0.914 | 0.941 | 0.914 |
| n_abs | 0 / 0 / 0 | 1 / 1 / 2 | 1 / 2 / 3 | 0 / 0 / 0 | 0.896 | 0.909 | 0.896 |
| log10_max_const | 0 / 0.30 / 0.30 | 0.62 / 1.36 / 1.93 | 0.48 / 1.14 / 1.75 | 0 / 0.30 / 0.30 | 0.895 | 0.893 | 0.893 |
| sat_depth | 0 / 0 / 0 | 1 / 2 / 2 | 1 / 2 / 3 | 0 / 0 / 0 | 0.885 | 0.922 | 0.851 |
| unary_frac | 0 / 0.20 / 0.50 | 0.35 / 0.42 / 0.50 | 0.37 / 0.45 / 0.52 | 0 / 0.29 / 0.50 | 0.721 | 0.753 | 0.666 |
| n_piecewise | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 0 | 0.666 | 0.716 | 0.666 |

**Do not separate** (AUC 0.48–0.56): `n_vars`, `var_frac` (fakes use every variable, median 1.00 in all classes), `exp_exp` (0.524), `tanh_tanh` (0.537), `cancel_pair` (0.511 — 4 fakes, 1 not-converged, 0 laws), `repeat_subtrees` (0.564). The huge-constant cancellation is real but rare in this corpus: \|const\| ≥ 1000 appears in 24 of 187 fakes, 28 of 476 not-converged, 0 laws, 0 true laws.

**Fakes and not-converged models have the same shape.** Every feature has near-identical quartiles for the two. A tower score separates "law-shaped" from "not law-shaped"; it does not separate accurate-wrong from inaccurate-wrong. That is fine for the purpose: the error objectives already handle the second.

**AUC alone does not pick the winner — size wins on AUC.** `nodes`/`ops`/`depth` have the highest AUCs. The difference shows up at the ceiling: set each threshold just above the largest value seen on ANY of the 133 true laws or 144 found laws (so zero laws flagged by construction), and count the fakes caught:

| feature | ceiling over all laws | fakes above it | not-converged above it |
|---|---|---|---|
| tt_pairs | 0 | 155 / 187 = 82.9 % | 88.2 % |
| n_T | 3 | 148 / 187 = 79.1 % | 88.4 % |
| n_abs | 0 | 148 / 187 = 79.1 % | 81.7 % |
| depth | 8 | 147 / 187 = 78.6 % | 88.0 % |
| n_float | 2 | 145 / 187 = 77.5 % | 72.5 % |
| **t_depth** | **2** | **143 / 187 = 76.5 %** | 83.4 % |
| tower_mass | 2 | 143 / 187 = 76.5 % | 84.9 % |
| ops | 15 | 139 / 187 = 74.3 % | 85.5 % |
| nodes | 37 | 121 / 187 = 64.7 % | 73.9 % |
| chain | 2 | 105 / 187 = 56.1 % | 61.6 % |
| sat_depth | 1 | 104 / 187 = 55.6 % | 52.7 % |
| log10_max_const | 1.40 | 92 / 187 = 49.2 % | 39.5 % |

The true-law ceilings, by name — the laws any threshold must clear:

| measure | distribution over the 133 true laws | the laws at the top |
|---|---|---|
| t_depth | 0: 71 · 1: 58 · 2: 4 · ≥3: 0 | `feynman_I_26_2` arcsin(n·sin θ₂); `feynman_I_29_16` sqrt(x1² − 2x1x2cos(θ1−θ2) + x2²); `feynman_test_10` arccos((cos θ₂ − v/c)/(1 − v cos θ₂/c)); `feynman_test_13` …/sqrt(d² − 2dr cos α + r²) |
| chain | 0: 39 · 1: 85 · 2: 9 · ≥3: 0 | all nine are `sin(..)**2`, `sin(..)**4`, `cos(..)**2` or `exp(-θ**2/2)`: `III_8_54`, `III_9_52`, `I_30_3`, `I_50_26`, `I_6_2a`, `test_1`, `test_11`, `test_20`, `strogatz_shearflow2` |
| tower_mass | 0: 129 · 1: 3 · 2: 1 | `test_10` = 2 |
| tt_pairs | 0: 133 | no true law has a T function whose direct argument is a T function (arcsin(n·sin θ) has a Mul between) |
| nodes | median 13, max 37 | `test_20` 37, `test_12` 33, `test_2` 31, `III_9_52` 28, `I_9_18` 28 |

`m*c^2/sqrt(1 - v^2/c^2)`-type laws (`I_10_7`, `II_13_23`, `I_15_3t`, …) have `t_depth` 1, `chain` 1.

Variant: not counting `Abs` as T gives AUC 0.963 and 123 / 187 fakes at ≥ 3 (vs 143 with Abs). The engine's wrappers and protected ops put `Abs` into the string, so `Abs` is doing real work in the measure; true laws have none.

---

## 2. Candidate tower scores

All numbers: deduplicated law-vs-fake, n = 331. Learned models: `GroupKFold(5)` by problem (no problem on both sides), pooled out-of-fold predictions. "True laws flagged" uses the model fitted on all 331 and applied to the 133 true laws, which were never training rows.

| | score | AUC (held-out where fitted) | threshold | fakes caught | found laws flagged (of 144) | **true laws flagged (of 133)** | cost |
|---|---|---|---|---|---|---|---|
| (a) | `t_depth` | 0.973 (nothing fitted) | ≥ 3 | 143 / 187 = 76.5 % | 0 | **0** | one pass, 1 int per node |
| | | | ≥ 2 | 89.3 % | 0 | **4**: `I_26_2`, `I_29_16`, `test_10`, `test_13` | |
| | | | ≥ 4 | 59.4 % | 0 | 0 | |
| (a′) | `chain` | 0.945 | ≥ 3 | 56.1 % | 0 | 0 | one pass |
| | | | ≥ 2 | 86.6 % | 5 | **9** (the `sin²`, `exp(-θ²/2)` laws above) | |
| (a″) | `tower_mass` | 0.947 | ≥ 3 | 76.5 % | 0 | 0 | one pass |
| (b) | hand sum `max(0,t_depth−2) + 0.5·tt_pairs + exp_exp + tanh_tanh + cancel_pair + 0.25·n_abs` | 0.957 | ≥ 0.5 | 86.1 % | 0 | **0** | one pass |
| (b′) | rule `t_depth≥3 OR chain≥3 OR max\|const\|≥1000` | — | — | 150 / 187 = 80.2 % | 0 | **0** | one pass |
| (b″) | (b′) `OR n_abs≥1` | — | — | 168 / 187 = 89.8 % | 0 | **0** | one pass; leans on Abs being an engine artefact |
| (c) | logistic, all 18 features | 0.990 | p ≥ 0.5 | 94.1 % | 0 | **19**: `III_4_33`, `III_9_52`, `II_35_18`, `II_6_15a`, `I_41_16`, `I_9_18`, `test_1`, `test_10`, `test_11`, `test_12`, `test_13`, `test_16`, `test_19`, `test_2`, `test_20`, `test_4`, `test_5`, `test_6`, `test_7` | 18 features + dot product |
| | | | p ≥ 0.8 | 91.4 % | 0 | **1**: `III_9_52` | |
| | | | p ≥ 0.95 | 87.7 % | 0 | 0 | |
| (c′) | logistic, tower-only features (no nodes/depth/n_T/n_vars/n_const) | 0.976 | p ≥ 0.5 | 89.8 % | 9 (`II_15_4`, `II_15_5`, `I_18_12`, `I_6_2a` ×5, `strogatz_shearflow1`) | **8**: `III_8_54`, `III_9_52`, `I_26_2`, `I_50_26`, `test_10`, `test_11`, `test_5`, `strogatz_shearflow2` | |
| | | | p ≥ 0.8 | 86.6 % | 0 | 0 | |
| (c″) | logistic on `nodes` only = **plain parsimony** | 0.983 | p ≥ 0.5 | 89.8 % | 0 | **31** (Appendix) | |
| | | | p ≥ 0.8 | 80.2 % | 0 | **12**: `III_9_52`, `II_35_18`, `II_36_38`, `I_9_18`, `test_1`, `test_12`, `test_16`, `test_17`, `test_19`, `test_2`, `test_20`, `test_4` | |
| (d) | decision tree, depth 3, all features | 0.982 | 0.5 | 94.1 % | 11 | **48** (Appendix) | trivial |
| (d′) | decision tree, tower-only | 0.971 | 0.5 | 90.9 % | 7 | **38** (Appendix) | trivial |
| (e) | bag of level-order n-grams over the **sympy** tree (proxy; NOT engine Karva), logistic | 1-grams 0.991 · 2-grams 0.989 · 1–3-grams 0.993 | 0.5 | 93.6 % / 92.0 % / 93.6 % | 4 / 3 / 4 | **44 / 34 / 44** (Appendix, first 30 of each) | vocabulary lookup per token |

Confusion matrices (fake = positive), out-of-fold: (c) at 0.5: TP 176, FN 11, FP 0, TN 144. (c) at 0.8: TP 171, FN 16, FP 0, TN 144. (a) at ≥ 3: TP 143, FN 44, FP 0, TN 144.

Learned weights, standardized: (c) n_const 2.00, depth 1.85, nodes 1.27, n_abs 1.12, T_frac 1.04, t_depth 0.82, n_T 0.73, log10_max_const 0.68, chain 0.61, unary_frac −0.45, n_vars 0.45, tt_pairs 0.30, rest < 0.1; intercept 4.68. (c′) t_depth 2.68, chain 1.86, log10_max_const 1.52, tt_pairs 1.06, T_frac 0.82, unary_frac −0.58, tower_mass 0.28, tanh_tanh 0.16, exp_exp 0.15, rest ≈ 0; intercept 3.38.

The trees the data produced: (d) splits on `depth ≤ 5.5` then `nodes` — it learned size. (d′) splits on `tower_mass ≤ 0.5`, then `T_frac ≤ 0.06`, then `unary_frac ≤ 0.45`. Both flag 38–48 true laws: with found laws at median 9 nodes, a tree puts its cut between "tiny" and "everything else", and the harder true laws fall on the wrong side.

Readings:

1. **Learned models score the highest held-out AUC and the worst on true laws.** Their training positives are the easy laws; they learn "bigger than an easy law". Given free choice the logistic puts its weight on `n_const`, `depth`, `nodes` — parsimony. This is the measured form of the risk Andrew named.
2. **The hand measures transfer**: `t_depth ≥ 3`, the hand sum, and rule (b′) flag 0 of 144 found laws and 0 of 133 true laws. Caveat: the threshold 3 was chosen after seeing that the true-law maximum is 2, so "0 of 133" is in-sample for that one integer. The 144 found laws were not used to choose it.
3. **(e) sequence models: the data volume does not support it, and the data is the wrong kind.** 331 rows, 49–61 problems, and the token stream is a level-order walk of a sympy tree, not the engine's Karva. The 1-gram model (a bag of 14 token counts) already matches the 3-gram model, so the n-grams add nothing measurable here, and it flags 44 true laws. A Karva-level learner needs the logging in §5 first.
4. What `t_depth ≥ 3` misses: 44 fakes. 11 problems have no fake flagged at all: `I_47_23`, `test_1`, `I_38_12`, `test_5`, `strogatz_lv2`, `strogatz_shearflow1`, `strogatz_glider2`, `II_2_42`, `II_37_1`, `II_3_24`, `I_13_12`. These are flat models with extra terms or near-miss constants — a different failure, and the one leave-one-part-out is built for. 22 of the 49 faked problems have every fake flagged.

---

## 3. Computing it on the Karva directly

In a K-expression every child has a larger index than its parent, and children are handed out in order. So the running `child` pointer that `decode_gene` (`src/evolve/engine.rs:85`) and `cleanse_gene` (`src/evolve/vary.wgsl:240`, which fills `cl_child[i]`) already keep IS the parent map. One forward scan, no tree, no recursion, no stack:

```text
inputs : tok[0..ht)            the gene's tokens (head + tail)
         arity[id]             existing table
         is_t[id]              NEW table, 1 bit per symbol: Abs Sqrt Log Exp Sin Cos Tan Tanh
                               Pow(binary) and every Protected* twin = 1;
                               Add Sub Mul Div Neg Pow2 Pow3 Inv, inputs, constants, "?" = 0
state  : parent[64], depth[64], tdepth[64], chain[64]     (u8 is enough; 64-node limit exists already)

need = 1; n = 0; child = 1
max_depth = max_t = max_chain = mass = 0; top_of_tallest = 0
while need > 0 and n < ht:
    t = tok[n]; a = arity[t]
    if n == 0:
        depth[0] = 1; tdepth[0] = is_t[t]; chain[0] = (a == 1)
    else:
        p = parent[n]
        depth[n]  = depth[p] + 1
        tdepth[n] = tdepth[p] + is_t[t]
        chain[n]  = (arity[tok[p]] == 1 ? chain[p] : 0) + (a == 1)
    if is_t[t]: mass += tdepth[n] - 1
    if tdepth[n] > max_t: max_t = tdepth[n]; deepest = n      # for the cleanse target
    max_depth = max(max_depth, depth[n]); max_chain = max(max_chain, chain[n])
    for k in 0..a: parent[child + k] = n
    child += a; need += a - 1; n += 1
```

Checked: a Python reference of exactly this loop against a recursive tree walk on 20,000 random K-expressions (head 48, 17 symbols): 0 mismatches on (nodes, depth, t_depth, chain, tower_mass). `exp(tanh(log(|cos(sqrt(x0))|)))` → t_depth 6, chain 6, tower_mass 15. (`karva_pass.py`. This is a check of the algorithm, not the Rust/WGSL parity test, which does not exist yet.)

Notes for the Rust + WGSL version:

- All integer. Rust CPU reference and WGSL kernel are bit-exact by construction (no floats until the final scale), so the parity test is an equality assert over a random population, same style as `cleanse_gene`'s.
- Cost per gene: ≤ 64 iterations, four `u8` arrays. `cleanse_gene` already allocates five `array<u32,128>` privates per invocation; this is smaller. It can ride inside the existing scoring dispatch or the GPU linter's walk. Only the expressed region (`n` nodes) is scanned — the non-coding tail is ignored, which is correct: it is not in the model.
- `Pow2`, `Pow3`, `Inv`, `Neg` are arity-1 in the engine (`src/gpu_eval.rs:69`). They count toward `chain` but not `t_depth`. That is one reason to prefer `t_depth`: `sin(x)**2` is chain 2 in nine true laws but `t_depth` 1.
- Binary `Pow` is marked `is_t` = 1 (conservative: x^y with a non-integer y is T in the sympy measure). If the exponent child is a small-integer constant this over-counts by 1; whether that matters needs Karva data (§5).

**Multi-gene.** Model = a·WRAPPER(LINKER(g1, g2, g3)) + b. The linkers (avgval, mulval, addval) are binary, so they break chains and add no T: the body's `t_depth` = **max over the three genes**, `tower_mass` and `n_T` = sum over genes. The wrapper sits on top of everything: Identity +0, SqrtAbs +2 (Sqrt, Abs), LogAbs +2 (Log, Abs), added to every gene's count, plus (wrapper T count) × (body n_T) on `tower_mass`. Suggested default: **score the genes only (max over genes), threshold 3, wrapper left out.** Two reasons. (1) The wrapper is not evolved: `engine.rs:575–591` scores every (linker, wrapper) pair for each chromosome and keeps the best, so evolution cannot grow a tower out of it. (2) With the wrapper included (+2), the four true laws at `t_depth` 2 (`I_26_2`, `I_29_16`, `test_10`, `test_13`) would be flagged the moment they were found under a non-Identity wrapper. The same margin argues for 3 and not 2 as the floor: the true-law maximum is 2, and the tidy may have hidden one level on the found laws (§0). Genes-only at 3 is more permissive than the sympy-side numbers here (which include the wrapper), so the 76.5 % is an upper estimate of what the Karva-side measure catches; the wrapper-included variant is the one to test once the Karva is logged (§5).

---

## 4. How it would be used — options, not a decision

| use | how | for | against |
|---|---|---|---|
| Extra HFF objective | `o = clamp((t_depth − 2) / 6, 0, 1)` pushed beside `s[9]` redundancy (`engine.rs:584`), `maxes = 1.0`, not log-scaled. Zero for every shape a true law has. | Acts all generation long; towers stop winning tournaments on error alone. Same plumbing as redundancy. | Changes the angle for every individual with `t_depth` ≥ 3, including stepping stones. 335 of 476 not-converged models (70.4 %) sit at `t_depth` ≥ 4 — towers may be how the search currently climbs. Could slow the pump's intake island. |
| Tournament tie-break | Only when two fitnesses are within ε, prefer lower `t_depth`. | Cannot hurt a better-fitting model. Smallest change. | Fakes at R² = 1.0000 vs a law at R² = 1.0000 is exactly a tie — but only if the law is in the population. It does not create the law. |
| Gate on harvest / final pick | Harvest-and-regrow and the final parsimony pick take the best individual with `t_depth` ≤ 2; fall back to the unrestricted best if none fits within tolerance. | Zero effect on search dynamics. Directly answers "the returned model is a tower". 0 of 133 true laws excluded. | Only helps if a flat, accurate individual exists somewhere in the population at the end. Unknown how often (§5, item 2). |
| Target for the cleansing mutation | `cleanse_gene` today picks a function node uniformly (`nth = below(row, 1, STREAM_CLEANSE, n_fn)`). Instead pick the top of the tallest tower (`deepest`'s ancestor where `tdepth` first reaches 1 or 3) and promote its child. | Turns the measure into variation, not pressure: offers the flat neighbour and lets HFF decide. No objective changes. `ab1_cleanse` vs `ab1_base` were both 46/133, so untargeted cleanse was neutral; targeted is untested. | Needs the scan inside `vary.wgsl` before the draw; changes the RNG stream use, so determinism tests need re-baselining. |
| Filter on the initial population | Reject / redraw random individuals with `t_depth` ≥ 3. | Cheap. | Weakest lever: towers are grown by variation, not drawn. The function-to-terminal ratio of the head already sets this. |
| Function-set weighting (adjacent idea) | Lower the draw probability of a T symbol when the parent slot is already under ≥ 2 T symbols. | Prevents rather than punishes. | Needs position-aware mutation on the GPU; bigger change. |

**Tower score vs parsimony — does it actually differ? Measured.**

| | Spearman with node count, all 807 models | within fakes | over the 133 true laws |
|---|---|---|---|
| depth | 0.880 | 0.842 | **0.824** |
| tower_mass | 0.850 | 0.803 | 0.089 |
| t_depth | 0.820 | 0.760 | **0.367** |
| chain | 0.687 | 0.501 | 0.519 |
| T_frac | 0.541 | 0.260 | 0.187 |
| unary_frac | 0.371 | 0.268 | 0.364 |

On evolved models `t_depth` and node count move together (0.82): evolved models that are big are also tall. On true laws they nearly decouple (0.37; `tower_mass` 0.09), while plain `depth` stays at 0.82. So the scores agree about the models the engine makes today and disagree about the laws it is supposed to find — which is the wanted behaviour. In cases:

- **Big but flat, not punished by `t_depth`, punished by size.** True laws with ≥ 20 nodes and `t_depth` ≤ 1: `test_20` (37 nodes, t_depth 1), `test_12` (33, 0), `test_2` (31, 1), `III_9_52` (28, 1), `I_9_18` (28, 0), `test_17` (27, 0), `test_19` (27, 0), `II_35_18` (26, 1), `II_36_38` (26, 0), `test_1`, `test_16`, `test_4` (26, 1). A node-count logistic flags 12 of these at p ≥ 0.8 and 31 true laws at p ≥ 0.5; `t_depth ≥ 3` flags none. Among evolved models, 14 have ≥ 25 nodes with `t_depth` ≤ 1 (13 not-converged, 1 fake).
- **Small but tall, punished by `t_depth`, missed by size.** `feynman_test_12` fake, 18 nodes: `0.266*x_0*x_2*x_3*Abs(log(43*tanh(x_3)**2))/x_1**2 − 0.012`. `feynman_III_15_12` not-converged, 19 nodes: `5.63*sqrt(x_0*(x_0 + 0.528))*exp(-exp(cos(x_1*x_2))) − 0.82`. Only 2 models under 20 nodes, but at the law-safe size ceiling (nodes ≥ 38) the gap is wider: of 187 fakes, `t_depth ≥ 3` alone catches 31 that size misses, size alone catches 9, both 112, neither 35.

What a tower objective is not: it puts no pressure on a model with `t_depth` ≤ 2, however long. Inside that region HFF runs exactly as now, with no parsimony.

---

## 5. Unknown, and what to measure next (in priority order)

1. **Log the Karva.** Nothing here was measured on a K-expression. `examples/evolve_fit.rs` prints `MODEL_INFIX` (l. 221) and `RAW_MATH` (l. 222); `hff/notebooks/_rust_race.py` (l. 175–181) keeps only `MODEL_INFIX` and writes `symbolic_model`. Needed per fit: `RAW_MATH`; the winning chromosome's three K-expressions as token ids + names (expressed region and full head+tail), the Dc domain and RNC values, the chosen linker and wrapper, a and b; and the four per-gene integers from §3. Better still, every harvest's champion, not only the last — that multiplies the training rows. One `KARVA\t…` stdout line per gene plus one more key in the race's `startswith` tuple.
2. **Cheapest experiment for "does it turn fakes into laws": the gate, offline-first.** At the end of a fit, dump the final population's top-k by HFF with their `t_depth`. Count, over the 49 faked problems, how often an individual with `t_depth` ≤ 2 and R² ≥ 0.999 is already present. If often → the gate on the final pick is enough and costs nothing in search. If almost never → the pressure has to act during search (objective or targeted cleanse). That is one race with extra logging, no engine behaviour change.
3. **Then the A/B**, development seeds only (the same ones the existing ab races used): base vs `t_depth` objective vs targeted cleanse, 133 problems, same budget. Report solved counts AND the count of `sol=n, R² ≥ 0.999` — the fake count should fall even if solved does not rise. Watch generations-to-early-stop on the 46 already-solved problems for any slowdown.
4. **Is the ceiling legitimate?** `t_depth` ≤ 2 came from looking at all 133 laws together. It is not problem-specific, but it is benchmark-derived. A defensible independent source: the physics prior in `src/physics.rs`, or simply fixing 2 as a prior ("a law nests at most two transcendental functions") and declaring it. Andrew's call.
5. **Wrapper convention** (§3): genes-only vs wrapper-included. Needs item 1.
6. **Is `Abs`/protected-op counting right?** `n_abs ≥ 1` alone catches 79 % of fakes with 0 laws flagged, but `Abs` comes from the wrappers and protected ops, so this is partly the engine describing itself, and the zero on found laws is partly the positivity tidy (§0). With Karva logs, measure T-depth over raw vs `Protected*` symbols separately.
7. **The 44 flat fakes / 11 problems** in §2 note 4 are out of reach for any tower score. Cross them with leave-one-part-out: if the dead-part measure catches most of them, the two together cover the fake class. Not computed here (needs data rows and evaluation).
8. **Learned Karva-level detector** (Andrew's original idea): revisit after item 1 has produced a few thousand labelled chromosomes across ≥ 100 problems. With 331 rows on 49–61 problems, the measured outcome is that a learner rediscovers parsimony. Any future learner should be trained with the 133 true-law *shapes* held out as the false-positive test, as here, and should be compared against `t_depth ≥ 3` as the baseline to beat: 76.5 % of fakes, 0 / 144, 0 / 133.

### Not done, and why

- No Karva-level features or sequence model on real token streams: the races did not save them.
- No Rust or WGSL written; no bit-exact parity test. Out of scope for a brainstorm; the algorithm was checked in Python only.
- No measurement of whether a tower objective changes solved counts. That requires evolution runs, which this task excluded.
- 8 of 1,230 fits not featured (sympy parse failures listed in §0).

---

## Appendix — every true law flagged, per detector (names without the `feynman_` prefix)

| detector | n | true laws flagged |
|---|---|---|
| `t_depth ≥ 3`, `chain ≥ 3`, `tower_mass ≥ 3`, hand sum ≥ 0.5, rules (b′) (b″), logistic (c) p ≥ 0.95, (c′) p ≥ 0.8 | 0 | — |
| `t_depth ≥ 2` | 4 | I_26_2, I_29_16, test_10, test_13 |
| `chain ≥ 2` | 9 | III_8_54, III_9_52, I_30_3, I_50_26, I_6_2a, test_1, test_11, test_20, strogatz_shearflow2 |
| (c) logistic all, p ≥ 0.5 | 19 | III_4_33, III_9_52, II_35_18, II_6_15a, I_41_16, I_9_18, test_1, test_10, test_11, test_12, test_13, test_16, test_19, test_2, test_20, test_4, test_5, test_6, test_7 |
| (c) p ≥ 0.8 | 1 | III_9_52 |
| (c′) logistic tower-only, p ≥ 0.5 | 8 | III_8_54, III_9_52, I_26_2, I_50_26, test_10, test_11, test_5, strogatz_shearflow2 |
| (c″) nodes only, p ≥ 0.5 | 31 | III_4_33, III_9_52, II_21_32, II_35_18, II_36_38, II_6_15a, I_15_3t, I_15_3x, I_29_16, I_32_17, I_34_14, I_41_16, I_9_18, test_1, test_10, test_11, test_12, test_13, test_14, test_15, test_16, test_17, test_18, test_19, test_2, test_20, test_3, test_4, test_6, test_8, test_9 |
| (c″) p ≥ 0.8 | 12 | III_9_52, II_35_18, II_36_38, I_9_18, test_1, test_12, test_16, test_17, test_19, test_2, test_20, test_4 |
| (c″) p ≥ 0.95 | 1 | test_20 |
| (d) tree, all features | 48 | III_14_14, III_15_12, III_4_32, III_4_33, III_9_52, II_11_28, II_11_3, II_13_23, II_13_34, II_21_32, II_24_17, II_35_18, I_10_7, I_15_10, I_15_3t, I_15_3x, I_16_6, I_29_16, I_32_17, I_34_1, I_34_14, I_41_16, I_48_2, I_50_26, I_6_2b, I_8_14, I_9_18, test_10, test_12, test_13, test_14, test_15, test_16, test_17, test_19, test_2, test_20, test_3, test_4, test_7, test_8, strogatz_bacres1, strogatz_bacres2, strogatz_barmag1, strogatz_barmag2, strogatz_predprey1, strogatz_predprey2, strogatz_shearflow2 |
| (d′) tree, tower-only | 38 | III_14_14, III_15_12, III_17_37, III_4_32, III_4_33, III_9_52, II_35_18, II_35_21, I_12_11, I_15_3t, I_15_3x, I_26_2, I_29_16, I_30_5, I_34_14, I_37_4, I_40_1, I_41_16, I_44_4, I_47_23, I_50_26, I_6_2b, I_8_14, test_10, test_13, test_14, test_15, test_16, test_2, test_3, test_4, test_5, test_7, test_8, strogatz_barmag1, strogatz_barmag2, strogatz_glider1, strogatz_glider2 |
| (e) 1-grams | 44 | first 30 printed: III_4_33, III_9_52, II_11_28, II_13_34, II_21_32, II_24_17, II_35_18, II_36_38, II_6_15a, I_15_10, I_15_3t, I_15_3x, I_29_16, I_32_17, I_34_14, I_41_16, I_48_2, I_6_2b, I_8_14, I_9_18, test_1, test_10, test_11, test_12, test_13, test_14, test_15, test_16, test_17, test_18 |
| (e) 2-grams | 34 | first 30 printed: III_15_12, III_9_52, II_11_27, II_11_28, II_21_32, II_24_17, I_15_3t, I_15_3x, I_18_4, I_34_14, I_41_16, I_48_2, I_6_2b, I_9_18, test_1, test_10, test_11, test_12, test_15, test_16, test_17, test_18, test_19, test_2, test_20, test_3, test_4, test_5, test_6, test_7 |
| (e) 1–3-grams | 44 | first 30 printed: III_15_12, III_9_52, II_11_28, II_13_34, II_21_32, II_24_17, II_6_15a, I_15_10, I_15_3t, I_15_3x, I_18_4, I_29_16, I_30_3, I_32_17, I_34_14, I_37_4, I_41_16, I_48_2, I_6_2b, I_9_18, test_1, test_10, test_11, test_12, test_13, test_14, test_15, test_16, test_17, test_18 |

The (e) lists are cut at 30 by the script's print; the counts are complete.
