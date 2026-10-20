# PLAN: the TSR kingdom on multi-typed GEP, with stream order and transcendental depth carried in the types

Date: 2026-10-20. Author: fuller session (plan agent). Status: design, with a host prototype of the type tables and their check; nothing is built in Rust.

## Context

**Objective.** Symbolic regression (SR) on the multi-typed GEP core (`nucleotable/docs/MultiTypedGEP.md`, revision 4) is today one type, so it is plain classical GEP. Andrew wants the real multi-typed SR, the **TSR kingdom**: a gene whose well-typedness is the same thing as two measured limits on the 133 SRBench laws. The first limit is Horton–Strahler stream order of the whole expression (observed 2 or 3, never 4). The second is the number of transcendental functions on any path (observed at most 2). A gene that would break either limit has no spelling, so no individual is built and then thrown away.

**Approach.** Copy what the WGSL kingdom does with nesting depth (`kingdoms/safewgsl/gen_tables.py`: the duals L0..L5, C0..C5, O0..O4, where a loop row exists only if its body dual exists at the next depth). Here the dual of a node carries its stream-order profile and its transcendental depth level. Each function becomes a set of rows, one per legal combination of argument duals, and the rev-4 layout (a head and a tail per type, sized from a program size N) does the rest. A generator writes the rows; a checker proves them against the engine's own order and depth code.

**Unique feature.** TSR is the only SR in which "how far apart the functions are" is enforced by the representation rather than scored afterwards. Mutation, crossover, the beam, fold and snap cannot leave the legal region because the domain of every operator is enumerated from the layout before the draw.

**Status.** The type design below is derived and has been checked by a host prototype (Section 2.6): 4.3 million trees of up to 9 nodes and 300,000 random trees, no disagreement with the engine's rules. Sizing is computed with the rev-4 formulas, which the prototype reproduces for the one-type SR kingdom (width 47 at N = 8, 191 at N = 32, as printed in MultiTypedGEP.md). No experiment has been run for this plan. Nothing here has been built in fuller or phylu.

**Why.** The stream-order rule lost its last test (84.69 % against 85.61 %), because it was bolted on after decoding and built from tidied spellings. The depth check also sits after decoding: since the rebuild, init and mutation no longer see it, so towers are drawn and then dropped. Types fix both causes: the rule is part of the table, the table is generated, and the table is checked against the engine's rules. Whether this also helps the search is an open question; Section 7 builds every arm and measures it.

## Findings first

| finding | number | where it comes from |
|---|---|---|
| Duals needed for the order limit at root order at most 3 | 11 (5R − 4 for ceiling R) | Section 2.2 |
| Duals for order and depth together | 33, plus one root dual = 34 types | Section 2.3 |
| Function rows (with the root's twin rows) | order only 1,148; order and depth 8,264 | Section 3 |
| Gene width at program size 17 (the guarantee of today's head 34) | one type 101; depth only 558; order only 3,656; order and depth 20,980 words | Section 3 |
| "exp never on a bare variable" | **cannot be carried in types** | Section 2.4 |
| Type system against the engine's rules | 0 disagreements in 4,335,881 exhaustive and 300,000 random trees | Section 2.6 |

In plain words: the order limit and the depth limit both fit in types, at the price of a gene 36 to 208 times wider and many equal spellings of the same tree. The one rule types cannot hold is a lower bound (exp needs an argument of order 2 or more), because a bare variable fits every upper bound. Arm D keeps that rule for the comparison.

## 1. What exists today

### 1.1 The two mechanisms as built

Both are checks made after the gene is decoded. Neither is part of the gene's layout.

| | typed depth check (`EVOLVE_TYPED_DEPTH`) | stream-order rule (`EVOLVE_TYPED_ORDER`) |
|---|---|---|
| What it rejects | an expression with more than 2 transcendental operators on one root-to-leaf path: a raising operator applied to a depth-2 value (`typed_decode.rs:244-255`, `typed_decode.wgsl:174-175`) | (a) root order above the ceiling, (b) an operator whose argument order is outside its window, (c) a unary operator stacked on a unary not in the 38-pair table (`typed_decode.rs:284-321`, `typed_decode.wgsl:338-354`, host predicate `order.rs:259-303`) |
| Thresholds | depth at most 2; raisers are `DEPTH_RAISING_OPS` (`engine.rs:2137`), 10 of them in the SR alphabet: sin cos tan tanh psqrt plog pexp abs pasin pacos. Neg, pow2, pow3, pinv do not raise | root order at most 3 (`Config::typed_order`, `engine.rs:1562`); windows `RULES` (`order.rs:63-76`): pow2 pow3 inv 1..2, exp 2..infinity, sin cos tan tanh 1..2; pairs `STACKS` (`order.rs:93-134`), Neg and Abs transparent |
| Order definition | not used | `stream_order` (`order.rs:193-234`): leaf 1; node = greatest arm order, plus 1 when two or more arms tie; arms of an add or sub chain are its non-additive subtrees, arms of a mul, div or protected div chain are its non-multiplicative subtrees |
| Switched on | `EVOLVE_TYPED_DEPTH=2` (`card.rs:566-571`); the law runner sets it for every run (`scripts/law_runner.py:66`); engine passes `typed` to the decode (`engine.rs:4428`, `resident.rs:47`, `typed_device.rs:688`) | `EVOLVE_TYPED_ORDER=3` (`card.rs:573-576`); off by default and in the law runner; only `scripts/srbench_tf_run.py --typed-order` sets it (`:49`, `:167`) |
| Also applied to | graft and write-back candidates (`engine.rs:5195-5196`, `write_back.rs:452-455`) | same two sites |
| Counter | `typed_refused` (`engine.rs:2036`, `:4479`) | same counter |

Since the rebuild onto the typed core, the earlier depth budget in init and point mutation (`DepthBudget`) is gone (MultiTypedGEP.md, phylu implementation notes, "The untyped engine is gone"); `InitParams::typed_depth` (`mod.rs:470`) and `GenParams::typed_depth` (`vary.rs:272`) are still passed but no typed init or vary code reads them (a search of `typed_init.*` and `typed_vary.*` finds no use). So today a tower is drawn like any gene, expressed, and dropped at decode.

The fuller table `typed_depth_table()` (`geneframe.rs:383-430`, types `Ty::T1`, `Ty::T2` at `geneframe.rs:46-60`, `DEPTH_RAISING` at `:360`) is the depth limit written as types, with unordered argument pairs and root type F. The engine never loads it (only a test does, `engine.rs:8844`), so depth-as-types has never been run.

### 1.2 What was measured

| measurement | result | source |
|---|---|---|
| Laws by transcendental depth (128 of 133 parse) | depth 0: 78, depth 1: 47, depth 2: 3, depth 3 or more: 0; directly nested transcendental pairs: 0 | `kingdoms/tsr/README.md`, `kingdoms/symbolic-regression/README.md` |
| Laws by root stream order (sympy trees) | order 2: 90, order 3: 43, order 4: none; exp takes an order-2 argument in 10 of 10 uses | `order.rs:20-27` |
| Accurate-but-wrong models at depth 3 or more | three quarters of 1,230 labelled fits | `engine.rs:2152` doc comment |
| Pair table against the engine's solved models | rejects 0.8 % of solved models and 56 % of unsolved towers (the sympy pairs alone rejected 9 % of solved models) | `order.rs:87-91` |
| Hard rule, 6 zero-laws, 180 s | 0/60 against 0/60: shape changed, outcome did not | `logs/AB_ORDER_ON/sweep.log` last line |
| Hard rule, 83 laws, 90 s | 84.69 % (81 scored) against 85.61 % (82 scored) | `logs/AB_ORDER_EASY/summary.txt`, `logs/TF_DEV1/summary.txt` |
| Same, recounted from the scored files, excluded rows dropped | 686 against 700 solved of 810 on the 81 common laws: 16 laws better, 13 worse | `results.scored.tsv` in both folders (the memory note quotes 704 against 718; both differ by 14) |
| The six laws the rule lost, solved of 10 (rule on / rule off) | II_13_23 1 / 8, I_10_7 3 / 7, II_13_34 1 / 5, I_15_10 1 / 6, I_12_11 5 / 9, I_6_2a 3 / 8 | same files |
| SR on the typed core, 40 laws, 10 dev seeds | old engine 382 of 400 passed the strict test, new 380; median evaluations to stop 179,820 in both; median model 31 characters in both | `docs/reports/2026-10-20_sr_typed_gep.md` and `logs/SR_OLD_NEW/run1` |
| A gene the typed check rejects rides along unscored (earlier engine, 3 genes per individual) | typing accumulated dead weight instead of removing shapes | `kingdoms/tsr/symbols.md` |

### 1.3 Why the hard rule lost

Three causes, none of them the idea of an order limit:

1. The pair table was harvested from tidied spellings. It has `(Sqrt, Inv)` but no `(Inv, Sqrt)`, so the spelling `ProtectedInv(ProtectedSqrt(Sub(...)))` of 1/sqrt(1 − v²/c²) is rejected (`order.rs:93-134`, read through `may_stack`). Four of the six lost laws are Lorentz forms (II_13_23, I_10_7, II_13_34, I_15_10). I did not isolate whether this spelling alone accounts for those losses; the memory note attributes them to the harvested table and the exp minimum together.
2. The exp minimum of order 2 came from sympy folding −1/2 into an exponent. A bare `Neg` spelling of exp(−x²) has order 1 and is rejected (`order.rs:426-429`), though the code comment measures that spelling at 1 of 732 solved models, so on its own it is small.
3. The ceiling changes shape without creating the right shape: the hard laws converge on flat rational forms and never place the exp or the root (memory note, stream-order-ceiling-ab-result).

The typed design below removes causes 1 and 2 by construction: it has no pair table and no exp minimum.

## 2. The type design

### 2.1 Definitions

**Order of a node, as the engine computes it.** A leaf has order 1. A unary node has the order of its child. A sum node (add, sub) collects its arms by flattening: an arm is any subtree under the chain of sum nodes whose root is not itself a sum node. A product node (mul, pdiv) does the same with product nodes. The order of a node with arms is the greatest arm order, plus 1 if two or more arms share that greatest order.

**The profile of a node** is the minimum that a parent needs to know: its family (sum, product, or neither), its greatest arm order b, and whether that order is shared (c = 2) or single (c = 1). Its order is b + 1 if c = 2, else b. A parent of the same family merges the child's profile into its own arms; a parent of another family sees one arm of the child's order. So the profile of a parent is a function of its operator and its children's profiles only. That locality is what lets order live in types.

**Order never decreases going up.** The order of a parent is at least the order of each child (merging can only add arms or raise the greatest). So a ceiling R on the root is a ceiling on every node, and no type of order above R is ever needed.

**Depth level of a node.** 0, 1 or 2: the largest number of transcendental (raising) operators on a path from the node to a leaf. A binary operator takes the larger of its children's levels, a raising unary adds 1, other unaries keep the level. There is no level 3, so a raising operator has no row for a level-2 argument. Names follow the code: F, T1, T2.

### 2.2 The order duals (shapes)

For R = 3 the profiles that can occur are the 11 below. In general there are 5R − 4: R plain shapes, and for each of the two families 2R − 2 shapes. I call the 11 *shapes*.

| shape | family | greatest arm b | shared (c) | order | example expression |
|---|---|---|---|---|---|
| P1 | none | 1 | 1 | 1 | `x`, `sin(x)`, `x²` |
| P2 | none | 2 | 1 | 2 | `sin(x·y)`, `√(x+y)` |
| P3 | none | 3 | 1 | 3 | `√(a·b + c·d)` |
| A12 | sum | 1 | 2 | 2 | `x + y`, `x + y + z` |
| A21 | sum | 2 | 1 | 2 | `x·y + z` |
| A22 | sum | 2 | 2 | 3 | `a·b + c·d` |
| A31 | sum | 3 | 1 | 3 | `√(a·b + c·d) + a` |
| M12 | product | 1 | 2 | 2 | `x·y`, `x·y/z` |
| M21 | product | 2 | 1 | 2 | `(x + y)·z` |
| M22 | product | 2 | 2 | 3 | `(x + y)·(z + w)` |
| M31 | product | 3 | 1 | 3 | `√(a·b + c·d)·a` |

A shape of order 4 does not exist, so no row can produce one. That is the root ceiling.

**Signature of add and sub** (the first argument down, the second across; the result shape, or none when the result would have order 4). Mul and protected division are the same table with the letters A and M exchanged, in the rows and the columns (checked in the prototype).

| first \ second | P1 | P2 | A12 | A21 | M12 | M21 | P3 | A22 | A31 | M22 | M31 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1** | A12 | A21 | A12 | A21 | A21 | A21 | A31 | A22 | A31 | A31 | A31 |
| **P2** | A21 | A22 | A21 | A22 | A22 | A22 | A31 | A22 | A31 | A31 | A31 |
| **A12** | A12 | A21 | A12 | A21 | A21 | A21 | A31 | A22 | A31 | A31 | A31 |
| **A21** | A21 | A22 | A21 | A22 | A22 | A22 | A31 | A22 | A31 | A31 | A31 |
| **M12** | A21 | A22 | A21 | A22 | A22 | A22 | A31 | A22 | A31 | A31 | A31 |
| **M21** | A21 | A22 | A21 | A22 | A22 | A22 | A31 | A22 | A31 | A31 | A31 |
| **P3** | A31 | A31 | A31 | A31 | A31 | A31 | none | A31 | none | none | none |
| **A22** | A22 | A22 | A22 | A22 | A22 | A22 | A31 | A22 | A31 | A31 | A31 |
| **A31** | A31 | A31 | A31 | A31 | A31 | A31 | none | A31 | none | none | none |
| **M22** | A31 | A31 | A31 | A31 | A31 | A31 | none | A31 | none | none | none |
| **M31** | A31 | A31 | A31 | A31 | A31 | A31 | none | A31 | none | none | none |

105 of the 121 cells have a result, so add has 105 rows in the order-only kingdom.

**Signature of every unary operator.** The result shape is P of the argument's order: P1 for a P1 argument; P2 for P2, A12, A21, M12, M21; P3 for P3, A22, A31, M22, M31. That is 11 rows per unary operator, one per argument shape.

### 2.3 The depth duals and the full set

A dual is a shape with a depth level: `shape.level`, for example `A21.T1`. With order and depth together there are 11 × 3 = 33 duals. Each row's level follows the rule in 2.1:

| operator class | operators in the SR alphabet | argument duals | result dual |
|---|---|---|---|
| sum | add, sub | (s₁.l₁, s₂.l₂) | (Merge_A(s₁, s₂)).max(l₁, l₂), no row if order would be 4 |
| product | mul, pdiv | (s₁.l₁, s₂.l₂) | (Merge_M(s₁, s₂)).max(l₁, l₂), no row if order would be 4 |
| raising unary | sin cos tan tanh psqrt plog pexp abs pasin pacos | s.l with l at most T1 | P(order of s).(l + 1); **no row for T2** |
| level-keeping unary | neg pow2 pow3 pinv | s.l | P(order of s).l |

Rows per operator: order only, 11 per unary and 105 per binary (574 rows); order and depth, 22 per raising unary, 33 per level-keeping unary and 945 per binary (4,132 rows, of which 352 unary and 3,780 binary). The depth-only kingdom (arm E) is the old `typed_depth_table` with ordered pairs: 3 duals, 68 rows.

**The root dual.** The root slot of a gene needs one type that any legal expression can fill. It is a dual of its own, `ROOT`, with a twin of every function row (same operator, same argument duals, output `ROOT`). Nothing takes `ROOT` as an argument, so its head is exactly one cell and its tail is empty (Section 3). A twin decodes to the same evaluator operator as the row it copies, so the decoder is unchanged.

**Terminals are polymorphic.** A slot of type `T` is filled by a function row outputting `T`, or by a terminal outputting `T`. Every variable and every constant placeholder is given one terminal row for every dual, so a leaf fits every slot. This is the "polymorphic spelling has one symbol per output type" of MultiTypedGEP.md section 1, and it is the only way to satisfy the check "a terminal for every demanded type" (kingdom_contract.rs:184-190), because a bare leaf has order 1 and a type such as A21 cannot otherwise have a terminal.

Because a leaf fits any slot, a dual is an **upper bound** on the subtree under the slot, not its exact profile. A function row's output is computed from the argument duals it asks for, so it is never below the true profile of what fills it. Two consequences:

- Soundness: any gene that expresses has true order at most 3 and true depth at most 2. The bound is monotone and the leaf is the bottom of the order, so the true profile is never above the declared one.
- Completeness: for any tree with root order at most 3 and depth at most 2, give every node its exact dual; each internal node is then an existing row, each leaf a terminal of that dual, and the root a twin row. So no legal tree is lost.

### 2.4 What types cannot carry: lower bounds

"exp never on a bare variable" is a lower bound (the argument must have order 2 or more). Every dual is an upper bound and every leaf fits every dual, so exp(x) is always spellable. Carrying the rule would need a slot type that no leaf fits, and a tail of that type would need terminals of order 2 or more. The kingdom check rejects exactly that. If leaves were given only the bottom dual, the check would print `our software has a bug: ... type Tsr(k) is demanded (a root or a required type) and has no terminal` for each of 33 types (32 argument types and the root; read from `kingdom_contract.rs:184-190`, not run).

So arms B, C, E and F drop the exp minimum. That agrees with the memory note's advice for any revisit ("drop the exp minimum"). The one honest way to keep it is a scaled-variable terminal (`k·x`, order 2 by Strahler, expanding to a multiplication at decode), which gives the exp slot a terminal. It is listed as arm G, a second run.

The upper bounds of the old windows are expressible: a row simply does not exist for an argument above the window. Arm F carries them (sin cos tan tanh pow2 pow3 pinv take arguments of order at most 2). The pair table is not carried at all: no ordering of unary operators is typed except through depth.

### 2.5 Worked examples

Notation `shape.level`. Constants and variables are `P1.F`.

| expression | typing | result |
|---|---|---|
| `x + y` | add(P1.F, P1.F) | A12.F, order 2 |
| `x + y + z` | add(A12.F, P1.F) | A12.F: a long sum is one confluence, order 2 |
| `x·y + z·w` | mul(P1.F,P1.F) = M12.F twice; add(M12.F, M12.F) | A22.F, order 3 |
| `(a·b + c·d)·(a·c + b·d)` | mul(A22.F, A22.F) | **no row** (order 4). Matches the engine's own test `deep`, which expects `Ceiling(4)` (`order.rs:434-435`) |
| `√(a·b + c·d) + √(e·f + g·h)` | psqrt(A22.F) = P3.T1 twice; add(P3.T1, P3.T1) | **no row** (order 4) |
| `sin(x)` | sin(P1.F) | P1.T1 |
| `sin(cos(x))` | cos(P1.F) = P1.T1; sin(P1.T1) | P1.T2 |
| `sin(cos(tan(x)))` | tan gives P1.T1, cos gives P1.T2, sin(P1.T2) | **no row** (a raising operator has no T2 argument) |
| `√(1 − v²/c²)` | pow2(P1.F) = P1.F twice; pdiv(P1.F, P1.F) = M12.F; sub(P1.F, M12.F) = A21.F; psqrt(A21.F) | P2.T1 |
| `1/√(1 − v²/c²)` | pinv(P2.T1) | P2.T1 (pinv does not raise). The Lorentz spelling the old pair table rejected is a plain typed tree |
| `m·v/√(1 − v²/c²)` | pdiv(M12.F, P2.T1) | M21.T1, order 2 |
| `exp(−x²/2)` as `pexp(mul(−0.5, pow2 x))` | mul(P1.F, P1.F) = M12.F; pexp(M12.F) | P2.T1 |
| `exp(−x²)` as `pexp(neg(pow2 x))` | neg(pow2(P1.F)) = P1.F; pexp(P1.F) | P1.T1: legal here, rejected by the old exp minimum |
| `x²`, `x³` | pow2(P1.F), pow3(P1.F) | P1.F (whole powers do not raise) |
| `(a + b)²` | pow2(A12.F) | P2.F |

Raw `pow` is not in the engine's SR alphabet (`SymbolTable::wide`, `engine.rs:160-171`); if added it would take a base dual and a literal exponent that is not an arm, which differs from `stream_order`, where a leaf exponent ties a leaf base (order 2). That mismatch must be settled by the sympy convention before `pow` is added.

### 2.6 What was checked

A host prototype (Python, Appendix A; to be committed as `kingdoms/tsr/gen_tables.py` and `check_types.py`) built the rows for each arm and compared "does this tree have a typing" against the rules computed independently from the sympy-style definition (arms by flattening, not by the profile the types are derived from):

| check | trees | disagreements |
|---|---|---|
| exhaustive, ceiling 2, all 18 operators, up to 6 nodes, 4 combinations of depth and windows | 696,103 each | 0 |
| exhaustive, ceiling 2, 5 operators plus the 4 binary, up to 9 nodes, depth and windows on | 4,335,881 (2,324,248 illegal) | 0 |
| exhaustive, same, order only | 4,335,881 (18,448 illegal) | 0 |
| random, ceiling 3, up to 40 nodes, 4 combinations | 60,000 each | 0 |
| random, ceiling 2, up to 30 nodes, depth and windows on | 60,000 | 0 |

Ceiling 2 is used for the exhaustive runs because an order-3 tree needs 7 nodes and an order-4 tree 15, so only ceiling 2 puts the boundary inside the enumerable range; ceiling 3 is covered by random trees (the generator reaches order 4 in them).

### 2.7 Gained and lost against the fixed-order rule (arm D)

| | the old rule (decode check, arm D) | types (arms B, C, F) |
|---|---|---|
| illegal shapes | built, expressed, dropped | cannot be written |
| individuals dropped at decode | counted in `typed_refused` | 0 by construction (a nonzero count is a bug) |
| root order at most 3 | yes | yes |
| depth at most 2 | yes (separate check) | yes (in the type) |
| argument windows, upper bounds | yes | arm F; not B or C |
| exp argument at least order 2 | yes | **no** (cannot be typed) |
| pair table | yes, harvested | none |
| stepping stones through illegal shapes | possible only inside one individual's expression | not possible in the genotype either; the search cannot pass through order 4 |
| gene width (one type 101 words at N = 17) | 101 | 3,656 (order) or 20,980 (order and depth) |
| equal spellings of one tree | 1 | 105 (order) or 945 (order and depth) for `x + y` alone, because the leaves can be read from any row |

### 2.8 One type is classical GEP

With the order limit and the depth limit both off, the generator emits one dual F and the 18 operators with all arguments and the result F. Then `a_{F,F} = 2` and `n_F = 2`, the layout is the physical Karva head and tail (`t = d = h + 1` at `h = 33`), and the contract at N = 17 gives width 101, the same as arm A. The root dual is not needed with one type (the root is F).

## 3. Layout and sizing under rev 4

Formulas (MultiTypedGEP.md §2): `h_T = [T = R] + (N − 1)·a_T`, `P_T = [T = R] + Σ_U h_U·a_{T,U}`, `L = 1 + Σ_U h_U·(n_U − 1)`, `t_T = max(0, min(P_T, L + h_T) − h_T)`, `d_T = min(P_T, L)`; every type is constant-capable, so each carries a Dc of length `d_T`.

**Program size.** One-type SR today has head 34 (`Config::head`, `engine.rs:1615`). Its guarantee is N = 17 (the largest N with `1 + (N − 1)·2` at most 34). The TSR kingdoms use N = 17 so both guarantee the same program size. The narrow variants B8 and C8 use N = 8.

| kingdom (arm) | duals | function rows (with root twins) | width N = 8 | width N = 17 | width N = 32 | against one type at N = 17 |
|---|---|---|---|---|---|---|
| one type (A) | 1 | 18 | 47 | 101 | 191 | 1.0 × |
| depth only (E) | 3 + root | 68 (136) | 252 | 558 | 1,068 | 5.5 × |
| order only (B) | 11 + root | 574 (1,148) | 1,622 | 3,656 | 7,046 | 36.2 × |
| order and depth (C) | 33 + root | 4,132 (8,264) | 9,244 | 20,980 | 40,540 | 207.7 × |
| order and depth, windows (F) | 33 + root | 4,047 (8,094) | 9,244 | 20,980 | 40,540 | 207.7 × |

Widths are in 32-bit words per gene, Dc included. A classical gene of head 34 is 104 words (34 + 35 + 35).

**Per-type budgets for arm C at N = 17.**

| quantity | value |
|---|---|
| `a_T` and `h_T` | 24 duals have `a_T = 2` and `h_T = 32`; 9 duals (P3, A31 and M31 at three levels; a row taking two of them would have order 4) have `a_T = 1` and `h_T = 16`; the root has `h = 1` |
| sum of heads | 913 |
| leaf bound L | 674 |
| tails `t_T` | between 33 and 674 (the root has 0); 3 duals at 674; sum 9,625 |
| Dc lengths `d_T` | between 49 and 674 (the root has 1); sum 10,442 |
| width | 913 + 9,625 + 10,442 = 20,980 |
| the root | `h = 1`, `P = 1`, `t = 0`, `d = 1` |

Arm B at N = 17: 8 duals with `h = 32`, 3 with `h = 16`, root 1; sum of heads 305, tails 1,619, Dc 1,732; L = 226.

Why so wide, in plain words: the exact tail bound says every type must be able to hold all the leaves of the biggest program (L = 674), because a program could consist of leaves of one type only. With 34 types, that is 34 tails and 34 Dc arrays that are nearly always empty. This is the design's exact, never-guessed bound, but it is the main practical cost, and arms B8 and C8 measure it.

**Memory.** An individual has 3 genes (`EVOLVE_N_GENES=3`, `law_runner.py:FIT_ENV`). With 2,000 rows (`EVOLVE_POP_INTAKE` 1,000 plus `EVOLVE_POP_CHAMPION` 1,000):

| arm | words per individual | per population copy |
|---|---|---|
| A | 303 | 2.4 MB |
| B | 10,968 | 87.7 MB |
| C | 62,940 | 503.5 MB |

**Sampling.** A head cell of type T draws one row of T's rows. In arm C the rows per output dual run from 4 to 560 (median 70), so operators with many argument combinations are drawn far more often than operators with few. The initialiser's "function share" (`FunctionShare`, MultiTypedGEP.md phylu notes) draws functions against terminals per type, not per operator. This changes the operator mix of the first population against arm A; it is recorded in the run card and is a risk (Section 8).

## 4. Kingdom contract checks

The check (`KingdomRelations::check`, `kingdom_contract.rs:78-211`) establishes: exactly one output row per symbol; ordered, contiguous signatures; a terminal for every root and demanded type. All three hold for the generated tables: one output per row by construction, argument order is the table's, and every one of the 34 duals has a polymorphic terminal for each variable and the constant placeholder.

Predicted output of `report` (`kingdom_contract.rs:506`) for the two main kingdoms at N = 17 and four input variables. The dual index is the row of `types.tsv`: arm B `Tsr(i)` with i the shape's position in the shape table of 2.2 ordered by (order, family, b, c) as P1 P2 A12 A21 M12 M21 P3 A22 A31 M22 M31, root last; arm C `Tsr(3i + l)` for shape i and level l, root last. These lines come from the reference model, not from Rust (the kingdom does not exist yet); the Rust check must print the same.

```
tsr-order (B): 1208 symbols (1148 functions, 60 terminals), 1 roots
  check passed; 12 types, n_max 2, program size N = 17
  root Tsr(11): 12 types reachable, L = 226, gene width 3656 words, heads guarantee N = 17
    Tsr(5): h 32 P 258 t 226 d 226
    Tsr(4): h 32 P 290 t 226 d 226
    Tsr(3): h 32 P 258 t 226 d 226
    Tsr(2): h 32 P 290 t 226 d 226
    Tsr(1): h 32 P 258 t 226 d 226
    Tsr(0): h 32 P 322 t 226 d 226
    ... 6 more types

tsr-order-depth (C): 8434 symbols (8264 functions, 170 terminals), 1 roots
  check passed; 34 types, n_max 2, program size N = 17
  root Tsr(33): 34 types reachable, L = 674, gene width 20980 words, heads guarantee N = 17
    Tsr(12): h 32 P 706 t 674 d 674
    Tsr(6): h 32 P 706 t 674 d 674
    Tsr(0): h 32 P 802 t 674 d 674
    Tsr(15): h 32 P 610 t 578 d 610
    Tsr(9): h 32 P 610 t 578 d 610
    Tsr(3): h 32 P 610 t 578 d 610
    ... 28 more types
```

The exact-leaf variant (a leaf only at the bottom dual) is the case the check must catch; it would print `CHECK FAILED, 33 defects` (read from the code, Section 2.4).

## 5. Changes by file, in build order

Each step ends with its tests green and its own commit. Cargo is never run while another cargo runs (`while pgrep -x cargo >/dev/null; do sleep 10; done`, then `CARGO_BUILD_JOBS=6`); a cold build is over 15 minutes and is not interrupted.

### fuller

| step | file | change |
|---|---|---|
| F1 | `kingdoms/tsr/gen_tables.py` (new, same shape as `kingdoms/safewgsl/gen_tables.py`) | parameters: ceiling R (or none), depth on or off, windows none or hi, program size. Writes `types.tsv`, `functions.tsv`, `TYPES.md`, `symbols.md` from the rules of Section 2; every row has a stable ref like the WGSL tables |
| F1 | `kingdoms/tsr/check_types.py` (new) | the exhaustive and random check of 2.6 against the sympy-style definition; runs with plain `python3 -I`, no GPU |
| F2 | `src/geneframe.rs` | `Ty::Tsr(u16)` appended after `Wgsl` (`:105`); code `TSR_CODE_BASE + i` with `TSR_CODE_BASE = Ty::TABLE_CODE_BASE + 4096`; `Ty::code` (`:146-153`) and `Ty::from_code` (`:164-170`, which today maps every table code to `Wgsl`) learn the second range |
| F2 | `src/tsr_kingdom.rs` (new), mirroring `src/wgsl/loader.rs` (`WgslKingdom::load`, `:297`) | loads the TSVs; polymorphic terminals (every variable and `?_T` at every dual); the `ROOT` twins |
| F3 | `src/kingdom_contract.rs` | `tsr_relations(variant, n_inputs)` beside `safewgsl_relations` (`:499`); extend `the_kingdom_reports_print` (`:759`) to print B and C at N = 8, 17, 32 |
| F3 | `src/typed_gep.rs`, `src/typed_chromosome.rs` | no change expected: the layout, walk, write-back and operators are generic over the contract. Verify with the tests of Section 6 |
| F4 | `src/geneframe.rs:383-430` (`typed_depth_table`) | kept until arm E has run and is retired; its tests (`:650-730`) pin the lockstep with `DEPTH_RAISING` |

### phylu

| step | file | change |
|---|---|---|
| P1 | `src/evolve/engine.rs` (`SymbolTable`, `:49-86` `Symbol`, `:160-171` `wide`) | `SymbolTable::tsr(n_inputs, variant)`: each row becomes `Symbol::Fn { id: the evaluator op, arity, out, ins }` (the table-defined row already exists), each polymorphic terminal a `Symbol::Input(col)` or `Symbol::Literal` with its dual from `KingdomTypes::leaf_ty` (`:142`); `KingdomTypes::root` = the root dual |
| P1 | `src/evolve/engine.rs` `Config` (`:1554`) | `sr_kingdom: single | order | order_depth | depth | order_depth_windows` and `program_size: Option<u32>` (replacing `head` for the TSR kingdoms); `src/evolve/card.rs` (near `:566`) reads `EVOLVE_SR_KINGDOM`, `EVOLVE_PROGRAM_SIZE`; the run card prints the kingdom, the number of duals and the gene width |
| P2 | `src/evolve/typed_write_back.rs` (`GeneCodec`) and every host rewrite | a rewritten tree must be typed before encoding: compute the exact dual of each node bottom-up and pick the row (`row_for(op, child duals)`); a tree whose root has no row is not offered (the operator's domain is enumerated before the draw, MultiTypedGEP.md §5). Call sites to route through it: `function_id` 21, `function_id_of_op` 1 (`engine.rs`, the graft encoder), `input_id` 4, `Symbol::Function(` 42 matches, `Symbol::Rnc` 19 matches |
| P2 | `src/evolve/constants.rs` | one pool `C_T` per dual (`ConstantPools::from_range`, `:55`), `n_rnc` constants each; `?_T` at every dual |
| P3 | `scripts/law_runner.py` | pass an arm's extra environment to the child (the child environment is built from a whitelist, `:239-275`); `FIT_ENV` stops hard-coding `EVOLVE_TYPED_DEPTH` (`:66`) |
| P3 | `scripts/sr_old_new.py` | generalise from two fixed arms (`ARMS = ("old", "new")`) to named arms read from `experiments/tsr_arms.json`; the table shows all arms side by side; the dev-seed guard stays |
| P3 | `experiments/tsr_arms.json` (new) | the eight arms of Section 7 |

### Removals (after the experiment, only if the rule of Section 7.5 selects types)

The old options stay until the experiment has finished, because arms A, B and D use them. When types are adopted, remove, keeping `stream_order` and `t_depth` as the test oracle and as sensors:

| file:line | remove |
|---|---|
| `src/evolve/typed_decode.rs:99-104` | `DecodeOptions` (`typed`, `order_ceiling`) |
| `src/evolve/typed_decode.rs:118-121, 126-134` | `DecodeOutcome::NotExpressed`, `TY_NONE`, `family` |
| `src/evolve/typed_decode.rs:230-321` | the depth and order derivation in `decode_reference` (the node array and hash at `:322-329` stay) |
| `src/evolve/typed_decode.rs:444, 448, 529, 703-713` | `op_raiser` and `op_rules` uploads, the parameter words, the two parity tests (replaced by the TSR parity tests) |
| `src/evolve/typed_decode.wgsl:23, 48, 59, 63-65, 137-138, 174-175, 285-305, 338-354` | `order_ceiling`, `TY_NONE`, the `op_raiser` and `op_rules` bindings (frees two of Metal's 29 buffer slots), `is_raiser_op`, the typed derivation, the order block |
| `src/evolve/typed_device.rs:688-689` | the `typed` and `order_ceiling` arguments of `decode` |
| `src/evolve/resident.rs:47-49, 301, 367` | `Run::typed`, `Run::order` and their use |
| `src/evolve/engine.rs:1554, 1562, 1722-1723, 9177-9178` | `Config::typed_depth`, `typed_order` and defaults |
| `src/evolve/engine.rs:4028-4033, 4428-4442, 5195-5196` | validation, the `typed`/`order` arguments of the chain run, the graft filters |
| `src/evolve/engine.rs:2036, 3709, 4342, 4479, 8562, 10380` | the `typed_refused` counter (a gene that does not express is a decode defect, `check_decode`) |
| `src/evolve/engine.rs:378-441, 458-459, 473-478` | `op_raiser`, `op_order`, `op_stack` construction in `typed_codes`; `depth_cost`, `sample_flat` |
| `src/evolve/mod.rs:251-266, 314-342, 402-449` | `TypedCodes::op_raiser/op_order/op_stack` and their checks; `SymbolCodes::depth_cost/sample_flat` and checks |
| `src/evolve/mod.rs:458-470`, `src/evolve/vary.rs:272` and their call sites (`engine.rs:7778, 8050, 11089-11141`, `typed_device.rs` tests, `subexpr.rs:101, 211`, `wgsl_typed_fit.rs:462, 586`, `engine/regex_profile.rs:24`) | `InitParams::typed_depth`, `GenParams::typed_depth` |
| `src/evolve/write_back.rs:403-404, 452-455` | the `typed_depth` and `typed_order` parameters |
| `src/evolve/order.rs:63-76, 93-188, 237-319, 337-347` and the tests that use them (`:376-455`) | `RULES`, `STACKS`, `canon`, `may_stack`, `op_stack_table`, `arg_bounds`, `OrderRefusal`, `order_ok`, `unary_stacks`, `op_order_table`. Keep `arity_of`, `family`, `stream_order`, `op_name` |
| `src/evolve/card.rs:566-576, 649, 748-773` | `EVOLVE_TYPED_DEPTH`, `EVOLVE_TYPED_ORDER` and their tests |
| `src/evolve/regex_typed.rs:189, 437-438`; `examples/regex_fit.rs:251-252`; `examples/evolve_fit.rs:781`; `examples/order_study.rs` | the matching fields and the study that uses `order_ok` |
| `scripts/law_runner.py:66, 106, 239-250, 364-371, 440-442`; `scripts/srbench_tf_run.py:42-49, 167, 187, 210`; `scripts/phylu_web_watch.py:132`; `scripts/runpod/README.md:36`; `experiments/README.md:23, 94`; `experiments/tsr-4k/README.md:16` | the switch and its documentation |

Init and mutation need no depth budget and no change: the typed initialiser and the typed variation operators already draw a row of the type of the cell, so a legal cell is the only draw there is.

## 6. Tests that certify it

| # | test | numbers |
|---|---|---|
| 1 | Closure over random genes: for arms B, C, E, F with 1 and 9 input variables, 100,000 random genes each from the typed initialiser; decode with `decode_reference`; node by node, `stream_order` at most 3 (B, C, F), `t_depth` at most 2 (C, E, F), windows hold (F); no gene fails to express | 100,000 × 4 arms × 2 |
| 2 | Closure of every operator: 100,000 children each from point mutation, inversion, IS, RIS, gene and tail transposition, one- and two-point crossover, gene recombination; plus cleanse, snap and fold write-back results and the beam's grafts; all pass test 1's checks, and an operator with an empty domain is absent from the schedule, never a no-op draw | 100,000 per operator |
| 3 | Type system against the rules (Section 2.6) as a committed check; the generated TSVs are the input | 4.3 M exhaustive, 300 k random |
| 4 | Classical equality: with order and depth off the generator's rows equal `SymbolTable::wide` (18 operators, one dual); the contract at N = 17 gives width 101 and `t = d = 34`; decode of 100,000 genes equals arm A's byte for byte | `the_typed_sr_generations_match_the_pinned_golden` (`typed_device.rs:940`, fixture `tests/fixtures/golden_typed_sr.txt`) unchanged |
| 5 | Host and device parity on the TSR layouts: `decode_reference` against the kernel, and typed init and vary against their host twins, in the style of `typed_decode.rs:703-713` and `typed_device.rs:785-960`; run one GPU process at a time | arms B and C |
| 6 | Contract: bound exactness (the program that attains `P_T`, the one that attains `L + h_T`, the chain that attains `h_T` from N) for B and C; the widths 3,656, 20,980, 1,622, 9,244 pinned; the check fails by name on the exact-leaf variant (33 defects) | MultiTypedGEP.md §7 items 1-2 |
| 7 | The golden: after test 5 first passes, pin one line per TSR arm in `tests/fixtures/` with `PHYLU_PIN_GOLDEN=1` at that commit (the pattern at `typed_device.rs:930-1000`); later commits compare | 2 new lines |
| 8 | Redundancy: for `x + y` the count of distinct row choices that decode to it equals 105 (B) and 945 (C); written as a test so a change to the leaf rule is noticed | 105, 945 |

## 7. Experiments

Andrew's rule: build every arm and measure; nothing is guessed and nothing is left to a choice. Dev seeds 7001 to 7010 only; SRBench's official seeds are rejected by the harness (`sr_old_new.py`, the `OFFICIAL_SEEDS` guard). Stop: 1,000,000 counted candidate evaluations, with the fixed law bar as early stop (log10 p at most −19 and 1 − R² at most 1e-6).

### 7.1 Arms

| arm | kingdom | decode checks left on | what it tests |
|---|---|---|---|
| A | one type (today) | depth check (`EVOLVE_TYPED_DEPTH=2`) | the baseline |
| B | order duals, N = 17 | depth check | order carried in types |
| C | order and depth duals, N = 17 | none | the whole design |
| D | one type | depth check and `EVOLVE_TYPED_ORDER=3` | the old hard rule, the reference |
| E | depth duals only, N = 17 | none | depth carried in types, never run before |
| F | C plus the argument windows | none | how much of D's rule types can carry |
| B8 | B at N = 8 | depth check | the cost of width |
| C8 | C at N = 8 | none | the cost of width |

Together A, B, C, E form a 2 × 2 (order: none or types; depth: decode check or types), and D, F bracket the old rule.

`experiments/tsr_arms.json` (to be created in step P3) maps each arm name to environment variables, for example `"C": {"EVOLVE_SR_KINGDOM": "order_depth", "EVOLVE_PROGRAM_SIZE": "17"}` and `"D": {"EVOLVE_SR_KINGDOM": "single", "EVOLVE_TYPED_DEPTH": "2", "EVOLVE_TYPED_ORDER": "3"}`.

### 7.2 Metrics

For each arm and law, over the 10 seeds: seeds passing the strict test; median counted evaluations to stop over the seeds that pass; median length in characters of `model_plain` over the seeds that pass; median seconds per seed; generations per second; seeds killed by the clock; gene width from the run card. Source rows: `law_runner.py` `result.json`, as in `logs/SR_OLD_NEW/run1`. Baseline for A on the 40 laws (measured in that folder, new arm): 380 of 400 passed, median 179,820 evaluations to stop, 90th percentile 269,640, median model 31 characters, 90th percentile 50, median 0.2 s and at most 1.4 s per seed.

### 7.3 Expectations, stated before the run

These are predictions, each with its basis. None is a measurement of the new arms.

| measurement (first 40 laws of `experiments/order133.txt`, 10 dev seeds, 400 seed-runs per arm) | expected | basis |
|---|---|---|
| seeds passing the strict test, A | 380 (95.0 %) | measured, `2026-10-20_sr_typed_gep.md`; re-measured in the same session |
| seeds passing, B, C, E | A minus 6 to A plus 6 | one standard deviation of the difference between two arms at 95 % over 400 runs is sqrt(2 · 400 · 0.95 · 0.05) = 6.2; types change which shapes are built, and on easy laws a ceiling changed shape and not outcome (0/60 against 0/60) |
| seeds passing, D | A minus 7 | the hard rule lost 14 of 810 solved on 81 laws in `AB_ORDER_EASY` against `TF_DEV1` (1.7 %, which is 7 of 400); older engine, so a weak basis |
| seeds passing, F | between C and D | carries the upper bounds, not the exp minimum or the pair table |
| seeds passing, B8, C8 | no more than B, C | a smaller guarantee (8 functions in any shape) can only lose programs |
| median evaluations to stop, passing seeds | A 179,820; B, C, E within 10 % of A | most of these 40 laws are found in 2 to 3 generations, little room to gain |
| median model length (characters), passing seeds | A 31; B, C, E 28 to 34 | types do not penalise length |
| the Lorentz and Gaussian laws D lost, in the first 40 | only feynman_I_6_2a is in the first 40 (position 39). A 6 of 10 (measured, new arm), D 3 or fewer of 10 (earlier engine: 3 of 10), B, C within 2 of A | `logs/SR_OLD_NEW/run1`; `AB_ORDER_EASY` |
| seconds per seed | A 0.2 s; B between 1 and 36 times A; C between 1 and 208 times A | gene width ratios 36.2 and 207.7; only the kernels that touch every cell scale with width, so the true factor is between 1 and the ratio; the calibration run narrows it |
| gene width in words, from the run card | B 3,656; C 20,980; E 558; B8 1,622; C8 9,244 | computed (Section 3); must match exactly |
| individuals not expressed at decode | C, E, F: 0; B: 0 for order, counted for depth; A, D: counted | by construction |

All 133 laws (after the first 40): the six laws D lost, with the old-engine counts of 10 (`AB_ORDER_EASY` against `TF_DEV1`).

| law (position in order133) | D then | A then | expected now: D | expected now: B, C |
|---|---|---|---|---|
| feynman_II_13_23 (52) | 1 | 8 | at most A minus 3 | within 2 of A |
| feynman_I_10_7 (64) | 3 | 7 | at most A minus 3 | within 2 of A |
| feynman_II_13_34 (71) | 1 | 5 | at most A minus 3 | within 2 of A |
| feynman_I_12_11 (77) | 5 | 9 | at most A minus 3 | within 2 of A |
| feynman_I_15_10 (79) | 1 | 6 | at most A minus 3 | within 2 of A |
| feynman_I_6_2a (39) | 3 | 8 | at most A minus 3 | within 2 of A |

### 7.4 Commands

Checks before every launch (one GPU process at a time):

```
pgrep -fl 'evolve_fit|regex_fit|wgsl_fit|law_runner|sr_old_new|cargo'; uptime
```

Build once after steps F1 to P3, copying the binary to its own path (an arm never runs from a path a build can overwrite):

```
cd /Users/andrewmorgan/Dev/gamakon/phylu
while pgrep -x cargo >/dev/null; do sleep 10; done
CARGO_BUILD_JOBS=6 cargo build --release --features gpu --example evolve_fit
mkdir -p logs/TSR_TYPES/bin && cp target/release/examples/evolve_fit logs/TSR_TYPES/bin/evolve_fit_tsr_20261020
```

Calibration (3 laws, to learn seconds per seed for each arm before the long run):

```
nohup python3 scripts/sr_old_new.py --binary logs/TSR_TYPES/bin/evolve_fit_tsr_20261020 \
  --arms-file experiments/tsr_arms.json --arms A,B,C,D,E,F,B8,C8 \
  --laws experiments/order133.txt --limit 3 --seeds 7001,7002,7003,7004,7005,7006,7007,7008,7009,7010 \
  --evaluations 1000000 --seconds 600 --out logs/TSR_TYPES/calib > logs/TSR_TYPES/calib.out 2>&1 &
tail -f logs/TSR_TYPES/calib/sweep.log
```

Run 1, the first 40 laws (the likelihood order), arms alternate per law so any drift hits every arm alike:

```
nohup python3 scripts/sr_old_new.py --binary logs/TSR_TYPES/bin/evolve_fit_tsr_20261020 \
  --arms-file experiments/tsr_arms.json --arms A,D,E,B,C,F,B8,C8 \
  --laws experiments/order133.txt --limit 40 --seeds 7001,7002,7003,7004,7005,7006,7007,7008,7009,7010 \
  --evaluations 1000000 --seconds 600 --out logs/TSR_TYPES/run1 > logs/TSR_TYPES/run1.out 2>&1 &
tail -f logs/TSR_TYPES/run1/sweep.log
./target/release/phylu-sr-watch --follow logs/TSR_TYPES/run1/stream.jsonl
```

Run 2, all 133 laws, with the arms that pass the rule of 7.5 plus A and D: the same command with `--limit 0` and `--out logs/TSR_TYPES/run2`.

`--seconds 600` is only the safety kill (the stop is the evaluation count); a seed killed by the clock is listed and counts as not passed.

### 7.5 Time

| step | estimate | basis |
|---|---|---|
| cold build of fuller and phylu | over 15 minutes | `Rules.md` |
| arms A and D, 40 laws | 8 minutes each | measured: 16.3 minutes (04:54:56Z to 05:11:15Z) for 800 seed-runs of the old and new arms |
| the other six arms, 40 laws | between 8 minutes (no width cost) and 8 minutes × the width ratio (all of the cost scales with width): E 44 minutes, B 4.8 hours, B8 2.1 hours, C and F 27.7 hours, C8 12.2 hours | computed bounds; the true value is in between and the calibration run measures it before the long run |
| all arms, 40 laws | 64 minutes best case, 75.6 hours worst case | sum of the above |
| all 133 laws, per arm | about 3.3 times the 40-law figure | 133 / 40; later laws are harder, so a lower bound |

The long runs go overnight under `nohup` with the log, one at a time.

### 7.6 Pass and fail rule

Fixed before the run. Let `sd` be sqrt(2 · n · p · (1 − p)) with n the seed-runs and p arm A's pass rate (6.2 for the first 40 laws).

| rule | condition |
|---|---|
| 1. no harm | arm total passes at least A's total minus `sd`, on the 40 laws and on the 133 laws |
| 2. the lost laws | on the six laws D lost, the arm's combined passes at least A's combined passes minus `sd` |
| 3. benefit (any one) | total passes at least A plus `sd`; or median evaluations to stop at least 10 % lower over the laws both arms solve; or wall time per generation within 25 % of A together with the structural benefit (no individual rejected at decode) |

An arm passes when rules 1 and 2 hold and rule 3 holds. If B or C passes, the TSR kingdom replaces SR and the removal list is executed; the better of B and C by total passes is kept. If neither passes, types are not adopted, the decode options stay, and the result is recorded as a null: "types did not improve the search". Arm D is never adopted; it is the reference for how much of the old rule types recovered. Arms B8 and C8 are reported either way, for the width cost.

## 8. Risks and open questions

| # | risk | what to do |
|---|---|---|
| 1 | Width. 20,980 words per gene and 62,940 per individual in arm C, and most of it is empty tails and Dc. Initialisation, variation and copying touch every cell. | Calibration run measures it; B8 and C8 measure the narrow guarantee. If the width is the problem and the search is not, the follow-up is a tighter tail bound that respects the design's exactness rule, not a smaller buffer |
| 2 | Many equal spellings of one tree: 105 (B) and 945 (C) rows can express `x + y`, because a leaf fits any slot. The genotype fragments: two individuals with the same tree can differ in every row, which may weaken crossover. | Record distinct genotype hashes against distinct phenotype hashes per generation (the engine keeps `gene_hash`); the experiment shows whether it costs search |
| 3 | Crossover. Homologous positions are per type and per gene id (MultiTypedGEP.md §5). The root cell is always the one cell of `ROOT`, which holds a twin of a function row, so two parents' roots can disagree on the argument duals of their children. | Measured by the pass rate; if it hurts, the alternative root is an identity wrapper with one row per argument dual (33 rows instead of the 4,132 twins), at the cost of one extra evaluator node |
| 4 | Operator mix. Rows per output dual run from 4 to 560 in arm C, so the draw is by row, not by operator: an operator with many argument combinations is over-represented against arm A. | The initial operator mix goes in the run card. A weighting that equalises operators is a change to the initialiser's draw (`typed_init.wgsl`), made only if the first run shows the mix matters |
| 5 | The stack table must be built from raw spellings. The type design has no stack table at all; arm D's table stays as harvested (tidied spellings) and its misses are exactly what Section 1.3 describes. If a pair table is ever re-introduced as a type, it must be harvested from `hof.tsv` and `MODEL_PLAIN` of raw genes, not tidied ones. | Nothing to build now |
| 6 | Engine spellings. Neg, Abs, the `1/sqrt` route, Compound operators (`SqrtSum`, `InvSqrtDiff`, ... which expand to Add plus Sqrt plus Inv; none is in `wide()`). The type design treats Neg and Abs like any unary (order of the argument), as `stream_order` does; compound rows, if a table ever has them, are the composition of the member rows. Raw `Pow` is not in the alphabet; see 2.5. | Test 3 includes all 18 operators |
| 7 | Rewrites. Cleanse, snap, fold, beam and subexpression regression make trees. Each must pick rows by type and drop a rewrite that has no row (a snap to a constant never raises order, fold never does, a beam graft can). The count of call sites in P2 (21 + 1 + 4 + 42 + 19) is the size of this work. | Test 2 covers every rewriting operator; this is the largest engineering item |
| 8 | The exp minimum is not carried. If it matters, arm G adds scaled-variable terminals so the exp slot has a terminal of order 2 or more. | Built as a second run if the first run shows D beating C on the exp laws |
| 9 | Stepping stones. The old A/B asked whether the search needs illegal shapes to reach legal ones. In types, order 4 cannot be built even in an intermediate individual. | The first run measures it directly: B, C, F against A on the laws where A passes through large intermediate shapes |
| 10 | The measured laws are sympy trees; the engine's binary spelling is flattened by family to match them (`order.rs:12-18`). A kingdom that flattens differently (for example treating Sub as its own family) would type a different set of trees. | The generator takes the family map as data; the check compares it with `stream_order` |
| 11 | Sampling of constants: one pool per dual (34 pools of 10 constants in arm C) instead of one pool. Constants are a leaf at every dual. | Same range `rnc_lo`, `rnc_hi`; recorded in the card |

No question is left for Andrew. Each binary choice here (order only or order and depth, windows or not, N = 17 or 8, twin root rows or an identity wrapper, exp minimum dropped or kept) is an arm or a follow-up arm of Section 7.

## Appendix A. Reference model used for the numbers

This is the prototype that produced every count in this plan; it becomes `kingdoms/tsr/gen_tables.py`. Run with `python3 -I`.

```python
BIN = {"add": "A", "sub": "A", "mul": "M", "pdiv": "M"}   # operator -> family
UN = ["sin","cos","tanh","psqrt","plog","pexp","neg","abs","pow2","pow3","pinv","tan","pasin","pacos"]
RAISE = {"sin","cos","tan","tanh","psqrt","plog","pexp","abs","pasin","pacos"}
HI2 = {"sin","cos","tan","tanh","pow2","pow3","pinv"}      # argument order at most 2 (arm F)

def order(p):  return p[1] + (1 if p[2] >= 2 else 0)       # p = (family, b, c)
LEAF = (None, 1, 1)
def unary(p):  return (None, order(p), 1)
def binary(fam, x, y):
    b = c = 0
    for ch in (x, y):
        ob, oc = (ch[1], ch[2]) if ch[0] == fam else (order(ch), 1)
        if ob > b: b, c = ob, oc
        elif ob == b: c += oc
    return (fam, b, min(c, 2))

def shapes(R):                                              # closure of LEAF under the operators, order <= R
    seen, new = {LEAF}, [LEAF]
    while new:
        new = []
        for p in list(seen):
            q = unary(p)
            if order(q) <= R and q not in seen: seen.add(q); new.append(q)
        for f in "AM":
            for x in list(seen):
                for y in list(seen):
                    q = binary(f, x, y)
                    if order(q) <= R and q not in seen: seen.add(q); new.append(q)
    return sorted(seen, key=lambda p: (order(p), p[0] or "", p[1], p[2]))

def build(R=3, depth=False, hi=False):                      # R=None: no order limit, one shape (arm E)
    levels = [0, 1, 2] if depth else [None]
    shp = shapes(R) if R else [LEAF]
    un = (lambda p: unary(p)) if R else (lambda p: LEAF)
    bi = (lambda f, x, y: binary(f, x, y)) if R else (lambda f, x, y: LEAF)
    legal = (lambda q: order(q) <= R) if R else (lambda q: True)
    types = [(p, l) for p in shp for l in levels]
    rows = []                                               # (operator, argument duals, result dual)
    for op in UN:
        for (p, l) in types:
            if hi and op in HI2 and order(p) > 2: continue
            if depth and op in RAISE and l >= 2: continue
            rows.append((op, ((p, l),), (un(p), (l + (op in RAISE)) if depth else None)))
    for op, fam in BIN.items():
        for t1 in types:
            for t2 in types:
                q = bi(fam, t1[0], t2[0])
                if legal(q):
                    rows.append((op, (t1, t2), (q, max(t1[1], t2[1]) if depth else None)))
    return types, rows

def bounds(rows, N, root="ROOT"):                           # rev 4: h, P, t, d per dual, L, width
    fn = rows + [(op, ins, root) for (op, ins, out) in rows]  # the root's twin rows
    a, n = {}, {}
    for op, ins, out in fn:
        for t in set(ins): a[(t, out)] = max(a.get((t, out), 0), ins.count(t))
        n[out] = max(n.get(out, 1), len(ins))
    reach, i = [root], 0
    while i < len(reach):
        for (t, u) in a:
            if u == reach[i] and t not in reach: reach.append(t)
        i += 1
    aT = {t: max([a.get((t, u), 0) for u in reach]) for t in reach}
    h = {t: (t == root) + (N - 1) * aT[t] for t in reach}
    L = 1 + sum(h[u] * (n.get(u, 1) - 1) for u in reach)
    P = {t: (t == root) + sum(h[u] * a.get((t, u), 0) for u in reach) for t in reach}
    tt = {t: max(0, min(P[t], L + h[t]) - h[t]) for t in reach}
    d = {t: min(P[t], L) for t in reach}
    return h, P, tt, d, L, sum(h[t] + tt[t] + d[t] for t in reach)
```

Calls: arm B `build(3)`, C `build(3, depth=True)`, E `build(None, depth=True)`, F `build(3, depth=True, hi=True)`; `bounds(rows, N)[5]` is the width. This code reproduces every width in Section 3 (E 252, 558, 1,068; B 1,622, 3,656, 7,046; C and F 9,244, 20,980, 40,540 at N = 8, 17, 32). Checks of the prototype against printed figures: the one-type SR kingdom gives width 47 at N = 8 and 191 at N = 32 (MultiTypedGEP.md, kingdom check figures); the two-type fixture of MultiTypedGEP.md §3 gives `P_S = 3, P_I = 2, L = 3, t = 1, 1, d = 3, 2`.
