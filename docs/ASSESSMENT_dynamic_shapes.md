# Assessment: shapes that are not roles — the third piece

**Status:** a design, measured where it could be. Nothing built. This is the piece `docs/ASSESSMENT_first_class_shapes.md` §13 named and §15.7 left open: the workload is *general array programs with dynamic dimensions*, and the class kingdom covers the half of it whose dimensions are the interface's **roles**.

Everything marked **measured** was measured against Futhark 0.28.1 on this machine; the transcripts are in `/tmp/shape/probe3/`. Everything marked **designed** is a proposal.

---

## 0. The answers, up front

1. **Futhark already expresses data-dependent dimensions.** Its sizes are not literals: a size parameter can be bound to a *runtime* value, and a `def` can be entered with sizes the caller supplies. The engine's problem is not that Futhark cannot express these programs — it is that the engine's *type table* is finite and its symbols are drawn from an alphabet enumerated before the draw.
2. **The third piece is therefore not a new IR with a typechecker.** It is the **same class machinery**, with two changes: a shape class's axes may be *dynamic*, and a dimension becomes a **value** of its own sort. The unification stays offline in the generator, the alphabet stays finite, and the GEP core is untouched — every invariant the architecture rests on survives.
3. **The cost is measured: the gene width goes from 322 words to 991** at three roles (3.08×), and the kingdom from 11 types and 71 rows to 27 types and 181 rows. The width is the real price, and it is the same price the class set already charges: a wider alphabet of types means a wider gene.
4. **What it does not buy is static checking of the dynamic relations.** Where a dimension is computed, the relation between two arrays becomes a *runtime* obligation — and Futhark 0.28.1's runtime size obligation **fails with an internal compiler error**, not a diagnostic. The engine must therefore treat a failed candidate as an ordinary failed candidate, and the harness must not confuse that with a timeout.
5. **The honest verdict:** this is the right shape of answer, and it is *not* free. It is the dynamic half of the split architecture §13 recommends. It is a *design* until the class set is generated, the interpreter models it, and the experiments of §6 have been run.

---

## 1. What Futhark's size discipline actually is

This is the part that had to be measured before anything could be designed, because the design has to agree with Futhark's rules **exactly** — a class system that is stricter than Futhark is safe but useless, and one that is looser is unsound.

### 1.1 Sizes are tracked inside a `def`, and they are values

**Measured.** Futhark's sizes are not erased inside a function body:

```futhark
def t [n][m] (a: [n][m]f32) : [m][n]f32 = transpose a     -- COMPILES
def sz [n] (xs: [n]f32) : i64 = n                         -- COMPILES
```

The first says the body's type is `[n][m]f32` — `transpose` swaps the roles. The second says a size parameter is an ordinary `i64` value in scope. So Futhark's type system is already **parametric over sizes**, and the engine's class set is a *finite specialisation* of it: the class kingdom pins the sizes to the interface's roles.

### 1.2 There are three tiers of size, and they do not mix

**Measured.** The tiers are:

| tier | an example | agrees with |
|---|---|---|
| **static** | a literal, `3` | only an equal literal |
| **symbolic** | `length ys` | only the same expression |
| **anonymous** | the size of an erased `[]f32` | nothing — a runtime obligation |

and the rules Futhark enforces are **exact**:

```futhark
def mk [n] (xs: [n]f32) : [n]f32 = xs
def dyn  (ys: []f32) : []f32 = mk (replicate (length ys) 0.0)   -- COMPILES:  n := length ys
def dyn2 (ys: []f32) : [3]f32 = mk (replicate (length ys) 0.0)   -- REFUSED:
--   Sizes "3" and "length ys" do not match.
```

**Futhark never defers a static size against an anonymous one.** The mismatch is a *compile-time* error, not a runtime check. That is the single most important fact for the design: the class system's "the inner axes must be the same" is not a strengthening of Futhark's rule — it *is* Futhark's rule, and a class system that said otherwise would be claiming to check something Futhark rejects.

### 1.3 Anonymous sizes defer to run time — and 0.28.1's runtime path is broken

**Measured.** When both sizes are anonymous, there is no static obligation at all:

```futhark
def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 = ...
entry main (A: [][]f32) (B: [][]f32) = matmul A B     -- COMPILES, and RUNS
```

With matching sizes it is correct (measured: `[[1,2],[3,4]] × [[5,6],[7,8]]` gives `[[19,22],[43,50]]` ✓). With mismatched ones:

```
Internal compiler error (unhandled IO exception).
Please report this at https://github.com/diku-dk/futhark/issues
b is not bound to a value.
-> #0  /prelude/soacs.fut:51:3-22
```

**That is a compiler defect, not a diagnostic.** Futhark 0.28.1 cannot report a runtime size mismatch. The engine therefore cannot rely on Futhark to *tell it* that a dynamic shape obligation was violated; it can only observe a non-zero exit. That is survivable — a candidate that fails is a candidate that fails — but it has to be a *designed* behaviour, not an accident (§4.3).

### 1.4 A dynamic dimension is expressible in two dimensions

**Measured.** A 2-D array whose *row* count is computed and whose *column* count is static is ordinary Futhark:

```futhark
def build (xs: []f32) : [][3]f32 = replicate (length xs) (replicate 3 1.0)
```

and it multiplies correctly against a static `[3][2]` (measured: four rows of `6.0` ✓). Note the type: `[][3]f32` — the row count is *anonymous*, the column count is *static*. Futhark's own type system therefore has exactly the structure the design needs: **each axis is separately static or dynamic.**

---

## 2. The design

### 2.1 A shape class's axes may be dynamic

**Designed.** A class is still a *type*, and the unification is still offline. The only change is that an axis is not a role but a member of a finite set:

```
axis ::= role            -- n, m, p: a size parameter of the kernel
      | dyn             -- anonymous: Futhark's erased size
```

so the class set for `R` roles becomes:

| | roles only (the built PoC) | with dynamic axes |
|---|---|---|
| matrix classes | `R²` | `(R+1)²` |
| 1-D classes | — | `R+1` |
| dimension classes | — | `R+1` |
| function classes | 2 | 2 |

The relations are resolved **offline, in the generator**, exactly as they are now:

* `map_rows` and `map2_rows` preserve a class ✓ (the shape-preserving operators, unchanged);
* `transpose` swaps the axes — so `mat.dyn.m` transposes to `mat.m.dyn` ✓;
* **`matmul`'s inner axes must be the *same* axis value**: `mat.dyn.m × mat.m.p → mat.dyn.p` ✓ (statically known, both `m`), `mat.dyn.m × mat.dyn.p` is **illegal** (the inner axes are `m` and `dyn`, which Futhark also refuses — §1.2), and `mat.dyn.dyn × mat.dyn.dyn → mat.dyn.dyn` ✓ (the obligation is deferred to run time, as Futhark does).

That last row is the important one: **the class set's rule and Futhark's rule are the same rule.** Every class-consistent program compiles; every program the engine can build is one Futhark accepts.

### 2.2 A dimension is a value of its own sort

**Designed.** A dimension is an ordinary value in the tree, of a new type `Ty::FutharkDim`. Its operators are a **finite** set, and that is the whole reason it fits the architecture: the *alphabet* is finite (the operator set), while the *tree* is unbounded (as everywhere else in a GEP).

| operator | signature | meaning |
|---|---|---|
| `length` | `mat.a.b → dim.a` | the row count; the role when it is known |
| `length1` | `vec.a → dim.a` | the length of a 1-D array |
| `replicate` | `dim.a, f32 → vec.a` | build an array of that size |
| `replicate2` | `dim.a, dim.b, f32 → mat.a.b` | build a matrix of that shape |
| `const_dim` | `f32 → dim.dyn` | a literal size, which Futhark knows and the engine does not |

The dimension's *class* carries what is known: `dim.n` is a dimension that is statically the role `n`; `dim.dyn` is one whose value is only known at run time. So:

* `length A` where `A : mat.n.m` gives `dim.n` — **statically** the row count ✓, and `replicate` of it gives `vec.n` ✓;
* `length A` where `A : mat.dyn.m` gives `dim.dyn`, and `replicate` of it gives `vec.dyn` ✓.

The engine's classes are then **coarser** than Futhark's types wherever a literal is involved: `replicate 5 x` is `[5]f32` in Futhark and `vec.dyn` in the engine. That is sound — the engine says less than the compiler knows — and it is the price of a finite class set. A literal size cannot be a class, because the set of literals is infinite.

### 2.3 Why this is not "a separate shape-aware IR"

**Designed.** The obvious alternative — a separate IR whose types carry dimension *expressions*, checked by a typechecker — is what §14 called "not ready at all", and the reason still holds: **a typechecker is legality by *checking*, and this architecture is built on legality by *construction*.** A checked IR would need a repair or a rejection step for every candidate, and §3 of the design assessment is the argument that neither is available.

The design above keeps everything:

| invariant | held how |
|---|---|
| the alphabet is enumerable before the draw | the class set is finite, as it is now; the dim sort's operators are finite |
| legality by construction | the class set and the dim classes are resolved offline, in the generator |
| no repair, no rejection | a tree that expresses is a legal program |
| the GEP core is untouched | a new type in the type space, like `Ty::FutharkShape` itself |
| the unification happens once, offline | the generator enumerates the extended class set |

### 2.4 The sharing question, answered

**Designed.** A computed dimension used twice — "compute `n = length xs`, then build an `n×m` matrix and an `n×k` matrix" — needs the value in two places. The tree is a tree, so the expression is *duplicated*: `length xs` twice. That is correct, because the expression is pure and Futhark computes it twice, and it is what the 1-D kingdom already does with `length`. The population DAG's homeotic rows would make it a *shared definition* — an optimisation, and a change to the gene's encoding, not a requirement of the semantics. **So the third piece does not need the DAG.** That is worth stating, because it is the point at which a design like this usually starts pulling in the rest of the system.

---

## 3. The cost

### 3.1 The model, and its validation

The bound formulas are transcribed from `fuller/src/kingdom_contract.rs` and validated two ways: against the published two-type fixture, and against the **built** class kingdom — the model reproduces the engine's own numbers exactly (11 types, 71 rows, 11 frames, gene width **322** at N = 3, leaf bound **75**). The dynamic model is priced with the same formulas, so the *ratio* is trustworthy even though the absolute figures are a model.

The script is `/tmp/shape/dyn_counts.py`.

### 3.2 The numbers

| | the built kingdom | with dynamic axes | ratio |
|---|---:|---:|---:|
| types | 11 | 27 | 2.45× |
| symbol rows | 71 | 181 | 2.55× |
| frames from the root | 11 | 27 | 2.45× |
| **gene width at N = 3** | **322** | **991** | **3.08×** |
| L (the leaf bound) | 75 | 139 | 1.85× |

and the growth in the number of roles:

| roles | matrix classes | types | rows | frames | width | L |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9 | 18 | 99 | 18 | 493 | 81 |
| 3 | 16 | 27 | 181 | 27 | 991 | 139 |
| 4 | 25 | 38 | 299 | 38 | 1737 | 213 |
| 5 | 36 | 51 | 459 | 51 | 2779 | 303 |

**The width is the real cost**, and it grows faster than the class count: the width is quadratic in the number of types, and the types are quadratic in the number of roles, so the width is quartic in the roles. The class kingdom's 322 words is already 7.7× the shipped 1-D kingdom's; the dynamic version is 24× it.

That is the number that decides whether the third piece is affordable, and it is the number a reader should weigh before anything else in this document.

### 3.3 What the width buys

Nothing, by itself: a wider gene is a more expressive search space, not a better program. The value of the third piece is that it makes a *class of programs* reachable at all — the ones whose dimensions are computed from the data — and the cost of that is the search space.

---

## 4. What it does not buy, and what it risks

### 4.1 The static relations are lost exactly where the dimension is dynamic

**Designed.** A class with a dynamic axis has no static relation on that axis. `mat.dyn.m × mat.dyn.p` is illegal statically, even though the two matrices might have equal inner dimensions at run time — that is the 1-D kingdom's `map2 (+)` case, and it is *not* expressible here. The dynamic class set buys expressiveness of *shapes*, at the cost of expressiveness of *relations between dynamic shapes*. The 1-D kingdom keeps the latter; the class kingdom keeps the former. **Neither replaces the other**, which is why both stay.

### 4.2 The class set is coarser than Futhark's types

**Designed.** A literal size (`replicate 5 x`) is `vec.dyn` in the engine and `[5]f32` in Futhark. The engine's class says less than the compiler knows. That is sound, and it is a *loss of precision in the engine's favour of finiteness* — the same trade the class set already makes.

### 4.3 A violated dynamic obligation is a compiler crash, and the harness must know it

**Measured** (§1.3): Futhark 0.28.1's runtime size obligation fails with an internal compiler error. So a candidate whose dynamic shape obligation is violated at run time produces a **non-zero exit with an ICE on stderr**, not a diagnostic.

The harness must therefore distinguish three outcomes that all look like "the program did not produce an answer":

| outcome | what it is | what the engine should do |
|---|---|---|
| the program failed its size obligation | the candidate is wrong | **reject the candidate**, count it as a failure |
| the compiler did not converge | a known Futhark defect (§15.3a) | reject the candidate, as the 1-D kingdom's rescue already does |
| the program timed out | the candidate is slow | reject the candidate |

All three reject the candidate, so the *engine* is safe either way. What must not happen is the ICE being reported as a crash of the engine, or the candidate being counted as a *success* with an empty result.

### 4.4 The dim sort's operators are a design choice with a cost

**Designed.** `length` and `replicate` are forced — they are how a dimension is read and how an array is built. But the *arithmetic* on dimensions (`length xs + 1`, `2 * length xs`) is a choice: a dimension's class is a role only if the expression is *syntactically* that role, so `length xs + 1` is `dim.dyn` — a value whose class is unknown even though Futhark would know it exactly.

This is the place where the design is least settled. The options are (a) keep only `length` and `replicate`, so a dimension is always a role or a literal; (b) allow arithmetic, and let the class degrade to `dim.dyn`; (c) allow arithmetic and track *expressions* over roles — which is the parametric design the assessment rejected. **(a) is the recommendation**, and the reason is that (b) buys very little (a dimension that is "the row count plus one" is not a role, so it is `dim.dyn`, and `dim.dyn` is what a literal already is) while costing a lot of precision in the class set.

---

## 5. What the third piece is *not*

* **It is not a repair step.** No candidate is ever fixed up; the classes are legal by construction.
* **It is not a rejection step.** A tree that expresses is a legal program, exactly as now.
* **It is not the shape-aware IR of §13's first wording.** §13 said "a separate shape-aware IR/evaluator for dynamic-shape tasks"; the measurements have shown that the *evaluator* already exists (Futhark's own type checker, and the shape kingdom's interpreter for the oracle), and that the IR need not be separate — it is the class machinery with dynamic axes.
* **It is not the population DAG.** §2.4: sharing is an optimisation, not a requirement.

---

## 6. The experiments that would decide it

The same discipline as §10.2 of the design assessment: three of these decide the design, and the rest record it.

| # | experiment | pass criterion |
|---|---|---|
| 1 | **legality by construction, at scale** | 10⁵ random genes over the dynamic class set express with zero `LayoutDefect`s |
| 2 | **Futhark accepts every rendered program** | every generated program compiles — the class set and Futhark's size rules agree |
| 3 | **the interpreter and Futhark agree, on the dynamic shapes** | element for element, including the degenerate and the violated-obligation cases |
| 4 | the dynamic classes are *usable*: a program that computes a dimension and uses it | a program that would not express in the class kingdom expresses here, and computes the same as an equivalent program written with the dimension inlined |
| 5 | the unrepresentable case | two matrices whose inner dimensions are equal only at run time do **not** express — the design's limit, shown rather than assumed |
| 6 | the width | the measured width at N = 3, against the model's 991 |
| 7 | the violated obligation | the harness rejects the candidate and the engine survives the ICE |

Experiments 1–3 are the ones that decide. Experiment 5 is the one that keeps the design honest: the dynamic class set does **not** subsume the 1-D kingdom, and a design that hid that would be hiding its own limit.

---

## 7. The verdict

**The third piece is the class machinery with dynamic axes, and it fits the architecture.** Everything that makes the architecture work — a finite alphabet, legality by construction, the unification offline, an untouched GEP core — survives, and Futhark's own size rules turn out to be *exactly* the rules the extended class set would state.

**The price is the gene width: 322 → 991 words at three roles, and quartic in the number of roles.** That is the number to weigh, and it is the reason this is a design and not a build: the third piece is worth building if the workload needs data-dependent dimensions *and* the search space can afford a 3× wider gene.

**The risk is Futhark's runtime size obligation, which in 0.28.1 is an internal compiler error.** The engine can live with it — a failed candidate is a failed candidate — but the harness has to be built to expect it.

**And the boundary is unchanged:** shapes that are not roles *and* not computable from the data — genuinely open-ended reshaping, a dimension read from a file's header, a shape that is only known to another program — are still not expressible, and the assessment's §14 verdict on them stands.
