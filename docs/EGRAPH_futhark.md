# Fuller's e-graph for the Futhark kingdom

Date: 2026-10-26. Status: Phase 1–3 of the e-graph specification, built and
measured. Phases 4–6 are deliberately not started; §6 says why.

Scope: `kingdoms/futhark_true`, `src/futhark_egraph.rs`, and the existing
Futhark execution harness. The Karva gene stays the canonical representation:
the e-graph is a *tool that finds equivalent programs*, not a new genome.

---

## 1. What was built

`src/futhark_egraph.rs` puts the Futhark kingdom inside Fuller's equality
saturation. The design decision that makes it work is that **one egglog sort
is one kingdom type**:

| sort | kingdom type | what it holds |
|---|---|---|
| `FR` | `f32.real` | a literal, or a reduction over an array |
| `FA` | `arr.f32` | an ordinary array: the input, a map, a map2, a reverse |
| `FS` | `scn.f32` | a scan result (Futhark's `acc`) |
| `FF` | `fn.f32_f32` | a function value |

The sort *is* the type, so egglog itself refuses to build an ill-typed
program — there is no second type checker to keep in sync — and the
accumulator distinction (`arr.f32` versus `scn.f32`) is carried structurally
rather than by a side table. A `map` over an ordinary array (`AMap`) and a
`map` over a scan result (`SMap`) are different constructors over different
sorts, exactly as Futhark's `acc` makes them different.

The 38 kingdom rows map one-for-one onto 38 constructors, plus two that exist
only in the e-graph:

* `FId` — the identity function `\x -> x`. Futhark has the expression; the
  kingdom has no row for it (the kingdom's `fn.f32_f32` rows are the
  elementary functions and the constant function).
* `FCompose f g` — the composition `f . g`. Futhark has no first-class
  functions, so it renders by *inlining* the two lambdas. The result is a
  valid Futhark program but not a kingdom row, so a candidate that uses it is
  **compilation-only**: there is no gene that denotes it.

Both are sound: `FId` is the identity on reals and `FCompose` is pointwise
composition.

The module offers:

* `import(tree, kingdom)` — an expressed Futhark tree as an egglog term.
* `render(termdag, id, kingdom)` — a term as Futhark source, in the kingdom's
  own spelling, so a candidate renders exactly as `FutharkKingdom::render`
  would have rendered the equivalent tree.
* `to_tree(termdag, id, kingdom)` — a term as an expressed tree, or `None`
  when the term uses an e-graph-only constructor or a literal the kingdom
  cannot spell. This is the **representability** boundary: `Some` means the
  candidate can be written back as a gene, `None` means it is compilation-only.
* `saturate(term, iters)` — a bounded saturation.
* `candidates(term, kingdom, k, iters)` — the equivalent programs, ranked.
* `saturate_tree(tree, kingdom, k, iters)` — all of it at once, with the
  e-graph's size and how long the saturation took.

### Why the candidates are enumerated, not extracted

Egglog's stock `extract_variants` varies the *root's* e-nodes and resolves
every child to its cheapest form. That hides exactly the alternatives the
rescue needs: the identity scan's expansion is two levels down. The vendored
`hff_extract` enumerates the members of each e-class recursively, so a
candidate may differ from the original anywhere in the tree. The module uses
that, with egglog's tree-additive cost (the number of e-nodes) as the score.

Determinism is by construction: the candidates are sorted by `(cost,
rendered source)`, never by egglog's internal iteration order, and the
extraction is a pure function of `(term, k, iters)`.

---

## 2. The rewrites, and the evidence for each

Every rule was checked **against Futhark 0.28.1 itself**: a program computes
both sides of the rewrite and the driver compares the raw bytes, over inputs
that include negatives, zeros, a one-element array, and `1e8`-scale values.
All eleven checks are **bit-identical**.

| rule | what it says | evidence |
|---|---|---|
| `AMap (FId) x → x` | `map (\x -> x) xs = xs` | bit-identical over an ordinary array |
| `SMap (FId) x → x` | the same over a scan result | bit-identical over a scan result, and under a `reduce` |
| `SScanIdentity → SMap (FId) (SScanAdd (AInput))` | the identity scan *is* `map id (scan (+) 0.0 xs)` | the terminal's own definition; lets the rule above fire on it |
| `ARev (ARev x) → x` | `reverse` is an involution | bit-identical |
| `ARev (AMap f x) → AMap f (ARev x)` | `map` commutes with `reverse` | bit-identical |
| `ARevScan (SMap f s) → AMap f (ARevScan s)` | the same for scan results | bit-identical |
| `AMap f (AMap g x) → AMap (FCompose f g) x` | pointwise fusion | bit-identical, including under a `reduce` over an `acc` |
| `SMap f (SMap g s) → SMap (FCompose f g) s` | the same for scan results | bit-identical |

**Nothing reassociates floating-point arithmetic.** The kingdom has no scalar
arithmetic to reassociate, and `map2 (+) a b` is not interchangeable with
`map2 (+) b a` in general: the shorter array decides the length, and the two
operands need not have the same length. Every rule above is an identity of
*computations*, not a mathematical identity.

### The one thing that does not work: `map id` and fusion together

The fusion rules are sound, but they **diverge** when they share a saturation
with `map id`, and the reason is structural rather than accidental.

The `map id` rule puts `map id xs` and `xs` in the same e-class. So that class
holds an e-node `AMap (FId) x` whose *argument is that very class*. The
fusion rule's pattern `AMap f (AMap g y)` then matches that node — the
argument's class *contains* a map, so the pattern matches — and builds
`AMap (FCompose f f) y`, which is another map node in the same class. The
next round finds two of them, and four after that.

Measured on the identity scan, the e-graph grows

```
5 → 7 → 13 → 55 → 1357 e-nodes   over five iterations
```

and does not converge after six. Fusion on its own is stable over the same
rounds, so the rules are usable — just not alongside `map id`. The core
ruleset (`FUTHARK_RULESET`) therefore has no fusion, and the fusion rules
live in `FUTHARK_FUSION_RULESET`, which the saturation does not use by
default. `the_fusion_rules_diverge_alongside_map_id` is a test.

---

## 3. The rescue experiment

The specification's Phase 3, on the one program in the corpus that Futhark's
optimiser never finishes.

### 3.1 The program

Regenerated from its seed, not copied: `GeneRng(0x5EED_1234_ABCD_0001)`, root
`f32.real`, program size 4, gene index 3 of 12 — the fourth of the twelve
genes the sweep drew.

```
entry main (xs: []f32) : f32 =
  reduce (*) 1.0
    (map2 f32.max
      (map2 f32.min
        (reverse (xs))
        (map (\x -> x - (reduce (+) 0.0 (map (\x -> x) (scan (+) 0.0 xs)))) (xs)))
      (reverse (map (\x -> f32.round x)
        (map2 (*) (map (\x -> x) (scan (+) 0.0 xs)) (scan (*) 1.0 (xs))))))
```

### 3.2 What the e-graph found

The e-graph of that program holds the program's term, the `map id` expansion
of the identity scan, and the `map`/`reverse` commutation — 8 distinct
equivalent programs, the cheapest first.

### 3.3 The record

See `RESCUE_futhark.md` for the full table: the candidate hashes, the costs,
the e-graph size, the saturation time, each candidate's compile verdict and
compile time, and whether the compiled candidate agreed with the independent
interpreter.

### 3.4 The verdict

**The rescue did not succeed.** Every equivalent candidate the e-graph found
— including the ones that remove the `map id` wrapper over the scan, the
leading hypothesis — hits the same wall-clock limit as the original.

That is a real negative result, and it is *informative*: the non-convergence
is not caused by the `map id` wrapper, nor by the `map`/`reverse`
commutation. §3.5 says what it *is* caused by.

### 3.5 What the non-convergence actually needs

Measured against Futhark 0.28.1, with a watchdog on every compile
(`/tmp/futhark_bug/hang.py`, `hang2.py`):

| the program | compiles? |
|---|---|
| the whole program | **hangs** |
| without the inner `reduce` over the `acc` | 0.1 s |
| with the inner reduce over a plain array instead | 0.1 s |
| with the second operand not reversed | 0.1 s |
| with the second operand a plain reversed array | 0.1 s |
| everything else (each piece alone) | 0.1 s |

So the trigger is the **conjunction**: an inner reduction over an *accumulator*,
a reversed accumulator-derived operand, and an outer reduction. Take away any
one of the three and the compiler is fine.

### 3.6 A sound rewrite that does not rescue

The accumulator is the piece the e-graph cannot remove on its own — and the
question is whether it can be removed *soundly*. It can:

> `reduce (+) 0.0 s` (over an `acc`) and `reduce (+) 0.0 (reverse (reverse s))`
> (over the same values, as an ordinary array) are **bit-identical**.

Measured over 19 inputs including `1e8`-scale and `1e-20`-scale values, for
`reduce (+)`, `reduce (*)`, and for a reduction under a `map`. Futhark reduces
an accumulator and an array with the same association.

So "flatten the accumulator" is a true rewrite, and the e-graph can express
it: `(rewrite (RReduceAddS s) (RReduceAdd (ARev (ARevScan s))))` — the root's
sort is `FR` on both sides, so egglog accepts it even though the child's sort
changes.

**It still does not rescue.** The flattened program hangs at 180 s, exactly as
the original does. Futhark's optimiser sees through the double reverse.

### 3.7 The honest bottom line

The rescue failed. The e-graph machinery works — it finds the equivalent
programs, ranks them, and re-encodes the representable ones back into genes —
but **Futhark 0.28.1's optimiser does not converge on this program, and none
of the equalities that hold for it change that**.

The qualification in the specification matters here: "a successful rescue is
useful, but it does not automatically establish that the broader evolutionary
search will improve". The converse also holds. A failed rescue on *one* program
does not establish that equality saturation cannot help — but it does
establish that this particular program is not rescued by these particular
rules, and that the non-convergence is a compiler defect in a corner of
Futhark's `acc` handling that equality-preserving rewrites of the program
cannot reach.

**The next thing to try**, in order of expected value:

1. ~~**Build the Futhark `webgpu` branch and test the program against it.**~~
   **Done, and it still hangs.** The branch's own CI publishes a macOS arm64
   build as an artifact, so no build was needed: Futhark 0.28.1, upstream
   master's 0.29.0-prerelease and the branch's 0.29.0-prerelease all hang on
   the program, while a control program compiles in 0.1 s on all three. The
   optimiser's rules are unchanged in that window (the only changes under
   `src/Futhark/Optimise/` are a mechanical traversal refactor and one
   narrowing guard on an `Update` rule the program does not use). The
   measurements are in `RESCUE_futhark.md` §6.1.
2. **Report the program upstream.** It is a legal, well-defined program, and
   three compilers fail to finish optimising it. §10.6 of
   `ASSESSMENT_futhark_kingdom.md` has the reproducer.
3. **Go on with Phase 4 regardless.** See §6.

---

## 4. What is *not* in this change

* **No new Futhark symbols in the kingdom.** The kingdom is still 38 rows.
  `FId` and `FCompose` live in the e-graph only. §6 says what it would take
  to change that, and what it would cost.
* **No repair, substitution or rejection in genetic expression.** Every gene
  remains legal by construction.
* **No change to the GEP encoding, the layout, or the contract check.** The
  e-graph is downstream of `express`, and the re-encoding goes back through
  the kingdom's own `Tree`.
* **No GPU path.** The GPU route is Phase 6 and was not started.
* **No cost model beyond the node count.** Phase 4 is next, and it does not
  depend on the rescue.

---

## 5. Reproducing

```bash
# the e-graph's own tests
cargo test --features wgsl --lib futhark_egraph

# the rescue experiment, with the full record
cargo run --release --features wgsl --example futhark_rescue -- --limit 120 --candidates 40
```

The rescue example regenerates the pathological gene from its seed, so it
needs no stored fixture, and it prints the whole record: the gene, the
e-graph's size and saturation time, and every candidate's hash, cost,
compile verdict, compile time and runtime correctness.

---

## 6. Why Phase 4 is not blocked by the failed rescue

An earlier draft of this document argued that Phases 4–6 should wait for the
rescue, on the grounds that the cost model would only be ranking candidates
that all fail to compile. **That was wrong**, and the objection is right: the
rescue was *one* pathological program, and the e-graph's value for evolution
is a different question — *given a program that compiles, is there a faster
equivalent?* The two objectives share the machinery but not the question.

So Phase 4 (a Futhark-specific cost model) proceeds, and so do the parts of
Phase 5 that make a discovered improvement heritable. What the failed rescue
does tell us is narrower and worth stating precisely:

* the e-graph **cannot** fix a compiler that does not converge on a program,
  however the program is rewritten equivalently; and
* the extraction has to be **representability-aware** to be worth anything to
  evolution, because a candidate the kingdom cannot spell is not heritable.

The second point is already implemented. `Saturation::heritable()` is the set
of candidates the kingdom can spell as a gene, and the e-graph always holds a
representable equivalent of everything it finds: the fused `map (f . g) xs`
sits in the same e-class as the unfused `map f (map g xs)`, and the
spelled-out identity scan sits in the same class as the kingdom's
`scn.f32.identity` row. `a_compilation_only_candidate_has_a_native_sibling`
is the test.

What is *not* yet settled is the other half: the native sibling of a fused
candidate is the **slower** form (two traversals instead of one), so an
extraction that always prefers the representable candidate gives up the
performance win. Making the *fused* form heritable needs `FCompose` to
become a kingdom symbol, with an introduction policy that keeps it out of
random sampling. That is a decision about the kingdom, not about the
e-graph, and it is not taken here.

The machinery for Phases 4–6 is in place and tested: the candidates are
ranked and reproducible, `to_tree` draws the representability line, and the
re-encoding back into a gene goes through `typed_gep::write_back`, which is
the same write-back the fold and promotion operators use.
