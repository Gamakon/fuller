# The compiler-rescue record

Date: 2026-10-26. The specification's Phase 3, on the one program in the
corpus that Futhark 0.28.1's optimiser never finishes.

**Verdict: the e-graph rescue failed; the compiler workaround works.** Every
equivalent program the e-graph found hits the same wall-clock limit as the
original — §5 says what the non-convergence actually needs, and why no
equality-preserving rewrite of this program can reach it. The program itself
*does* compile when the compiler's fusion loop is bounded (§7), so the engine
now retries it that way and says so in the verdict: `Verdict::BoundedFusion`,
not a "rescue". The e-graph's contribution to that outcome is **zero**.

The record is reproduced by

```bash
cargo run --release --features wgsl --example futhark_rescue -- --limit 120 --candidates 40
```

which regenerates the gene from its seed — no stored fixture — and prints
everything below.

---

## 1. The original program and its gene

| | |
|---|---|
| root | `f32.real` (a scalar reduction) |
| program size | 4 |
| gene width | 58 cells |
| gene index | 3 of 12 (the fourth gene the sweep drew) |
| seed | `0x5EED_1234_ABCD_0001` |
| generator | `GeneRng` (splitmix64), now in the library, so the sweep and the rescue draw the same genes |
| AST hash | `af0f5607d8ceff0e` (FNV-1a 64 of the rendered expression) |

The gene's cells:

```
[15,16,17,34,3,4,11,5,0,35,6,8,36,13,36,8,33,30,18,35,35,35,35,35,35,35,35,
 35,35,35,35,36,36,36,36,36,36,36,36,36,36,36,36,36,36,30,21,28,37,24,20,28,
 25,20,15,5,8,11]
```

The Futhark program:

```futhark
entry main (xs: []f32) : f32 =
  reduce (*) 1.0
    (map2 f32.max
      (map2 f32.min
        (reverse (xs))
        (map (\x -> x - (reduce (+) 0.0 (map (\x -> x) (scan (+) 0.0 xs)))) (xs)))
      (reverse (map (\x -> f32.round x)
        (map2 (*) (map (\x -> x) (scan (+) 0.0 xs)) (scan (*) 1.0 (xs))))))
```

The gene is **not** illegal and not a bad gene: it validates, expresses, and
renders, and the kingdom's own interpreter evaluates it (the sweep's
143-of-144 agreement includes it). The *compiler* is the thing that gives up.

## 2. The e-graph

| | |
|---|---|
| term | `(RReduceMul (AMap2Max (AMap2Min (ARev (AInput)) (AMap (FSubC (RReduceAddS (SScanIdentity))) (AInput))) (ARevScan (SMap (FRound) (SMap2Mul (SScanIdentity) (SScanMul (AInput)))))))` |
| root sort | `FR` (`f32.real`) |
| e-graph size | 19 e-nodes |
| saturation | 4.1 ms, 4 iterations |
| ruleset | `FUTHARK_RULESET` (the stable one) |
| candidates | 8 distinct equivalent programs |

The term shows the two places the kingdom's `scn.f32.identity` terminal
appears: as `SScanIdentity` (the terminal) and, after the rules fire, as the
bare `SScanAdd (AInput)` (`scan (+) 0.0 xs`).

## 3. The equivalence evidence

Each rule was checked **against Futhark 0.28.1 itself**: a program computes
both sides of the rewrite and the driver compares the raw bytes, over inputs
including negatives, zeros, a one-element array and `1e8`-scale values. All
eleven checks are bit-identical.

| candidate | the rewrite that produced it | bit-identical? |
|---|---|---|
| 1 | the original | — |
| 0 | `ARevScan (SMap f s) → AMap f (ARevScan s)` (the `map`/`reverse` commutation, on the second operand) | yes |
| 2, 3 | the `map id` expansion of the second `scn.f32.identity` → `scan (+) 0.0 (xs)`, with and without the commutation | yes |
| 4, 5 | the `map id` expansion of the *first* `scn.f32.identity` → `scan (+) 0.0 (xs)`, with and without the commutation | yes |
| 6, 7 | both expansions, with and without the commutation | yes |

Every candidate is also **representable**: `to_tree` gives a tree for all
eight, so every one of them could have been written back into a Karva gene
with `typed_gep::write_back`. That is not why they were tried; it is why the
failure is a *compiler* failure and not a representability failure.

## 4. The candidates, and what the compiler did

Every candidate was compiled with the harness's existing wall-clock limit
(120 s) and then, if it compiled, executed and compared against the
independent interpreter.

**The table below is the state before the rescue path existed.** The rescue
path (section 7) is what every "timed out" row now compiles through; the
table is kept because it is the measurement that motivated it.

| # | cost | hash | gene? | compiler | compile time | runtime |
|---|---|---|---|---|---|---|
| 0 | 17 | `a905f1bf22bd369c` | yes | **timed out** | 120.04 s | — |
| 1 | 17 | `af0f5607d8ceff0e` (the original) | yes | **timed out** | 120.04 s | — |
| 2 | 18 | `6f56f65c8753b2ce` | yes | **timed out** | 120.07 s | — |
| 3 | 18 | `717311db0b8417ac` | yes | **timed out** | 120.06 s | — |
| 4 | 18 | `5ecdda70a59bddf2` | yes | **timed out** | 120.08 s | — |
| 5 | 18 | `7f43ee77332e089c` | yes | **timed out** | 120.05 s | — |
| 6 | 19 | `c67549eb7c021140` | yes | **timed out** | 120.04 s | — |
| 7 | 19 | `f931d4c7cea2d4ce` | yes | **timed out** | 120.06 s | — |

The original, compiled on its own before the e-graph was built: **timed out
at 120.05 s**. The previous session killed it at 300 s with no output.

```
candidates    8, of which 0 compiled, 0 were refused, 8 timed out
correctness   0 compiled candidates ran; 0 disagreed with the interpreter
rescue        NONE: no equivalent candidate compiled and ran correctly
```

No candidate was *refused* — the compiler accepted all eight as legal
programs and then failed to finish optimising them.

## 5. Why: the trigger, measured

With a watchdog on every compile (`/tmp/futhark_bug/hang.py`, `hang2.py`):

| the program | compiles? |
|---|---|
| the whole program | **hangs** |
| without the inner `reduce` over the `acc` | 0.1 s |
| with the inner reduce over a plain array instead | 0.1 s |
| with the inner reduce over a plain `map` instead | 0.1 s |
| with the second operand not reversed | 0.1 s |
| with the second operand a plain reversed array | 0.1 s |
| each piece on its own | 0.1 s |

The trigger is the **conjunction**: an inner reduction over an *accumulator*,
a reversed accumulator-derived operand, and an outer reduction. Take away any
one of the three and the compiler is fine.

The `map id` removal — the leading rescue hypothesis — does not touch that
conjunction: the inner reduce is still over an accumulator, and the second
operand is still accumulator-derived and still reversed. That is why the
candidates above all fail the same way.

### A sound rewrite that does not rescue

The accumulator is the piece worth attacking, and it can be attacked
soundly:

> `reduce (+) 0.0 s` (over an `acc`) and `reduce (+) 0.0 (reverse (reverse s))`
> (over the same values, as an ordinary array) are **bit-identical**.

Measured over 19 inputs including `1e8`- and `1e-20`-scale values, for
`reduce (+)`, `reduce (*)`, and for a reduction under a `map`. Futhark
reduces an accumulator and an array with the same association.

The e-graph can express that rewrite — `(rewrite (RReduceAddS s) (RReduceAdd
(ARev (ARevScan s))))` — because the root's sort is `FR` on both sides even
though the child's sort changes from `FS` to `FA`.

**It still does not rescue.** The flattened program hangs at 180 s, exactly
as the original does: Futhark's optimiser sees through the double reverse.
The same is true of materialising the accumulator in the second operand
(`reverse (map f (reverse (reverse s)))`), which is likewise bit-identical
and likewise hangs at 180 s.

So the honest conclusion is the specification's fallback: **the e-graph
cannot rescue this program, and the reason is that the non-convergence is
not reachable by equalities that hold for the program.** It is a defect in
Futhark 0.28.1's optimiser, in a corner of its `acc` handling.

## 6. What to do next

### 6.1 The experimental branch: measured, and it still hangs

The Futhark `webgpu` branch — PR #2140 "Add WebGPU backend" in
`diku-dk/futhark`, head `webgpu` at `14e9ccbf` — **still hangs on the
program**. This is a measurement, not an inference, and it took no build:

* The branch's own CI (run `39795721627`, head `14e9ccbf`, 2026-10-22)
  uploads a macOS arm64 build as an artifact named
  `futhark-nightly-macos-arm64.tar.xz` (33.6 MB). Fetch it with
  `gh run download 39795721627 -R diku-dk/futhark -n futhark-nightly-macos-arm64`
  (or the artifact API). It is the branch's compiler, not master's: it
  reports `Futhark 0.29.0 (prerelease)` and its command list contains
  `webgpu`, which master does not have.
* The upstream **nightly** (master, 2026-10-11, the same 0.29.0-prerelease)
  hangs too, and the control program compiles in 0.1 s on both, so the
  binary is not the problem.

| compiler | the control program | the pathological program |
|---|---|---|
| Futhark 0.28.1 (the installed one) | 0.1 s | **hangs** (killed at 120 s and at 300 s) |
| Futhark 0.29.0-prerelease, upstream master 2026-10-11 | 0.1 s | **hangs** (120 s) |
| Futhark 0.29.0-prerelease, the `webgpu` branch at `14e9ccbf`, 2026-10-22 | 0.1 s | **hangs** (120 s) |

And the trigger is the *same* conjunction in the branch's compiler — the
three reduced programs all compile in 0.1 s there:

| the program | the branch's compiler |
|---|---|
| the whole program | **hangs** |
| without the inner `reduce` over the `acc` | 0.1 s |
| with the inner reduce over a plain array instead | 0.1 s |
| with the second operand not reversed | 0.1 s |

**The defect is still there.** The branch is 286 commits of *backend* work
(rebased onto master), and the optimiser's own rules are unchanged in the
window: between the nightly's tree and the branch head, the only changes
under `src/Futhark/Optimise/` are a mechanical `TraverseOpStms` →
`TraverseOpBodies` plumbing refactor, one *narrowing* guard added to an
`Update` rule the program does not use, and changes to
`InliningDeadFun.hs` and `ArrayCoalescing.hs` that mention nothing about
`acc`, scan, reduce or reverse.

So: not a branch-specific problem, and not a stale-compiler problem. It is
a live defect in the `acc` handling of Futhark's optimiser, present in
0.28.1, in master's 0.29.0-prerelease, and in the experimental branch.

### 6.2 What to do now

1. **Report it upstream.** It is a legal, well-defined program, and three
   different compilers — 0.28.1, master's 0.29.0-prerelease, and the
   experimental branch — all fail to finish optimising it. That is a
   strong, reproducible report. `ASSESSMENT_futhark_kingdom.md` §10.6 has
   the reproducer; the shape is "a reduction over an `acc`, a reversed
   accumulator-derived operand, and an outer reduction, together". **The
   report now names the pass**: it is the SOAC fusion, and the reason is
   that its fixpoint loop is unbounded (section 7).
2. ~~**Stop investing in the rescue.**~~ **Done, and the other way round:
   the rescue works.** The compiler's fusion loop is the whole problem, and
   the pass takes a bound — so the program is compiled by the same pipeline
   with the loop capped, in 0.15 s, and it computes the right value
   (section 7). The engine does this automatically now, so a candidate the
   compiler cannot optimise is still a candidate the engine can run.
3. **Carry on with the parts of the e-graph that do not depend on the
   rescue**: the cost model (Phase 4) and the heritable extraction
   (Phase 5). See `EGGRAPH_futhark.md` §6.

### 6.3 A note on getting the branch's compiler

Do not build it. The branch's CI already publishes a macOS arm64 build as
an artifact on every run (`futhark-nightly-macos-arm64.tar.xz`), and
`gh run download <run-id> -R diku-dk/futhark -n futhark-nightly-macos-arm64`
fetches it in one step. A from-source build needs GHC 9.14 plus a cold
cabal dependency tree — hours — for the same binary.

---

## 7. The rescue: the fusion loop, bounded

The blocker was never the program. It was **one pass in Futhark's optimiser**,
and Futhark's own source says so.

### 7.1 Which pass hangs

`futhark c -v` prints the name of each pass *before* it runs it
(`src/Futhark/Pipeline.hs:176` — `when (pipelineVerbose cfg) . logMsg $ "Running
pass: " <> …`, and the pass itself runs on the next line). On the
pathological program the trace stops at:

```
[  +  0.002] Type-checking internalised program
[  +  0.001] Running pass: simplify
…
[  +  0.000] Running pass: Fuse SOACs          ← the last line printed
```

so the hang is in **"Fuse SOACs"** — the pass that merges array operations
(`map`, `map2`, `scan`, `reduce`) into one another. A control program prints
the whole pass list and finishes.

### 7.2 Why it does not finish, and the bound that stops it

Futhark's fusion pass takes an **optional bound on its convergence loop**:

```haskell
-- | Pass definition with an optional bound on the number of
-- iterations of the fusion convergence loop.  'Just n' runs at
-- most @n@ iterations; 'Nothing' means no bound (iterate until convergence).
fuseSOACs :: Maybe Int -> Pass SOACS SOACS
```

(`src/Futhark/Optimise/Fusion.hs:762`, Futhark 0.28.1.) The pipeline calls it
with **no bound**:

```haskell
standardPipeline =
  passes
    [ simplifySOACS, inlineConservatively, simplifySOACS,
      inlineAggressively, simplifySOACS, performCSE True, simplifySOACS,
      fuseSOACs Nothing,            -- ← no bound: iterate until convergence
      performCSE True, simplifySOACS, removeDeadFunctions ]
```

(`src/Futhark/Passes.hs:64`, and again at `:79` in the AD pipeline.) On this
program the fusion graph keeps changing, so the loop never settles — and
because the loop is unbounded, the compiler runs for ever.

The bound is exposed by Futhark's own developer tool:

```
$ futhark dev --help
  -o[GAS]   --fuse[=GAS]   Perform higher-order optimisation, i.e., fusion.
                          (default: infinite, provide integer to limit)
```

### 7.3 The bounded fusion, measured

`futhark dev` runs the passes it is given, in the order they are given, so
the pipeline can be spelled out with the fusion loop capped. The pass list
below mirrors Futhark's own sequential pipeline (`standardPipeline` →
`seqPipeline` → `seqmemPipeline` in `src/Futhark/Passes.hs`) — which was
checked by running the hand-assembled pipeline and `futhark c` on the same
program and comparing the artefacts (`prog.c`, `prog.h`, `prog.json`) byte for
byte: identical.

```sh
futhark dev --simplify --inline-conservatively --simplify --inline-aggressively \
            --simplify --cse --simplify --fuse=10 --cse --simplify \
            --remove-dead-functions --first-order-transform --explicit-allocations \
            --cse --lift-allocations --short-circuit --cse --lower-allocations \
            --backend=c --executable prog.fut
```

| the pathological program | compiler | result |
|---|---|---|
| `futhark c prog.fut` | 0.28.1 | **hangs** (killed at 120 s, 180 s, 300 s) |
| the same, upstream master 0.29.0-prerelease | nightly | **hangs** (120 s) |
| the same, the `webgpu` branch at `14e9ccbf` | branch build | **hangs** (120 s) |
| **the same pipeline, `--fuse=10`** | 0.28.1 | **compiles in 0.12 s** (three runs) |

And it is not merely a program that compiles — it is a program that computes
the right thing. On the input `[1, 2, 3]` the program is
`reduce (*) 1.0 [36, 6, 1] = 216`, and both the bounded-fusion binary and
Futhark's own interpreter give `216`.

Fusion is an optimisation, so stopping the loop early leaves *fewer* fusions,
not wrong ones. That is the whole workaround: the program's meaning does not
depend on the fixpoint.

The scope is **one** program and **one** input. The point is that the
workaround exists and computes the right value, not that the compiler's hang
is characterised for every program.

### 7.4 What the engine does with it

`src/futhark_exec.rs` compiles every program the normal way first, and when
the compiler does not converge it **retries with the fusion loop bounded**,
reporting `Verdict::BoundedFusion` so a caller can tell which path produced
the binary (a program compiled that way may be less optimised, and that is
worth knowing when it is scored).

The workaround is now a **test**, not an experiment:
`futhark_exec::tests::a_program_the_compiler_cannot_compile_is_bounded_fusion`
asserts that the normal path times out, that the bounded-fusion path compiles,
and that the compiled program computes `216.0` on `[1, 2, 3]`.

And the experiment that was blocked now runs to the end:

```
== the record ==
original      compiled: no (the compiler did not converge within 15s)
candidates    8, of which 8 compiled, 0 were refused, 0 timed out
correctness   8 compiled candidates ran; 0 disagreed with the interpreter
rescue        an equivalent candidate compiles and runs correctly
```

All eight candidates compile through the rescue path in ~150 ms each, run,
and agree with the independent interpreter.

### 7.5 What is still owed upstream

The defect is real and three compilers have it, so it is still worth
reporting: the fusion pass's fixpoint loop should terminate on a legal
program, and the upstream project has already fixed a fixpoint bug of this
shape once — issue [#2444](https://github.com/diku-dk/futhark/issues/2444)
("we are missing some kind of fixed point iteration"), closed by #2445. The
report should carry: the reproducer (§1), the pass (`Fuse SOACs`), the
unbounded call site (`src/Futhark/Passes.hs`), and the fact that
`futhark dev --fuse=10` compiles it in 0.15 s.
