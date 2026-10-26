# Review: `docs/ASSESSMENT_first_class_shapes.md` (codex, adversarial)

**Date:** 2026-10-26 · **Reviewer:** `codex exec` (codex-cli 0.151.0), read-only sandbox · **Target:** `docs/ASSESSMENT_first_class_shapes.md`

The review was run on the command line with the instruction to *verify the document's factual claims against the implementation* and then to *try to break its conclusions*. The raw output is reproduced below, followed by the disposition of every finding: what was accepted, what was corrected, and what was **refuted with evidence**.

---

## 1. Disposition summary

| # | finding | disposition |
|---|---|---|
| P1.1 | "every array operator is shape-preserving" — `reduce` changes rank | **accepted** — corrected in §2 and §9 |
| P1.2 | U2 should be "(parent symbol, argument index)" | **accepted** — corrected in §3 |
| P1.3 | the §4 `transpose` example is ill-typed for the stated root | **accepted** — corrected in §4 |
| P1.4 | the §5 lemma is too strong | **accepted** — narrowed to this design's own invariants in §5 |
| P1.5 | "option A degenerates into B" / "the same expressive power" is too broad | **accepted** — qualified in §0, §5, §12 |
| P1.6 | the shipped-kingdom widths are 44/45/45, not 40/42/42 | **REFUTED** — the real engine says 40/42/42 (§2 below) |
| P1.7 | the PoC table's frames and widths are wrong (11/307, 8/158) | **REFUTED** on frames (the review counted an output-only type as reachable) and on widths (consequent) — §2 below |
| P1.8 | the "3.6–6.8×" figure rests on the wrong widths | **partly accepted** — the figure stands on the correct widths; the *asymptotics* claim is now stated properly from the measured growth curve (§6.4) |
| P1.9 | "9 matrix classes means 9 matrix inputs" | **accepted** — corrected in §6.3 (a *terminal* is required, not an external input) |
| P1.10 | `arr.f32`/`scn.f32` is a kingdom provenance split, not a demonstrated Futhark nominal type | **accepted** — corrected in §6.1, with a measurement |
| P1.11 | "the datatype/ruleset grows linearly in class count" | **accepted** — corrected in §8 (it grows with the *rows*, which are cubic in the role count) |
| P1.12 | "the write-back is unaffected" | **accepted** — corrected in §8 (the core write-back is; the e-graph *bridge* is hard-coded per row) |
| P1.13 | "zero engine changes" is conditional on layout↔kingdom identity | **accepted** — qualified in §0 and §7, added as a risk |
| P1.14 | the Futhark transcript is not in `/tmp/shape` | **accepted** — the transcript is now captured in `kingdoms`-independent form at `/tmp/shape/futhark_shape_transcript.txt` and cited as a measurement (§2, Appendix A) |
| P1.15 | Appendix B's line citations are accurate | **accepted as correct** (and re-verified after edits) |
| P2.1–P2.10 | the adversarial objections | **all accepted** and folded into §6.3, §6.4, §8, §10 and §12 |

---

## 2. The two refuted findings, with evidence

Both refuted findings concerned the count table in §6.2, and both were settled by **running the engine itself**, not by re-deriving the formulas by hand.

**P1.6 — the shipped kingdom's widths.** The review asserted 44/45/45. The engine says otherwise:

```
$ cargo run --release --features wgsl --example futhark_true_emit -- --root f32.real --size 3 --count 1 --show 0
kingdom Futhark: 4 types, 38 rows, root f32.real, program size 3, gene width 40
$ ... --root arr.f32  →  gene width 42
$ ... --root scn.f32  →  gene width 42
```

`40 / 42 / 42` — the assessment's numbers. Hand-derivation agrees: with root `f32.real` and N = 3, the heads are `(real 3, arr 4, scn 4, fn 2)`, `L = 9`, `P = (3, 15, 15, 8)`, and `t_T = max(0, min(P_T, L + h_T) - h_T)` gives `t = (0, 9, 9, 6)`, `d = (3, 9, 9, 8)`, so the width is `6 + 13 + 13 + 8 = 40`. The review quoted the correct `P` values and then computed `t_T = P_T`, which is the bug: `t_T` is `min(P_T, L + h_T) - h_T`, not `P_T - h_T`.

**P1.7 — the PoC table's frame counts.** The review counted 11 frames (full) and 8 (minimal), against the script's 10 and 7; the difference is the `scalar` class. `KingdomContract::reachable` follows **argument** types only:

```rust
pub fn reachable(&self, root: Ty) -> Vec<Ty> {
    let mut seen = BTreeSet::from([root]);
    let mut queue = vec![root];
    while let Some(u) = queue.pop() {
        for f in self.functions_of(u) {
            for t in self.signature(*f) {          // arguments, not outputs
                if seen.insert(*t) { queue.push(t); }
            }
        }
    }
    ...
```

`scalar` is only ever an **output** (`reduce : (C) → scalar`), never an argument, so it is not reachable from a matrix root and gets no frame. The script implements exactly this rule. The widths follow from the frames and are correct as printed.

**Independent validation of the transcription.** The bound formulas were transcribed from `kingdom_contract.rs` into `/tmp/shape/poc_counts.py` and checked against two independent references:

1. the engine's own published two-type fixture (`L = 3`, `P = (3, 2)`, `t = (1, 1)`, `d = (3, 2)`, width 5) — reproduced exactly;
2. the real engine's gene width for the shipped Futhark kingdom at N = 3 — reproduced exactly for all three roots;
3. the single-type Karva-tail identity `t = d = h·(n_max − 1) + 1` — reproduced for h ∈ {1, 7, 32}.

---

## 3. The raw review

### PART 1 — FACTUAL ERRORS

1. **"Every array operator … is shape-preserving" and "no row changes a length" (§2). — Wrong.**
   `reduce_*` consumes an array/scan and returns `f32.real`; it changes rank and eliminates the length.
   Evidence: `kingdoms/futhark_true/functions.tsv:16`–`19`.
   Minimal fix: say that no current row creates a *competing array length*; `reduce` is shape-consuming, not shape-preserving.

2. **The U2 formulation says a slot type depends on its parent symbol "alone." — Wrong/overstated.**
   It also depends on the argument index; the root depends on `layout.root`, not any parent symbol.
   Evidence: `src/typed_gep.rs:561` initialises the root; `src/typed_gep.rs:597`–`599` uses `signature(symbol)[i]`.
   Minimal fix: "A non-root slot type is determined by `(parent symbol, argument index)`."

3. **The matrix example in §4 is internally ill-typed. — Wrong.**
   For a root `Matrix(n,p)`, `matmul` needs `Matrix(n,m), Matrix(m,p)`. A transpose producing `Matrix(n,p)` needs `Matrix(p,n)`, not the stated `Matrix(n,m) → Matrix(m,n)` unless extra equalities are assumed.
   Evidence: assessment lines 93–97.
   Minimal fix: use `transpose : Matrix(p,n) → Matrix(n,p)` in this root example.

4. **The lemma in §5 is too strong. — Overstated.**
   "No reading-frame mechanism whose cell alphabet depends on the walk path can preserve legality" does not follow. A redesigned, sequential decoder could enumerate a context-dependent domain after each prefix choice and still never reject or repair. What fails is this engine's fixed `GeneLayout::alphabet` model, which derives a domain from only `(layout, position)`.
   Evidence: `src/typed_gep.rs:220`–`230`; the design requires enumeration before *the draw*, not before the full walk: `nucleotable/docs/MultiTypedGEP.md:193`–`197`.
   Minimal fix: limit the lemma to "the current fixed per-domain alphabet representation."

5. **"Option A degenerates into option B" and "the same expressive power is reachable" — Unsupported/false as general claims.**
   A finite class table cannot express open-ended dimensions, data-derived reshapes, or compatibility relations that are only known at runtime. The assessment itself concedes this at §12.5, so the earlier equivalence claim is too broad.
   Minimal fix: say "the same expressive power for a fixed, declared finite interface."

6. **The shipped-kingdom count table (§6.2) is wrong.** — **REFUTED; see §2 above.** The engine reports 40/42/42.

7. **The PoC count table (§6.2) is wrong, and conflicts with §10.1.** — **REFUTED on frames and width; see §2 above.** The distinction the review asks for — "all kingdom types" vs "declared shape classes" vs "root-reachable frames" — **was adopted**.

8. **"Width is … not an explosion" / "3.6–6.8×" — unsupported because based on incorrect widths.**
   Minimal fix: report corrected numbers and state the operator-dependent asymptotics. — **Adopted**: the growth curve in §6.4 of the assessment now states the asymptotics (rows cubic in the role count, width quadratic), measured for 2, 3, 4, 5, 8 and 10 roles.

9. **"A 9-class matrix kingdom takes 9 matrix inputs" — Wrong.**
   `check()` requires a terminal for each globally demanded type, subject to the `tail_free_roots` exception; it does not require an external input terminal, nor one distinct input per type. `scn.f32.identity` is already a zero-argument derived terminal.
   Evidence: `src/kingdom_contract.rs:187`–`202`; `kingdoms/futhark_true/functions.tsv:38`.
   Minimal fix: "requires at least one terminal per demanded class; external-input count is a generator/interface choice."

10. **"`arr.f32` and `scn.f32` are two duals for one Futhark type, split because Futhark's `acc` vs `[]f32` distinction is nominal." — Unsupported and misleading.**
    The repository establishes a kingdom-level distinction and renders both as ordinary Futhark expressions; it does not establish a public Futhark nominal type distinction. The table calls both one-dimensional arrays, and the renderer directly substitutes scan expressions into `map`, `reverse`, and `reduce`.
    Evidence: `kingdoms/futhark_true/types.tsv:3`–`4`; `src/futhark_kingdom.rs:200`–`250`; `src/futhark_egraph.rs:390`–`428`.
    Minimal fix: describe `scn.f32` as a deliberate kingdom/e-graph provenance type, not as a demonstrated Futhark nominal type.

11. **"Datatype/ruleset grows linearly in class count" (§8). — Wrong for this design.**
    Constructors track specialised rows. `matmul` produces one row per consistent triple, i.e. cubic in a dense set of dimension roles, not linear.
    Evidence: `src/futhark_egraph.rs:88`–`141`, `147`–`205`.
    Minimal fix: "linear in generated rows; generated rows may be cubic or worse in role count."

12. **"The write-back is unaffected" — mechanically true, but operationally overstated.**
    `place` and `write_back` use only row IDs and `Ty`, but e-graph import/export is presently a fixed mapping over named constructors and a hard-coded renderer.
    Evidence: `src/typed_gep.rs:629`–`696`; `src/futhark_egraph.rs:298`–`319`, `439`–`470`.
    Minimal fix: say "core Karva write-back needs no change; the current Futhark e-graph bridge does."

13. **"Zero engine changes" — conditionally right, not unconditional.**
    `Ty::Futhark(u16)` carries only a table index, not a kingdom/interface identity; two generated interfaces can reuse `Futhark(0)` with different meanings. `homologous()` checks only root and head lengths.
    Evidence: `src/geneframe.rs:112`–`115`; `src/typed_gep.rs:174`–`178`, `247`–`255`.
    Minimal fix: qualify it.

14. **The compile-time/runtime Futhark claim is plausible but not verifiable from the supplied `/tmp/shape` evidence alone.**
    Minimal fix: cite a captured compiler transcript/versioned measurement. — **Adopted**: `/tmp/shape/futhark_shape_transcript.txt` now holds the transcript, with the compiler version.

15. **Appendix B's line citations are accurate for this checkout. — Right.**

### PART 2 — ADVERSARIAL FEEDBACK

1. **Static alphabets only remain static after freezing the entire interface universe.** Two problems with different shape-role tables can both use `Ty::Futhark(0)` with different meanings; `swap_range` checks only root/head lengths. → *Adopted:* §7 and §12.7.

2. **One terminal per class is insufficient for useful interfaces.** `zipWith A B` with two distinct `[n][m]` inputs cannot be expressed by one `input@mat.n.m` terminal. → *Adopted:* §6.3.

3. **The "interface-minimal" class set closes badly under ordinary composition.** One matmul yields `[n][p]`, transpose yields `[p][n]`, another legal matmul then needs `[p][m]` or `[m][n]`. → *Adopted:* §6.3 and §12.4 — the assessment now states that the class set must be closed under the operators' effects, and reports the closure size.

4. **Bounds can grow superlinearly before rows do.** → *Adopted:* §6.4 now publishes the measured growth for 2–10 roles, with `P_T`, `t_T` and `d_T` maxima.

5. **One sort per class preserves only the table's existing relations; it does not validate shape logic.** A generator that emits `matmul` with its operands swapped is faithfully preserved by egglog. → *Adopted:* §12.2 — the generator becomes the new trust boundary and needs an independently checked shape-relation model plus witness tests.

6. **Ruleset complexity is worse than "one rule per class."** → *Adopted:* §8.

7. **The current e-graph round trip is not generic.** `constructor_of`, `row_of`, `render`, `to_tree` lack mappings for generated rows. → *Adopted:* §8.

8. **"Legality by construction" has a generator-sized hole.** → *Adopted:* §12.2 and §13.

9. **Class specialisation loses programs the current 1-D kingdom can express.** Two external vectors of declared `[n]` and `[m]` but equal runtime sizes: `arr.f32` can express `map2`, and Futhark checks the condition at runtime; distinct static classes reject it outright. → *Adopted:* §12.5 — this is now an explicit scope decision, not a footnote.

10. **The recommendation is only right for a narrow product decision.** Batching, slicing, data-dependent dimensions, reshape, or a variable number of inputs either explode the table or silently exclude productive programs. → *Adopted:* §12.6 and §13.

---

## 4. What the review did not find

For the record, the review did not identify an error in:

- the **frame-uniformity analysis** (§3 of the assessment) — (U1) and (U2) as statements about the built machinery;
- the **core of the recommendation** — resolve the unification offline in the generator, make the class part of the type, change no core-engine code;
- the **appendix citations**;
- the claim that **legality by construction is preservable** under that recommendation — this is the review's central concession, and the objections it raises are about the *generator* and the *scope*, not about the mechanism.
