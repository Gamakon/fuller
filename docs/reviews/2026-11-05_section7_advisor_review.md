# Advisor review: plan section 7 (data types, properties and scope), 2026-11-05

Reviewed: `docs/PLAN_futhark_kingdom.md` section 7 at `f49b9f0a`, against the agreed design
`docs/DESIGN_futhark_types_and_properties.md` at `f66891be`, reading the code each task touches (fuller at `f49b9f0a`,
phylu at `95bf4317`).

## Verdict

**Ready after nine changes; no change to the design.** The plan implements the design faithfully: the six tables,
arity from the Argument rows, the five passes, the GPU running pass 1 only, `fit` and `clampix`. Four changes matter
before any agent starts:

1. **How a size reaches a row is not decided.** The design gives `tabulate` its size as an argument (`out.len = in0`);
   the plan says nothing about it, and says no row produces `?`. A size computed by the gene does produce `?`, and can
   be 0 or negative, which Futhark rejects when the program runs (recommendation 2).
2. **Every task must end with both repositories green.** F1 removes the contexts that the interpreter (`binder_slot`,
   `src/futhark_sig.rs` line 1322) and the renderer still read until F2 and F3. And phylu builds against fuller by path,
   so a G2 commit breaks phylu's build until P1 (recommendation 4).
3. **phylu's Futhark variation operators are missing.** `src/futhark_eval/rings.rs` and `operators.rs` walk the Head,
   Tail and Dc spans. With no Dv ring, the ring walk never edits an index word (recommendation 6).
4. **The critical path runs through F1, not G3.** It is about 11.5 agent days, not 10.5. Splitting F1 brings it back
   to about 10.5 (recommendation 5).

The depth cap should stay, but described as what it is. It is not a type device: it is the **work bound**. The
master's own witnesses call it that (M5, S6 and P6 in `gen_tables.py`), and your GPU rule needs it (a runaway kernel
cannot be killed). The head length does not bound the work: N = 192 functions can nest 190 loops, n^190 trips
(recommendation 1).

## Recommendations

### 1. Keep the cap, name it the work bound, and write it in the design's own terms

Futhark has no depth limit, and the design removes depth from the types. But depth is still what bounds the work. The
head length bounds the number of nodes, not the nesting, and n trips per level make the work grow as n to the power of
the depth. Master 5 bounds it on purpose (`futhark_view.rs` line 535: "the `shallow` rule and the work bound"). It
covers every heavy row, not only loops and reductions: binders, `replicate`, `rowdiv`, `map2.*`, `scan.*`, `hist.*`,
`histrows.*` and the reductions (`gen_tables.py` S6 and M5). The design's own model holds this already: the
environment carries the trips above a node as a property, and a heavy row's requirement is trips < depth. This is the
same move the design makes for symbolic regression's transcendental depth (design section 8), with a substitution as
its adapter.

Plan text, 7.1, choice 1. Replace with:
> 1. **The work bound.** Master 5's `shallow` condition bounds the trips a program runs (witnesses M5, S6), not the
> number of types. Futhark itself has no depth limit, and the head length bounds the nodes but not their nesting. So
> the bound stays, as a property and a requirement: the environment (pass 2) carries `trips`, the binders above the
> node (a `let` adds none), and every heavy row of S6's list requires `trips < depth`. A failing node renders as its
> data type's constant, the substitution adapter of design section 8. Witness M5 is carried into master 6 against the
> new columns. Andrew agrees to this in F0, since it is the one rule the design text does not state.

### 2. Decide in F0 how a size enters a row, and allow `?` where it arises

Master 5 writes the size into the spelling: `tabulate {D}`, `replicate {D}`, `hist ... {D}`, `for i < {D}`, `last`
= `{D} - 1`, `{0} % {E}.i64 {D}`, `zeros_{D.tag}`. In the design (section 3.1), `tabulate` takes its size as input 0.
That input is an `i64`, and the gene may compute it (`i + 1`, an element converted). Its value is then `?`, and it may
be 0 or negative. `futhark check` accepts `tabulate (-1)`, but the program fails when it runs. And `clampix 0 i`
indexes an empty array. The design's guarantee that every dimension is at least 1 holds only for the entry's sizes.

Plan text, 7.1, the paragraph after the two choices. Replace "`?` is in the rule language from the start, though no
master 6 row produces it" with:
> A size is an `i64` argument (design 3.1, `tabulate`: out.len = in0). Its value property is the size's name when the
> argument is a size variable of the environment (`n`, `k`, `d` from the entry's `[n]`), and `?` otherwise. A `?`
> size renders as `i64.max 1 (...)`, so every length is at least 1 and `clampix` stays in range. F0 lists, row by row,
> which master 5 spellings read `{D}` and how each reads its size in master 6. `?` is therefore produced from the
> start, by every computed size.

### 3. Proof 1 must also run the programs, not only type-check them

`futhark check` checks types only. Indexing out of range, a negative size and the `fit`/`clampix` arithmetic are
caught only when a program runs, and 8 compiled genes a problem (F3) is too few to find them.

Plan text, 7.3 item 1. Append:
> All 1,000 genes a problem also run on F3's interpreter on one stored case with no run-time error (the interpreter
> checks bounds and sizes as Futhark does), and 100 of them compile on the `c` backend and run, N of N, equal to the
> interpreter bit for bit.

And in F3's CPU tests, replace "the first 8 random genes a problem compile and equal the interpreter bit for bit" with
the sentence above.

### 4. Master 6 sits beside master 5 until a final deletion task, and every fuller task builds phylu

The master is versioned per row and the views pin a version, so master 6 can land beside master 5. Each task then
keeps `cargo test` green. The replay checks can also compare old and new texts in one build, and 7.4's deletion
becomes one last task, not something done midway through F1.

Plan text, 7.2 preamble. Replace "Each task is one agent, commits only its own paths, and ends with ... clean in its
repository" with:
> Each task is one agent, commits only its own paths, and ends with `RUSTFLAGS="-D warnings" cargo test` and clippy
> clean in its own repository **and** `cargo test --no-run` in phylu (phylu builds against fuller by path). Master 6
> rows and views land beside master 5's; the master 5 path stays loadable until task D1.

Add a row to the table:
> | **D1** | fuller + phylu | 0.5 | the deletions of 7.4; `Ty::MAX_FUTHARK_DUALS` and the test that the `i32` + `f32` view does not load turned into the test that it loads | the files of 7.4 | the pixel family's and the deprecated 1-D kingdom's tests unchanged; both full suites | L1 |

### 5. Split F1 so it leaves the critical path, and fix the critical path in the text

Durations as planned: G1 (1), then G2 (2), then F1 (3) makes F1 finish at day 6, later than G3 (day 5). So F2 starts
at day 6, and the path is G1, G2, F1, F2, F4, P3, L1, about 11.5 days. The Python half of F1 needs only F0.

Plan text, 7.2, row F1. Replace with two rows:
> | **F1a** | fuller | 2 | master 6 rows and Binding rows; `sniff.py` writes the root environment with its properties; `view.py`; `gen_tables.py`'s witnesses (every data type has a constant; every requirement has an adapter; every rule names existing properties; M5 against the new columns) | `kingdoms/futhark/master/*`, `sniff.py`, `view.py`, `gen_tables.py` (table part) | `gen_tables.py --check` | F0 |
> | **F1b** | fuller | 1 | `futhark_view.rs` computes data types; `from_tables` reads the new columns | `src/futhark_view.rs`, `src/futhark_table.rs` (loader) | `view.py` and `futhark_view.rs` byte for byte; the `i32` + `f32` k-means view loads; the pixel family's tests unchanged | F1a, G2 |

Replace "the critical path G1, G2, G3, F2, F4, P3, L1 is about 10.5" with "the critical path G1, G2, G3, F2, F4, P3,
L1, D1 is about 11 (F1a and F1b finish by day 4.5)". Also drop F1's "the type counts of 7.3 item 4 measured": L1
measures them once.

Same-file overlaps the DAG must keep apart: F3 regenerates `costs.tsv` through `gen_tables.py`, which F5 edits too.
F3 and F5 overlap in time, so F3 hands the cost generator change to F5's file only after F5 has committed. The plan
should write "F3 after F5 for `gen_tables.py`". F1b and F2 both edit `futhark_table.rs`; they are already in sequence.

### 6. Add phylu's Futhark operators and the frame-table readers to P1 and P2

`FRAME_WORDS` going from 10 to 12 changes the frame table every kingdom's device code reads (`walk.wgsl`, shared by
symbolic regression, regex and WGSL; `typed_init.wgsl`, `typed_vary.wgsl`; host twins in `typed_walk.rs`,
`typed_decode.rs`, `typed_write_back.rs`). phylu's Futhark operators read the frame spans directly:
`src/futhark_eval/rings.rs` (the Head, Terminal and Constant rings) and `src/futhark_eval/operators.rs` (the spans
`[Head, Tail, Dc]`). Their tests (`rings/tests.rs`, `measure/tests_kmeans_seed.rs`) name sized types.

The design's claim holds: the device reads Argument (each symbol's ordered child types) and DataType (the frame of
each type), plus the constant pools it already reads. `typed_decode.wgsl` is symbolic regression's Phase B and is not
used by Futhark, which runs the walk-only stage (`walk_stage.rs`). But it reads the frame table at the new stride, so
it must be rebuilt and tested.

Plan text, 7.2, row P2, files. Append "`src/futhark_eval/rings.rs` (a Dv ring), `src/futhark_eval/operators.rs` (Dv
spans), their tests". CPU tests, append "the ring walk edits index words". Row P1: its tests on "1,000 genes a
problem" need the Futhark kingdom, so P1 comes after F1b, or it tests on G2's test kingdom and moves the
`futhark_typed.rs` handover to P3. Write "P1 after G2 (test kingdom); `futhark_typed.rs` in P3". Row L1: append "the
gated GPU tests of symbolic regression, regex and WGSL, since the frame table's stride changed".

### 7. One environment function, shared by the decoder, the reader, the renderer and the interpreter

The index word means "the (w mod c)-th variable of T in the environment". So the environment's order is part of what
a gene means. The reader (F4), the renderer's binder names (F2) and the interpreter's variable slots (F3) must all
read that order from G3's function and never rebuild it.

Binder names must also never shadow. Today's names are unique because the ordinal counts the same class already in
scope (`var_name`, `futhark_view.rs` line 657). If the new key (data type plus properties) gives two binders in one
environment the same name, the inner one hides the outer one in Futhark, while the decoder still lists both. The
interpreter and the compiled program would then disagree.

Plan text, 7.2, row G3, what. Append "the environment's order is one public function, the only one F2, F3 and F4
use". Row F2, CPU tests. Append "nested binders of the same data type and properties render distinct names, and the
interpreter and the `c` backend agree on them".

### 8. Determinism and the e-graph's soundness, stated as tests

Determinism: G3 decodes twice. Add, to G3 and F2, that the environment and the rendered text are the same across
two processes (no HashMap order; the `BTreeMap` rule of CLAUDE.md).

The e-graph: every rule in `master/rules.tsv` today is local (commutativity, associativity, projections, `map2.min a
a = a`). None moves a subtree across a binder, so one sort per data type is sound. Positional variable indices would
silently change meaning under a rule that moves a subtree into or out of a binder (fusion, hoisting).

Plan text, 7.5 risk 3. Replace the witness with:
> a `gen_tables.py` witness: no rule's right side places a pattern variable under a binder other than the one it was
> under on the left (a rule never changes a subtree's environment), and the bridge checks scope on write-back.

### 9. Remove the side measurements and name the orphan reader file

Risk 1's study of 20,000 edits a problem is a measurement for its own sake. Andrew forbids those mid-plan. Delete
"The share of point edits ... reported, not gated." Risk 4's adapter share is free: it is a counter printed by
proof 1, so keep it as that and run no separate study.

`src/futhark_reader/kinds.rs` (447 lines, untracked since 2026-11-03, not referenced by `futhark_reader.rs`) reads
`i32` constructs into the `f32` k-means view. The design makes this unnecessary, since the `i32` + `f32` view loads.
Plan text, 7.2, row F4. Append "first, the lead deletes the untracked `src/futhark_reader/kinds.rs` or commits it;
F4 does not start with an unowned file in its module".

The repository merge (6.2) moves files and redoes none of this work, provided section 7 is finished before the merge
starts. One name to avoid: phylu already has `src/evolve/typed_decode.rs` (symbolic regression's device decoder).
After the merge, two modules called `typed_decode` would sit in sibling crates. Plan text, 7.1 row "decoder" and row
G3: rename the new fuller module `src/typed_environment.rs`.
