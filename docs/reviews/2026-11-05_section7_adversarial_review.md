# Adversarial review: plan section 7 (data types, properties and scope), 2026-11-05

Reviewed: `docs/PLAN_futhark_kingdom.md` section 7 at `5cc9c96e` (with the advisor's review applied), against the
agreed design `docs/DESIGN_futhark_types_and_properties.md`, reading fuller at `5cc9c96e` and phylu at `95bf4317`.
Read only; nothing was built or run.

## Verdict

**Not ready. The design holds, but four of the plan's assumptions are false in the code, and each would stop a
task midway.**

1. **A size moved from the type into an `i64` argument breaks the work bound that protects the GPU.** The
   per-submit cap reads node executions from the sizes in the types. A computed size (`?`) gives it nothing to read.
2. **The variable index word is relative to its scope, and four places treat equal words as the same variable:**
   the e-graph's hash-consing, the saturated fold's shared parts, marriage's subtree hashes and the walk's gene keys.
   None of the planned tests can see this, because the program they produce is still legal.
3. **Three adapters, as the design spells them, produce programs Futhark rejects or programs that grow
   exponentially.** These are `fit` inside an array-valued lambda, the `map2` adapter's repeated operands, and `fit`
   on a loop state whose length is `?`.
4. **Proof 2 cannot run as written.** The frozen ledgers and the reference programs are stored as master-5 row names,
   not as Futhark text. The reader cannot read k-means today. No task converts the references.

Fix these in the plan, and in F0 for the size question, before G2 starts. The task sizes are about 60 % low:
roughly 35 agent days, not 22.

## Findings, by severity

### Critical

**C1. A computed size defeats the work cap, and the cap is the GPU's only guard against a kernel that will not
stop.**
- *Where:* plan 7.1, "A `?` size renders as `i64.max 1 (...)`"; fuller `src/futhark_sig.rs:1821-1866`
  (`work_parts`); phylu `src/futhark_eval/evaluator.rs:621` (`submit_work_of`, `CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT`,
  phylu's last commit).
- *Failure:* `work_parts` multiplies by `n[d]`, which it takes from the output class (`Op::Tab(d, ..)`,
  `Op::Loop { dim, .. }`, `Op::Hist(_, d, _)`, `class.dims()`). When a size is an `i64` subtree there is no `d`. A
  size such as `i64.max 1 (i64.i32 x)` taken from a data element can be 2^31. The cap then either cannot be computed
  or lets the program through. The trips < depth rule bounds nesting, not extent. The adapters (`fit` is a
  `tabulate`) are heavy rows too, and the plan does not count them in the work bound.
- *Fix:* decide in F0, as a decision for Andrew, between two options.
  - (a) Keep master 5's arity: a size is chosen by the function cell's **own** index word (Dv is one word per head
    and tail cell, so a function cell already has one) over the environment's size variables. There is never a `?`,
    the gene keeps the shape of master 5, and the reference programs reproduce directly.
  - (b) Keep the design's size argument, add a `gen_tables.py` witness that every `i64` a size slot can receive is
    bounded by a size in scope, render `?` as `i64.max 1 (i64.min S (...))` with S that bound, and make `work_parts`
    charge S and charge every adapter.

  Add `work_parts` and `submit_work_of` to F3's files under either option.

**C2. Positional variables are treated as the same variable wherever subtrees are compared by their words.**
- *Where:* plan 7.1 e-graph row ("a variable constructor `(T, index)`", "the fold shares closed subtrees only"); plan
  risk 3; phylu `src/evolve/marriage.rs:206-229` (`subtree_hashes` hashes the symbol and `const_index` only) and
  `:363` (a pair of identical subtrees is excluded); fuller `src/saturated_fold.rs` (shared parts are e-classes
  reached from two individuals); `typed_gep.rs:825` (`walk_expressed` visits the symbol and the Dc word only, and the
  evaluator keys oversized genes with it).
- *Failure:* `(var.[]i32, 2)` means `is` when two `[]i32` variables are in scope and `s_a_k` inside a loop that adds
  a third. Each of the following goes wrong:
  - The e-graph hash-conses both into one e-class.
  - The fold turns that class into a shared part used in both scopes. That changes the program, which breaks the
    contract that values are bit-identical with or without the shared rows.
  - Marriage counts two different programs as one tree, so it rejects valid parents and bars valid pairs.
  - Two genes that differ only in an index word get the same key.

  In Futhark every variable, entry parameters included, is a `var.T`. So "closed subtrees only" leaves nothing to
  share but constants. F5's test ("the extracted variable is in scope where written") passes even when the variable
  is the wrong one.
- *Fix:* the e-graph's variable constructor, the fold's key, marriage's hashes and the walk's visit carry the
  **resolved variable** (its binder's identity: the depth at which it was bound, plus its ordinal), taken from G3's
  environment function. The bridge turns it back into an index word at write-back, from the destination's
  environment. Add the test: a shared part used under two different environments evaluates bit for bit as the
  unshared program. Add `marriage.rs` and `walk_expressed`'s callers to P2's files.

**C3. Making `var.T` "flagged as constants are" sends it to Dc, not Dv.**
- *Where:* plan 7.1 contract row; `typed_gep.rs:789` (`express` reads `gene[dc_off + q]` for every
  `is_constant` symbol), `:920` (`place` writes Dc), marriage `PairCheck` (counts constants against `d_T`); 34 calls
  of `is_constant(` in 18 files across both repositories.
- *Failure:* with the flag as written, a `var.T` takes a Dc word, runs Dc out, and is counted against the constant
  pool's length.
- *Fix:* a separate `is_variable` flag. G2 lists and audits every call of `is_constant` in both repositories.

### High

**H1. Several adapters as the design spells them are rejected by Futhark, or grow the text exponentially.**
- *Where:* design section 5's table; plan F2.
- *Failures:*
  - (a) `tabulate n (\i -> fit (i64.min (length xs) (length ys)) 0 ...)`, or any array-valued body whose length
    names a variable bound by the lambda (`tabrows`, M0063; the value rows of `histrows`). The result size escapes
    the lambda, and Futhark rejects that statically (an irregular array).
  - (b) `let l = i64.min (length xs) (length ys) in map2 f (fit l z xs) (fit l z ys)` writes `xs` and `ys` out twice
    each. Nested adapters double the text at every level, so ten levels give about 1,000 copies. `futhark check` and
    the compile then hit their time limits.
  - (c) A loop state whose length is `?`: `loop (s: []i32) = ... do fit (length s) 0 body` gives the body the type
    `[length s]`, which is not the parameter's size variable.
  - (d) A `{0.len}` placeholder (planned for `next`, the `replicate {D}` of `histrows`) on a `?` property can only be
    printed as `length ({0})`, which copies the subtree again.
- *Fix:*
  - Every adapter, and every placeholder whose property is `?`, binds its operands with `let`, under names reserved
    for adapters (never a gene binder's name).
  - A Requirement on every array-valued lambda body: its length is a size from outside the lambda. When it fails,
    the body gets `fit` to that size, and the size is taken as a row argument or an index word (C1).
  - A `?` loop state is bound first (`let s0 = init let m = length s0 in loop (s: [m]t) = fit m z s0 ...`).
  - F2's fixtures cover each adapter inside a `tabulate` lambda, inside a loop body, and nested three deep, not one
    fixture a row.

**H2. Literal indices make reference texts change.**
- *Where:* `kingdoms/futhark/sig/kmeans/pool/seed.fut` (`(centres)[(0i64)]`), the pathfinder reference
  (`row.rows.cols walls zero`), master rows M0021 `zero`, M0022 `last`, M0019 `next`; design 4 pass 4 ("comparing
  the expressions").
- *Failure:* a constant `0i64` has no bound that equals any length, so equality fails, and the reference renders as
  `centres[clampix (length centres) 0i64]`. Proof 2 then fails on pathfinder and k-means.
- *Fix:* the property language gets a bound kind "valid for every length of at least 1" (for `0` and for
  `{size} - 1`), and an index Requirement is "bound ≤ len", not "bound = len". F0 writes this.

**H3. Proof 2's replay compares things that cannot match.**
- *Where:* plan 7.3 item 2; phylu `src/futhark_eval/evaluator/tests.rs:250-256` (`generation_zero` redraws generation
  0 from `GeneRng::new(1)` and asserts `sexp(...) == row.text`), `:262` (`gene_of` builds a tree from the ledger's
  text by row name); `histogram_r_seed1_ledger.tsv` (the text column is `(hist.add.k (hist.min.k zeros (loop.k ...`,
  master-5 row names, not Futhark).
- *Failure:* under master 6 the layout differs, so the redraw gives other genes. The row names `hist.add.k` and
  `loop.k` do not exist. "Matched on the text column" matches nothing.
- *Fix:* the replay renders each ledger sexp to Futhark under master 5 (which stays loadable until D1), reads that
  text with the master-6 reader, and compares the rendered Futhark texts and their quality inputs. Generation 0 comes
  from the ledger's texts, not from the RNG. Name this change in P3, or in a task of its own.

**H4. No task converts the reference programs, and the reader cannot read k-means.**
- *Where:* `src/futhark_sig.rs:130` (`reference()`), `:276-310` (`kmeans_seed`, `KMEANS_TEXTBOOK`, planted
  slowdowns: sexps of master-5 names); `kingdoms/futhark/sig/kmeans/pool/manifest.tsv`
  (`pool.fut_read_by_reader 0`: "the spelling of sel@f32:c does not parse as a pattern: float literal");
  `src/futhark_reader.rs:10` ("every `let` ... is inlined").
- *Failure:* F2's byte-for-byte test needs master-6 reference trees. These come only from hand conversion (no task
  owns it) or from the reader, which is F4, after F2. The reader reads 0 k-means programs today. It inlines every
  `let`, so `read(render(t)) = t` fails for every tree with the kernel-boundary `let`. P3 sends the start population
  through the reader, so it stops on k-means.
- *Fix:* a task F2a, before F2, that converts every reference sexp (four benchmarks, A7's three, k-means's three)
  and checks each against its master-5 rendering. F4 also fixes float-literal patterns and reads `let` as the `let`
  row. F4's size goes to at least 3 days.

**H5. F5 edits the wrong files for the problems that matter.**
- *Where:* plan F5 files; `src/futhark_sig.rs:2604-2608` (a signature view's e-graph is the view's own);
  `src/futhark_view.rs:1443` ("One sort per dual: the sort IS the type (class and context)").
- *Failure:* `kingdoms/futhark/egraph.*` and `futhark_saturation.rs` are A7's matrix e-graph only. Histogram,
  pathfinder and k-means (the 149 s e-graph) get theirs from `futhark_view.rs` (F1b's file) and `futhark_sig.rs`'s
  `Bridge`, `import`, `to_tree` and `FoldEgraph` (F3's file). The plan did not see these two overlaps.
- *Fix:* add `futhark_view.rs` (the e-graph part) and `futhark_sig.rs` (`Bridge` onward) to F5. Sequence F5 after
  F1b and before F3.

**H6. The interpreter and the cost model read sizes from the type throughout, and their callers are in phylu.**
- *Where:* `futhark_sig.rs:1308-1326` (variable slots from `Ctx::of_dual`), `:1359-1437` (`class_dim(out)` for
  `next`, `clamp`, `last`, `tab`, `replicate`, `zeros`, `hist`, `loop`), `:1582-1614` (`env.sizes[d]`);
  `futhark_cost.rs:226` and `costs.tsv` (a cost keyed `map_rows@mat.n.n`, by sized row name); phylu calls
  `Interp::new` in 14 places (`evaluator.rs:1949`, `batch/tests.rs`, `measure/tests*.rs`, `kmeans_converged`).
- *Failure:* every `Op` that carries a size changes. `eval` needs the decoder's annotations, so its signature
  changes, and phylu's 14 callers stop compiling. That breaks the rule that phylu must still build after every task.
  A cost table keyed by sized row name cannot price a node whose sizes are properties.
- *Fix:* F3 owns the phylu callers as well, or keeps `eval(tree, inputs)` and decodes inside it. `costs.tsv` becomes
  per-row formulas over argument properties, with adapter costs. F3 goes to about 4 days.

### Medium

**M1. G2 breaks phylu's build on its first commit.** `Domain` gains `Dv`, and phylu matches on it exhaustively
(`src/futhark_eval/rings/tests.rs:140`). `Tree` is built as a struct literal 629 times in 58 files across the two
repositories. If the index word goes into `Tree`, all of them change.
- *Fix:* G2 carries the phylu arms. The index word is stored in a field with a constructor, or in `const_index`
  with `is_variable` (C3). G2's size goes to 3 or 4 days.

**M2. Dv as a domain of its own separates an index word from its cell.** P2 keeps "row operators inside Dv spans",
as Dc does. Inversion, IS/RIS transposition and one- and two-point crossover then move a `var.T` without its word,
so a heritable choice of variable becomes a random one.
- *Fix:* Dv is a lane parallel to head and tail. Every row operator applies the same permutation or cut to a cell
  and to its Dv word. Test: after an operator, every `var.T` cell keeps its word.

**M3. A7's experiment changes, and the plan does not say so.** Under data types all matrices are `[][]f32`, so a
random A7 gene becomes a program padded with `fit`. That is the search space of a different experiment from the one
the shape-typed paper and `docs/measurements/2026-10-27_futhark_shapes_A7_learning.md` report.
- *Fix:* state that A7's published results are reproduced at their own commit. A7's inner-dimension requirements
  (`matmul`'s inner sizes, `map2_rows`, the value columns of `histrows`) need a per-dimension adapter,
  `map (fit l z)`, which design section 5 does not list.

**M4. Committed fixtures that name sized types belong to no task.** 48 files use master-5 names. Among them:
- the pool tables (`sig/*/pool/*.tsv`), and `tests/fixtures/futhark_pool_values/*.tsv` (about 1,500 lines);
- `futhark_sig/tests_micro.rs`, `tests_kmeans.rs`, `tests_view.rs`;
- phylu's ledgers, `kmeans_p_seed1_test_case.tsv`, `batch/tests.rs`, `measure/tests.rs`.

D1 deletes master 5 in 0.5 days, and every one of these then fails.
- *Fix:* each task converts the fixtures of its own module; D1 deletes only. D1 goes to 2 days.

**M5. The work bound's substitution is decided in four places.** A heavy row at trips ≥ depth renders as a
constant. The interpreter, `work_parts`, the reader and the e-graph must all make the same substitution, and
`read(render(t)) = t` cannot hold for such a `t`.
- *Fix:* the substitution is an annotation of G3, read by all four. The round-trip test runs on canonical trees
  (substituted subtrees removed, index words reduced mod c).

### Low

- **L1.** The replay scripts named in proof 2 live in `/private/tmp/claude-501/fbench/plan5a/`, which a reboot can
  clear. Commit them.
- **L2.** `mod` renders `{0} % i32.i64 {D}`. A clamped `?` size above 2^31 wraps to 0. Under C1 (b), keep S
  below 2^31.
- **L3.** Without `top`, the "each let class once" of M0085 is gone too. M5's `top` clause in `gen_tables.py:2245`
  must be rewritten, not only carried over.

## Task sizes, as read from the code

These figures are in agent days.

| task | plan | judged | why |
|---|---:|---:|---|
| G2 | 2 | 3.5 | `is_variable` audit (34 calls), `Tree` and `Domain` across both repositories (M1, C3) |
| F2 | 2 | 3.5 | the adapters' `let` binding, the lambda requirement, the literal-index bound (H1, H2) |
| F2a | 0 | 1.5 | the reference sexps converted (H4) |
| F3 | 2 | 4 | the interpreter's `Op`s, `work_parts`, the cost formulas, phylu's callers (C1, H6) |
| F4 | 1.5 | 3 | float literals, `let`, the canonical round trip (H4, M5) |
| F5 | 2 | 3 | the view's e-graph and `futhark_sig`'s bridge, the resolved variable (C2, H5) |
| P2 | 2 | 3 | the Dv lane, marriage's hashes (C2, M2) |
| D1 | 0.5 | 2 | the fixtures (M4) |

About 35 agent days in all; the critical path is about 17.
