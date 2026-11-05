# Section 7 F2 code review: master 6's renderer, pass 5 (`src/futhark_table.rs`, `src/futhark.rs`)

Reviewed 2026-11-05: fuller `4ddf1fba` (plan `docs/PLAN_futhark_kingdom.md` section 7, row F2, and proof 1 in 7.3).
Against the design `docs/DESIGN_futhark_types_and_properties.md` section 5 (the adapters, "Sizes are chosen, not
computed"), master 6's format `kingdoms/futhark/master/README.md` section 5.5, `CLAUDE.md` and the house rules. Read
only, no build, no run. Every line of the diff was read: `futhark_table.rs` (+700), `futhark/tests6.rs` (498),
`futhark.rs` (+10). The proof's printed counts were read from the agent's test logs. Line numbers are at `4ddf1fba`.

## Verdict

**Accept. The renderer reads the decoding and decides nothing again, the adapters match the design, the text is
deterministic, and proof 1's `futhark check` half is real. Eight findings. Finding 1 must be made before the work is
called finished: the two changed adapter texts are not written back into the README or the design. Finding 2 weakens
the "runs legally" claim for A7, and finding 3 makes the adapter share misleading. The rest are small or latent.**

What was checked and is right:

- **The genes are master 6's.** `random_programs` (`tests6.rs:49`) draws with `futhark_sig::random_gene` over the
  master 6 kingdom's own contract and `GeneLayout::from_program_size`, expresses with `typed_gep::express` and decodes
  with G3's decoder. phylu's P1 draw (`TypedSampling`, function share one half, root a function) is the same rule, so
  this is the start population's distribution. It is not master 5. Mutated and crossed genes are not drawn, which is
  fine: legality comes from the decoding, not from the draw.
- **`fit` and `clampix`** (`futhark.rs:59-67`) are the design's section 5 texts, character for character. `fit` pads
  with the element's constant (`0f32`, `0i32`) and truncates. A matrix's rows are padded with a zero row
  `replicate cols 0` (`zero`, `futhark_table.rs:967`), and its columns with `map (fit l z)`. `clampix` clamps into
  `[0, m)`. Every dimension is at least 1, so the clamp is in range.
- **`subst` and `subst.const`** render the decoder's annotation (`futhark_table.rs:1214-1226`): the own-type input, or
  the data type's constant. The decoder has already replaced the node's properties with its substitute's, so the
  parent sees what renders. `hist` takes its constant, as the G3 review asked.
- **No needless adapters from spelling.** `PropertyExpr::min` sorts and dedups its arguments (`geneframe.rs:778`), and
  every relation compares normal forms. So `min(k,n)` and `min(n,k)` are one text and one value. A known-length
  `fit.pair` fits only the operand whose length is not already the target. The high adapter share comes from genuine
  mismatches: a random gene picks `n`, `k` or `d` uniformly; constants have length 1; and the root's length is drawn
  freely (`fit.root` is 704 of 1,000 histogram genes, and 1,571 dimensions over 1,000 A7 genes). It is not the
  renderer's doing. Risk 6 is still real for A7 (3.3 % adapter-free) and k-means (7.8 %), and the lead should read it.
- **The prelude holds only what renders.** `uses` is set only by a rendered `fit_to`, `clampix` or `matmul` spelling.
  A substituted subtree's children do not render, so they add nothing.
- **Determinism.** Everything is `BTreeMap` or `BTreeSet`, the ADF sizes are sorted by the root environment's order,
  and a two-process test (FNV over 1,400 texts) proves it.
- **Binder names** are master 5's `var_name` from class and ordinal. The nested-binder test also checks the
  interpreter against `c`, bit for bit.
- **The two departures** (an existential `ad_l` for two known lengths, and the `?` state's size binding) are measured
  against Futhark 0.28.1. Both are explained in the code (`futhark_table.rs` section comment, `bind`). The new
  known-length form is the one the design's section 6 histogram example already writes (`fit (i64.min k n) 0 ...`).

## Findings

1. **The changed adapter texts are not written back** (README `kingdoms/futhark/master/README.md:220-221`, design
   `docs/DESIGN_futhark_types_and_properties.md:128`). The README's `fit.pair` row still prescribes
   `let ad_a .. let ad_l = i64.min (length ad_a) (length ad_b)` for every pair. Its `fit.state` row still prescribes
   `let ad_s = .. let ad_m = length ad_s`. The design's `map2` row still reads `let l = i64.min (length xs) ..`.
   Futhark rejects the first two. **Failure:** F3's interpreter, F4's reader (`read(render(t)) = t`) and F5's bridge
   are written from the README, so they would expect texts the renderer never writes. A reader built from the README
   would also fail on every adapted `map2`. **Fix:** in README 5.5, `fit.pair` with two known lengths becomes
   `fit <min spelled> 0{E} {k}` on each operand whose length differs, and the `let`/`ad_l` form stays for `?` only.
   `fit.state` with a `?` state becomes `let [ad_m] (ad_s: [ad_m]{E}) = {1} in`. Update the design's `map2` row the
   same way. Cite the two Futhark messages and the date.

2. **A7's run cases are square, so the run proves nothing beyond `futhark check` for A7**
   (`tests6.rs:159`, `run_cases`). The smallest case is `held_n1_m1_p1_s5` and the smallest training case is
   `train_n8_m8_p8_s1`. In both, `n = m = p`. Every `fit` and `fit.inner` (1,532 + 811 + 1,571 over 1,000 genes) then
   copies without padding or truncating. **Failure:** an adapter fitted to the wrong one of `n`, `m`, `p` in a place
   where both texts type-check (an existential through a `?` operand, an ADF parameter) runs legally here. It would
   fail at run time on rectangular sizes. Histogram has the same weakness more mildly: both of its cases have `k = 1`.
   **Fix:** pick the run cases by distinct sizes, not by bytes. For A7, add `held_n7_m5_p3_s4` (1,107 bytes). For
   histogram, add a case with `k > 1` (`test_k3_n31_s18` is 289 bytes, or `held_small_k7_n300_s14`). For k-means, the
   one case (`train_n200_k4`: `k < n < d`, all distinct) is enough for legality: Futhark checks every `fit`'s length
   statically, so what is left at run time is the index bounds, and distinct sizes are what catch a clamp to the wrong
   bound.

3. **The adapter share counts work-bound substitutions as adapters, and some of them under a `fit` name**
   (`futhark_table.rs:1214-1216`, `Program6::adapter_free` at :869). A substituted node pushes
   `reqs[s.requirement].adapter`. When the substitution happened because a heavy adapter would run at
   `trips >= depth`, G3 sets `s.requirement` to that adapter's requirement (`typed_environment.rs:562`). The renderer
   then logs the substitution as `fit.state`, `fit.body` or `fit.pair`. **Failure:** risk 6's guard reads a mix. For
   reduce, 31 of its 36 "adapters" are substitutions, and the `fit.*` counts for histogram and k-means include an
   unknown number of substituted nodes that render no `fit` at all. **Fix:** record a substitution as
   `subst:<cause>` (or a separate `substitutions` list in `Program6`), print the adapter-free and substitution-free
   shares apart, and copy the measured numbers into the plan's F2 status.

4. **What "runs legally" means is narrower than the plan's wording, and the test's comment drifts**
   (`tests6.rs:189-193`). A run is legal when the program exits with status 0 and its output's dimensions equal the
   stored answer's (k-means: the centres', `[k][d]`). Values are not checked; that is F3's job (proof 1's second half,
   bit for bit against the interpreter). The doc comment says "the view's smallest stored case", but the test runs
   two. **Fix:** say "exit 0 and the declared output shape, values not compared (F3)" in the comment and in the plan's
   F2 status, and correct "smallest stored case" to "two stored cases".

5. **The 1,000-gene check removes the entry point** (`tests6.rs:71`, `as_module`). It renames `entry main` to
   `def main` inside a module, so the rules Futhark applies only to entry points are not checked for 900 of the 1,000
   programs. **Failure:** low risk. The header is the same signature for every gene of a view, and the 100 compiled
   programs re-declare it as an entry (`entry_calling`). **Fix:** add one line in the doc comment saying that the
   entry's rules are covered by the 100 compiled programs.

6. **A test split case is used for development** (`tests6.rs:159`). `run_cases` sorts all cases by bytes, so
   pathfinder runs on `test_2x7_s18`, a TEST case. No fitness reads it, but the house rule keeps the test split for
   the exactness claim only. **Fix:** filter `split != Test` before choosing (pathfinder has `held_1x37_s7`, which is
   also rectangular).

7. **Two `?` adapters on one node would collide on the reserved names** (`futhark_table.rs:1064`, `bind`). `clampix`,
   `fit` and `fit.body` all bind `ad_a`/`ad_m` on a `?` side, and `fit.pair` binds `ad_a`. A node with two failing
   requirements that both meet a `?` returns "binds the reserved name twice": a bug report, not a program. The
   `clampix` known branch also spells the bound from the decode-time `o.against` (line 1105). A later adapter on the
   same node that fits that array shorter would leave a stale clamp, which is an out-of-bounds read. Neither can
   happen in today's master: no row gives `?`, and no row pairs `clampix` with another adapter on the indexed array.
   Only the synthetic `filter` fixture (`tests6.rs:339`) reaches the `?` branches. **Fix:** give each `let` a suffix
   from its requirement index (`ad_a3`), and read the clamp length from `ops[a]`'s current properties. Alternatively,
   add a table check (G3) that rejects such rows, so the case stays impossible by construction.

8. **The ADF rendering is tested only on adapter-free references** (`tests6.rs`, the ADF test). A hoisted subtree that
   holds an adapter, a substitution or a `?`-bound operand is not tested. Its result type `type_text` needs every
   property spelled, so a `?` result raises a bug. **Fix:** add one ADF test on an adapted random gene from the
   k-means view, compared with its plain program on `c`.

Jargon: the code comments use the plan's own names (pass 5, G3, F2a, Dv), which is fine in code. Errors say "our
software has a bug" through `bug6`, as the house rule asks. Nothing for Andrew's prose.
