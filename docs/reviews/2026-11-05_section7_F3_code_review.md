# Code review: plan section 7 task F3 (master 6's interpreter, work and static cost)

Commits fuller `adc7ea80` (`src/futhark_sig/interp6.rs`, `Interp::new6`, `futhark_cost::program_cost6`,
`kingdoms/futhark/master/6/costs.tsv` with witnesses C1 to C6) and phylu `2113e9a4` (the doc comment on
`submit_work_of`). Read only: `git show` and the files, no build, no test run. 2026-11-05.

## Verdict

**Accept, with three fixes before the cost is used for selection or the cap in a run (findings 1, 3, 5).** The
interpreter is a faithful reading of the renderer: every check Futhark makes is made, and every adapter is applied as
rendered. Two things are not yet shown. The comparison with the `c` backend has a tolerance arm, which the house
rules forbid (finding 1). And the claim "an adapted gene costs more than its clean form" holds only where the adapter
is needless at the case's sizes, because the test sets every size to 4 (finding 3). The cost and the work cannot
price a `?` length that the renderer does render, and no test reaches that path (finding 5).

What was checked and holds:

- **Futhark's checks are made, never tolerated.** An index is checked against its array (`get`, `row`:
  `interp6.rs:876`, `979`). The two arrays of `map2` must have one length and one element (`917`). So must a
  histogram's indices and values (`935`), the rows of a histogram of rows (`948`), a division of rows (`964`), a
  product's inner sizes (`1003`) and `map2_rows` (`1035`). Every row of a `tabulate` of rows must have one length
  (`798`). On top of that, every node's value is checked against its decoded properties in `Run::at` (`589-619`):
  an index must be at least 0 and below its bound, a size must equal its value, an array must have its lengths. A
  histogram index out of range is dropped (`hist_index`), as Futhark's `reduce_by_index` does: that is Futhark's
  meaning, not a check skipped. A selection evaluates only the branch taken, with `c < 0` (`Num::is_negative` is
  `x < 0.0`, the spelling `if {0} < 0{E}`), so `-0.0` and NaN go the same way in both.
- **The adapters are interpreted as rendered.** `fit` is `FIT_HELPER` (`tabulate l (\i -> if i < m then xs[i] else
  z)`): the first `min(l, m)` elements, then zeros. Rows are padded with zero rows, and columns use `map (fit ..)`
  (`interp6.rs:225-250`, `futhark_table.rs:1103-1120`). `clampix` is `max 0 (min (m - 1) i)` (`674`,
  `CLAMPIX_HELPER`), applied last and measured on the indexed array as the other adapters left it, which is the
  renderer's order (`futhark_table.rs:1335`). `fit.pair` fits both operands to the smaller length. Fitting an
  operand that already has that length is the identity, so it does not matter that the renderer skips it. `fit.inner`
  uses the renderer's three cases. `fit.state` and `fit.body` run once a trip, on the body's value. The root's fit
  uses `fit.root`. An input substitution evaluates its substitute, and a constant one gives the data type's constant.
- **`work_parts` charges the adapters in `Work::nodes`**, the measure `submit_work_of` reads. A body adapter is
  charged once a trip and the root's fit once (`interp6.rs:437-474`). The charge is computed by the same
  `adapters6` decision that `program_cost6` uses.
- **Cropping a gene that is over the cap is legitimate.** phylu never runs such a gene (`prefetch` leaves it out and
  scores it worst), so the full case is not a path production takes. The cropped case keeps the stored case's own
  values (but see finding 7).
- **`costs.tsv` has one line for every master row, in order (C2), and every row is restated (C3).** A leaf costs
  nothing. The matrix and reduction rows are written out by reference, and every other row takes the elementwise
  default. There is one line for every adapter (C5). Every `fit` is charged `fitted.cells` and `clampix` 2 (C6).
- **The phylu comment is accurate.** It is conditional ("a master 6 interpreter ..."). Today phylu's evaluator
  still builds `Interp::new` (`evaluator.rs:2016`), so the cap counts adapters only once a P task switches it to
  `new6`.

## Findings

1. **The `c` test has a tolerance arm, and it passes silently** (`tests_interp6.rs:321`, `333`). An `f32` output
   that differs in its bits but is within `Exactness::F32SumBound { products: 2 }` is counted as `bounded`, and the
   test asserts only that `bad` is empty. CLAUDE.md requires the interpreter and the compiled programs to agree bit
   for bit on these paths, with no tolerance anywhere, and the plan's F3 row asks for the same. The measured run had
   `bounded` = 0, but the arm means a future drift (a reordered reduction, a fused `a*b+c`) would pass the test
   unseen. *Fix:* delete the arm, so any difference in bits that is not finding 2's fold goes into `bad`.

2. **The `0 / x` allowance is too loose in one way and incomplete in another, and the fault has no issue note**
   (`tests_interp6.rs:260-272`).
   - Too loose: the allowance accepts a zero for *any* NaN cell, whatever produced the NaN, as long as the program's
     text contains `" / "` anywhere. A NaN from `sqrt` of a negative number, or `inf - inf`, that `c` turned into 0
     would be counted as "the fold".
   - Incomplete: the same rule in Futhark's simplifier (a dividend known to be zero folds the division to that zero)
     also makes `0 / -3` equal `+0.0` where IEEE gives `-0.0`. A sign-of-zero difference fails both `same_value` and
     the allowance, so it will show up as a `bad` run that the comment does not explain.
   - It is a Futhark fault worth a note: the rule is unsound for IEEE floats (`x` zero, NaN or negative), and whether
     it fires depends on fusion. The repro is already in the test's comment. `docs/issues/` holds the other Futhark
     notes (for example `2026-11-03_futhark_inline_nonterminating_kmeans.md`). This one is recorded only in the plan
     and a test comment.
   - *Fix:* have the interpreter count the divisions it performs whose dividend is a zero (the event the fold
     changes), and allow the fold only for a run where that count is not zero, for NaN and for `-0.0`. Write
     `docs/issues/2026-11-05_futhark_zero_dividend_fold.md` with the four repros measured on 0.28.1, and report it
     upstream. Also measure whether `0 * x` is folded in the same way (then `0 * NaN` and `0 * inf` give 0) before
     the note claims either way.

3. **"An adapted gene costs more" is shown only where the adapter is needless** (`tests_interp6.rs:356`). The test
   sets every size name to 4, which makes every `fit` the identity. At a stored case's sizes, a `fit` to a SHORTER
   length lowers the cost of everything downstream. Take k-means' `map2.add zeros_n zeros_k` with n = 1000 and
   k = 10. `fit.pair` fits both operands to `min(n, k)` = 10, because `zeros_k`'s `Size(k)` is not structurally
   `Min(n, k)`, so both are charged. Its traffic is about n + k for the leaves, 2k for the fits, 2k reads and k
   written: about 1,060. The pair it is tested against, `map2.add zeros_n zeros_n`, has traffic of about 5,000. So the
   cost rewards truncation. Only quality stops it, and a reduction (`maximum`, a `hist` whose dropped tail holds no
   bucket) can keep its value on a case while it reads less. The cost model is honest here, because the program
   does read less, but the Fable guard (plan 7.5 risk 4, README section 10: "an elite with a needless adapter loses
   to its own clean form") holds only for needless adapters. *Fix:* state that scope in risk 4 and in README section
   10. Add a test at distinct sizes that pins the fact that a shortening `fit` costs less, so it is a known property
   and not a surprise. Rely on the planned adapter share per generation (plan 7.3, proof 5) to show whether
   selection keeps such adapters.

4. **A substitution costs nothing, so it ties its clean form, and the two measures disagree on it.**
   `program_cost6` returns at a substituted node (`futhark_cost.rs:730`), so both kinds of substitution cost 0.
   `work_parts` charges a constant substitution 1 but an input substitution 0 (`interp6.rs:448-451`, where
   `rendered6` has already followed the input). A gene that carries a substituted node renders the same program as
   its clean form and ties it on cost, so selection never removes it on cost. That contradicts the general claim
   "the cost charges every adapter". *Fix:* charge each substitution 1 in both measures (`subst` and `subst.const`
   traffic `1` in `adapter6_traffic` and witness C6), so the clean form wins the tie. Or, if the tie is wanted,
   exempt substitutions by name in README section 10 and in the plan.

5. **A `?` length makes the cost and the work fail, and no test reaches it** (`futhark_cost.rs:553`, `adapters6`
   `633-664`). `known6` turns `?` into "our software has a bug". `adapters6` writes `PropertyExpr::Unknown` for any
   fit target that cannot be spelled (a `fit.pair` whose minimum names an input's property, or a `fit` against a
   bound operand), and `cells6` then fails on it. The renderer supports exactly these programs (`futhark_table.rs:1189`,
   `1217-1230`, and F2's fixtures "a loop of `?` state length and a `map2` over two `?`-length operands"). If a
   random gene reaches one, `submit_work_of` returns an error for the generation instead of a cap. The 7,000 random
   genes did not reach it, but nothing shows that the path works. *Fix:* the sizes are numbers when the cost is
   computed, so `adapters6` should keep the numeric target (`PropertyExpr::min` of the two sides, substituted from
   the operands' properties) for costing, and use `Unknown` only for the spelling. Then run `work_parts`,
   `static_cost` and `eval` on F2's `?` fixtures as a test.

6. **Witness C4 does not check what its docstring says** (`gen_tables.py:3391`). The docstring says each property
   must be one "their data types carry". The code checks against one global set (`len, rows, cols, bound, value,
   cells`), so `in0.cols` on a vector, or `in0.len` on a matrix, passes the witness and fails only when Rust
   evaluates it. Also, C3's elementwise default is the generator's own rule copied, so a row it misclassifies would
   pass both. *Fix:* check each `in<k>.<p>` against the properties of argument k's data type in `datatypes.tsv`
   (`cells` always allowed). For C3, derive "elementwise" from the spelling (a scalar output whose spelling uses no
   SOAC), not from the default.

7. **The crop can make two sizes equal** (`tests_interp6.rs:96`). The crop takes `s.min(full[d])` with the
   sequence 7, 5, 3, 2. A stored size smaller than its slot collapses onto another size (for example k = 3 with its
   slot at 5, next to a d cropped to 3). The case was chosen for distinct sizes, so that a `fit` or `clampix` to the
   wrong size shows up. *Fix:* after cropping, assert that the cropped sizes are as distinct as the case's were, or
   choose the smallest unused distinct value at least 2.

8. **A negative size would be cast to `usize`** (`interp6.rs:912-913`, `934`, `947`). `replicate`, `hist` and
   `histrows` cast an `i64` with `as usize`, so a negative size would allocate a huge array instead of raising
   Futhark's error. Sizes are at least 1 by the entries' contracts, and a size node is checked only against its
   value. *Fix:* use `usize::try_from` and report the bug, as `fit` does (`227`).

9. **`layout_bound` uses master 5's per-node bound for master 6** (`futhark_sig.rs:1293`). The bound is
   `1 + d²` a node, but a master 6 `matmul` costs `1 + d³`, and a node with `fit.pair` plus a costed read can reach
   about `1 + 4d²`. Nothing in fuller's or phylu's `src` calls it. *Fix:* remove it, or make the master 6 branch use
   `1 + d³ + 3d²`.

10. **Reference coverage is narrower than the plan row** (`tests_interp6.rs:135-138`). Plan 7.3 item 2 says "every
    stored case". The k-means references are checked on their two smallest cases. This is disclosed in the plan's
    status paragraph. Keep the disclosure, or run the rest as a slow test.

11. **Wording.** phylu's "a body's once a trip" needs its noun: "a body adapter once a trip". In fuller, "proof 1's
    second half", "G3's decoding" and "passes 2 to 4" are plan-internal. They are fine in the plan, but the module
    header (`interp6.rs:1-18`) would read better with "the decoder (`typed_environment`)".
