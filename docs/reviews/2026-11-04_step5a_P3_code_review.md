# Step 5a P3 code review: the point mutation in expressed cells, marriage as an engine beat

Reviewed 2026-11-04: phylu 95ed5b22 and 8249a5ff (fuller `docs/PLAN_futhark_kingdom.md` 5a, row P3). Read only, no
build. Line numbers are at 8249a5ff.

## Verdict

**Pass with corrections.** The point mutation is correct: its edit count matches the deprecated runner's `m_of`, the
device and the host twin agree, and every edit stays inside its cell's legal alphabet. The parents of a marriage are
never duplicates. Moving the RNG and the hash does not change `true_wgsl_fit`. The GPU log
(`/private/tmp/claude-501/fbench/plan5a/p2a_mock/p2a_mock.log`, 04:18Z) shows 9 of 9 `the_external_` tests passing.
Regime P changed 1 or 2 cells per child, counted 193 and 207. Regime R changed 1 to 4, counted 85, 112, 104 and 99. A
marriage run made 160 offspring. Fix findings 1 and 2 before P4 builds a run on these commits.

## Checked and correct

| Item | Result |
|---|---|
| The edit count `1 + below(k)` | The device (`typed_vary.wgsl:509`, `vy_below(row, 0u, STREAM_LEAF_COUNT, edits)`) and the host twin (`typed_vary.rs:521`, `below(d(0, STREAM_LEAF_COUNT), edits)`) use the same stream and the same slot. `MAX_PICK` and `MAX_LEAF_EDITS` are both 16, and the flag bit is masked the same way on both sides. With k = 2 or 4 the count is 1 to 2 or 1 to 4, the same distribution as `m_of` (`futhark_evolve.rs:1672`). The draws are not the same, because one side uses a stateless hash and the other `GeneRng`, so only the distribution matches. |
| The point table (`build_point`, `typed_vary.rs:407`) | Its cells are `coding_cells`: each expressed node and the Dc cell of each constant. Each cell's class is `layout.alphabet(k, cell, const_len)`, which depends only on the position, so it is the cell's whole legal alphabet and never another type's. Cells with fewer than 2 symbols are left out. A Dc alphabet is checked to be its own indices. The current value's place is skipped, so the new value always differs. Mutation runs before crossover (`typed_vary.rs:952`), so the table matches the cloned parent. |
| Parents never duplicates | `partners` drops every row with the first parent's tree hash, which also drops the first row itself, before the second tournament draws. The GPU test asserts that the two parents' rows differ and their trees differ. |
| RNG switch | `GeneRng` and `fuller::wgsl::true_kingdom::Rng` run the same splitmix64 step and the same `% n` in `below`. They differ only at `n = 0`, where `Rng` panicked and `GeneRng` returns 0. `true_wgsl_fit`'s two seeds give the same sequences. `splitmix` and `subtree_hashes` moved byte for byte. Only one test's draws changed: `ten_thousand_random_marriages...` now uses its own domain-pick stream. |
| Checkpoint format | Format 5 (`point_rate`) was committed in 95ed5b22, and 8249a5ff goes back to format 4. Any version other than 4 is a named error that stops the resume (`checkpoint.rs:317`): the file is not treated as torn and not skipped. The files the P3 GPU run wrote at format 5 were in temp directories. |

## Findings

1. **`marriage_max_nodes` is a workaround that silently turns marriage off in regime R** (`engine.rs:1544`,
   `marriage.rs:502`, `marriage.rs:525`).
   - Failure: the engine's own regime-R start draws trees of 694 to 4,009 nodes, and the GPU log shows R parents of
     up to 4,312. Every such row is above any bound that keeps the subtree domain affordable, so the beat makes no
     pair. Each missing pair is counted as "without a partner" (`engine.rs:5633`), so the stream gives a false
     reason. Inside an island, the tournament runs only over rows within the bound, so a large row is never a
     parent even when it is the fittest. That weakens selection for this one operator. Survival selection is not
     affected. In regime P the trees stay small (largest parent 45 nodes), so the bound never binds there.
   - The default bound is 0, meaning no bound. With that default, one pair of drawn genes ran more than 600 s, with
     no check of `killme.trigger` inside the enumeration.
   - Fix the root cause in `row_subtree_domain`: sample the domain instead of enumerating it. Draw a node of parent
     a by its weight, then a node of parent b of the same type, which is linear in the trees' sizes. Then remove
     the bound.
   - Until that is done: count the rows excluded by the bound as their own number in the summary event, and make
     `futhark_config_errors` reject `marriage.share > 0` when the bound is 0.

2. **With `expressed_edits` on, marriage offspring are mutated by a different operator** (`engine.rs:5606`,
   `marriage.rs:676`).
   - Failure: `expressed_edits` makes the schedule `leaf: 1.0` (`engine.rs:9182`). `install` then builds a
     same-signature `LeafTable::build` with `LeafSpec::default()`: 1 edit, the arity kept, Strahler weights. So a
     mutated offspring gets neither the regime's point mutation nor its count.
   - The deprecated runner never mutated marriage offspring.
   - The GPU marriage test sets `mutation: 0.5`, so it runs this mixture and does not detect it.
   - Fix: when `expressed_edits > 0`, give `install` the point table (`build_point` over its genome with the same
     edits). Otherwise reject `marriage.mutation > 0` for the Futhark kingdom, which is what the old runner did.

3. **Multi-edit children differ from `point_mutate`** (`typed_vary.rs:407`, `typed_vary.wgsl:509`).
   - The old operator recomputes `coding_cells` after every edit and draws with replacement
     (`fuller src/futhark_rings.rs:382`). The new one picks m distinct cells of the parent's program at once.
   - Failure: when edit 1 turns an expressed function into a terminal, edit 2 can land on a cell the child no longer
     reads, so the edit is wasted. When edit 1 turns a terminal into a function, the newly read cells cannot be
     edited in the same child. The count distribution is the same; the distribution of which cells and symbols are
     changed differs for m >= 2.
   - The tests count the cells of the parent that changed, so they cannot see this.
   - Fix: state the difference in `Config::expressed_edits`'s documentation. Measure on the host twin the share of
     children whose later edit lands on a cell the child does not read. If that share is not small, apply the edits
     one at a time against the child's own cells.

4. **`expressed_edits` leaves the other row operators at the island's rates** (`engine.rs:9182`,
   `futhark_config_errors` at `engine.rs:3934`).
   - Failure: every child is edited and then paired for crossover (`typed_vary.rs:952`). Inversion, IS, RIS, the Dc
     operators and cleanse still fire at their rates. A crossed child can therefore change more than k expressed
     cells. In the old runner a child came from crossover or from mutation, and a crossover child was mutated half
     the time.
   - The GPU tests set the rates to `none`, so this mixture is untested.
   - Fix: P4 sets and states the rates on the run card. `futhark_config_errors` should name every non-zero row
     operator rate when `expressed_edits > 0`, unless the run asks for it.

5. **The brief is out of date.** Row P3 of `docs/PLAN_futhark_kingdom.md`
   (line 894) and row 8 (line 998) still describe `expressed_hits` as a per-cell rate over the median node count.
   8249a5ff measured that this fails (278 cells per child at the median) and replaced it. Fix: rewrite the P3 row and
   P4's card item as `expressed_edits` (2 in P, 4 in R, edits per child) and name the failed rate.

6. **The GPU marriage test checks less than its documentation says** (`futhark_typed.rs:1306`, `:1312`).
   - The documentation says each offspring is in its island's last rows and both offspring of every pair are
     present. The test asserts only that the row is not an elite row and that the count is even.
   - Fix: assert that each row is at least `hi - 2 * pairs` and that each pair's two rows are adjacent with the same
     two parents.

7. **Small items.**
   - `let _ = std::fs::remove_dir_all(&out);` at `futhark_typed.rs:437`, `:472` and `:570` discards an error, which
     the house rule forbids. The GPU tests already use `unwrap_or_else(panic)`.
   - Two blank lines in a row at `engine.rs` around line 8919 and in `true_wgsl_fit.rs` after
     `DUMP_HEADER_WORDS`.
   - No test feeds a newer-format (format 5) checkpoint to the reader. The code path is the same as format 3's.
   - When both are on, `point_table` and `marriage_breed` each read the whole population back and express every row
     every generation, so every row is expressed twice. Express each row once a generation and share the result.
