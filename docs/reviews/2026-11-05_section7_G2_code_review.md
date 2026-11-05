# Code review: plan section 7 task G2, the variable terminal `var.T` and its Dv lane

Reviewed 2026-11-05: fuller `ef09d26e`, phylu `f2cba8b0`. Read only (`git show`), nothing built or run.

**Verdict: accept fuller's part; phylu needs two fixes before any master 6 layout reaches `futhark_eval`.** The
contract, layout, expression, write-back and every fuller operator are right and tested. But the 34-call
`is_constant(` audit missed a second way code decides "this is a constant": `const_index.is_some()`. Since G2 puts the
`var.T` word in `const_index`, two phylu host sites now send a `var.T` to Dc. That is the review's finding C3 coming
back by another route (findings 1, 2). The ring walk also moves cells without their Dv words (finding 3). Only the
device path is guarded, so nothing stops a Dv layout reaching these sites.

## What was checked and holds

- **Contract** (`kingdom_contract.rs`). The check rejects a variable that takes arguments, one that is also a constant,
  two of one type, and an id that is not a symbol, each with a message that starts "our software has a bug". `bounds`
  leaves `h`, `P`, `t` and `d` alone and adds `dv() = h_T + t_T`. `width` counts it.
- **Size and position of the lane, against design 3.2 item 5 and 7.1.** What was built is one word for each head and
  tail row of each type that has a `var.T`, and no lane for other types. `var.T` is a terminal, so it can sit in every
  head and tail cell of its type. The two options in the question are therefore the same thing, and nothing is wasted
  beyond what the design itself accepts (a word beside a function or another terminal is unread, the same as GEP-RNC's
  Dc). Frames are sized as today. The lanes come after every Dc rather than beside their own type's region. That choice
  is what keeps every existing layout word for word, so it is right.
- **Expression.** `express` and `walk_expressed` read `dv_pos(frame_k)`. A `var.T` takes no Dc word: the test checks
  that the `?` after a `var.I` still reads `Dc_I[0]`. `validate` caps each Dv word below `DV_WORDS`. `place` and
  `write_back` write each word to the row of its own cell. Rows the tree does not use keep their previous words.
- **fuller operators.** `swap_range`, `invert`, `transpose` (IS and RIS), `invert_rows`, `transpose_rows`, `invert_rel`,
  `transpose_rel`, `crossover_rows` and `crossover_rel` all carry the lane. `lane()` puts the tail's words at
  `dv_off + head_len`, and the test checks `dv_of_cell == dv_pos`, so an offset error that the operators shared would
  still be caught. Crossover never swaps a Dv word for a Dc row (`dv_pos` returns `None` past `h + t`). Whole-gene
  operations (`copy_gene`, gene crossover) copy the full width, Dv included.
- **Tests that would catch a broken lane.** `every_row_operator_moves_a_cell_and_its_dv_word_together` tags each word
  with its position and checks every (word, symbol) pair after every operator, over 300 seeds, with `moved > 300`. A
  lane that stays put while its cells move fails as soon as a moved cell's symbol differs, which happens many times over
  300 seeds of the toy alphabet. `an_edit_of_a_dv_word_changes_no_symbol` checks more than 100 edits that reach an
  expressed `var.T`.
- **The 34 decisions.** Each listed site is right as decided. None of them treats a `var.T` as a constant or a constant
  as a variable in a kingdom that exists today. Line numbers drift by one in places (regex_typed.rs 455, not 454;
  typed_layout.rs 295, not 290). Finding 5 is a site where "unchanged" is defensible but the fix is one word.
- **Existing kingdoms unchanged.** The test fingerprints 17 kingdoms against constants taken at `e526a62f`, and asserts
  `dv_off == None` for every one. Finding 7 covers what the fingerprints leave out.
- **phylu.** `ChromosomeShape::new` (typed_layout.rs:128) rejects a layout with a Dv lane as a named bug. That covers
  the device, `typed_vary`, `marriage` and the checkpoint genome.
- **Checkpoints.** fuller's genes are flat `Vec<u32>` of `layout.width`, so Dv words are saved in place with no format
  change. phylu's `Checkpoint.genome` holds device rows, which cannot carry a Dv lane today, so no checkpoint holds a Dv
  word. When P1 widens the frame table it must bump `format_version` (see finding 8).
- **House rules.** Neither diff adds `#[allow]` or `_`-prefixed names. There is no fallback: every gap is a
  `LayoutDefect`. The wording is plain apart from "LANE PARALLEL" and "the adversarial review's M2" in the module doc
  of `typed_gep.rs`. Say "beside its cell" and drop the review label.

## Findings

1. **phylu `src/futhark_eval/rings.rs:371`, `coding_cells`: a `var.T` is taken for a constant (C3 again).**
   `if n.const_index.is_some()` pushes `dc_off + q` and advances `q` for every node that has a word, and a `var.T` now
   has one. *Failure:* in a master 6 view, a gene with a `var.I` and then a `?_I` marks `Dc_I[0]` as the `var.I`'s
   coding cell. Point mutation (`point_mutate`, :383) and `crossover_once` then edit the `?`'s constant on the
   `var.I`'s behalf, and the last `?`'s Dc cell is never coding. In a type with a `var.T` and no `?`, every call fails
   with "a constant with no Dc". The `var.T`'s Dv word is never coding, so no host operator ever changes which variable
   it names. *Fix:* use `contract.is_constant(n.symbol)` for the Dc cell. For `contract.is_variable(n.symbol)`, return
   `layout.dv_of_cell(n.pos)` to point mutation only. The movers (`invert_once`, `crossover_once`,
   `invert_rows_coding_once`) must anchor on the cell itself, or they hit `LaneMovedAlone`. Test it on fuller's G2 toy
   kingdom. *The audit:* also grep `const_index.is_some()` and `if let Some(.) = .const_index` in both repositories.
   phylu `typed_init.rs:507` (test only) counts the same way.
2. **phylu `rings.rs:198`, `Rings::add`: a `var.T`'s word goes into the Constant ring.** `if let Some(c) = t.const_index`
   pushes the Dv word, which can be up to 65,535, as a Dc index. *Failure:* `Rings::complete` and a canvas write that
   index into a Dc cell, and the gene fails `validate` with `DcIndexPastConstants`, or picks a wrong constant if the
   word happens to be below `C_T`. *Fix:* guard with `contract.is_constant(t.symbol)`. The Dv ring belongs to P2.
3. **phylu `rings.rs:330`, the ring walk moves cells without their Dv words (M2).** `cells.rotate_left(k)` shifts a head
   or tail window but not `layout.lane(&s)`. The comment defers this to P2, but fuller's commit message says "every
   operator moving a cell moves its Dv word", and this one does not. *Failure:* after a walk, each `var.T` in the window
   reads the word of the cell `k` rows on, so it names another variable with no edit to its own cell. *Fix:* rotate the
   lane window by the same `k`. Words freed at the end may keep their old values (an index word is legal whatever its
   value). Alternatively, until P2, have the evaluator's constructor reject a contract with `variable_capable`, as
   `ChromosomeShape::new` does. That one guard would also cover findings 1 and 2.
4. **The Dv alphabet is listed out in full: 65,536 values.** fuller `typed_gep.rs:293` (`alphabet`) and
   `wgsl/typed_gene.rs:366` build a 65,536-entry `Vec`. *Failure:* once finding 1 makes Dv words coding,
   phylu's `evaluator.rs:1446` (`neighbours`) lists 65,535 one-edit neighbours for each `var.T`, each a separate `Tree`
   (they differ in `const_index`), although only `c` of them are different programs. `point_mutate` (rings.rs:389)
   allocates 256 KB on every draw. *Fix:* callers draw `rng.below(DV_WORDS)` and check with `in_alphabet`. `neighbours`
   over a Dv word should go through the `c` variables in scope (G3's resolved tree): a note for P2 and G3.
5. **fuller `typed_chromosome.rs:312`, the ADF check uses `is_constant` where it needs `takes_word`.** A `var.T` with
   `dc_index: None` passes `AdfSet::new` and fails later, at `write_back`, as `MissingVariableIndex`, far from the
   cause. *Fix:* `k.takes_word(n.symbol) && n.dc_index.is_none()`. The field `AdfNode::dc_index` (:240) now also holds a
   Dv word: document that, or rename it `word`.
6. **fuller `typed_gep.rs:404` and `:507`, `invert_rows` and `invert_rel` on `Domain::Dv` behave inconsistently.**
   `reversal_changes` (:359) sees no lane for a Dv span, so a Dv row range that reads the same both ways returns
   `Ok(0)`, while any other range returns `LaneMovedAlone`. *Fix:* reject `Domain::Dv` when the function is entered.
7. **The fingerprints do not cover every kingdom.** With `default = []` features, plain `cargo test` checks 14
   kingdoms. The three WGSL kingdoms need `--features wgsl`. The k-means and A7 signature views, the three pixel views
   (`volumepassage`, `cuneus`, `image`) and `Epoch` layouts are not fingerprinted. The reasoning that they are unchanged
   is sound: the Dv lane is added only when `t.variable` is set, and every one of those kingdoms has an empty `variables`
   set. So the risk is low. *Fix:* add k-means, A7 and the pixel views to the golden table, or state in the test that it
   covers only these 17.
8. **The geneframe export and the checkpoint format have no place for a Dv word.** The relational export in
   `typed_gep_tests.rs:735` has no `dv_index` column, so a Dv row exports as two `NULL`s. phylu's `Checkpoint` stores
   device rows at the old width. *Fix:* add the column the next time the relational form is used, and have P1 bump
   phylu's `format_version` when it adds `CELL_DV`, so an old checkpoint is rejected rather than read at the wrong stride.
