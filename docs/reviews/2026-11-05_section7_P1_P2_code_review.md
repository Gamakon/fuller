# Code review: plan section 7 tasks P1 and P2, the Dv lane on phylu's device and the keys on the resolved tree

Reviewed 2026-11-05: phylu `fb69e6cc` (P1) and `47348333` (P2). Read only (`git show` and the files at HEAD); nothing
built or run (the lead runs the GPU tests). Checked against the house rules (`qdrant/docs/Rules.md`), both CLAUDE.md
files, the design (`docs/DESIGN_futhark_types_and_properties.md` 3.2, 4 and 6: the device never resolves a variable),
plan section 7's P1 and P2 rows and tests, and the G2 review.

**Verdict: accept, with fixes 1 to 3 before marriage or the evaluator runs on a master 6 view.** The device and its host
twin agree on every Dv word, read side by side. No operator can part a word from its cell. A kingdom without `var.T`
takes the old code path everywhere, and the goldens show it for 15 kingdoms. The keys are taken on G3's resolved tree
and fail as a named bug without a resolution, so nothing falls back to raw words. Three problems remain. Marriage
grafts raw index words into another scope (finding 1). The MIDDLE mode spends its domain picks on Dv lanes (finding 2).
The size guard now builds and decodes the oversized genes it exists to skip (finding 3). The device half is unproven
until the four GPU tests run: the three new Dv tests, and the existing parity tests on the kingdoms without `var.T`,
because P2 changed `typed_vary.wgsl` for every kingdom.

## What was checked and holds

- **Offsets and strides.** `FRAME_WORDS` = 12, with `dv_off` and `dv_len` at words 10 and 11 (`typed_layout.rs:428`,
  :506). All four kernels read the table at stride 12 (`walk.wgsl`, which `typed_decode.wgsl` includes, plus
  `typed_init.wgsl` and `typed_vary.wgsl`). `kernel_contracts.rs` checks `FT_STRIDE`, `FT_DV_*`, `CELL_*`, `DV_WORDS`,
  `TI_STREAM_DV`, `SYM_*` and `LEAF_DC` against the host. No other kernel binds `frame_table`. `cell_map` marks
  exactly the Dv span with `CELL_DV | dense`, and the test checks it against fuller's `spans()`.
- **Init RNG.** The host draws `below(draw(seed, gen, row - base, pos, STREAM_DV), DV_WORDS)` and the device draws
  `ti_below(row, c, TI_STREAM_DV, DV_WORDS)`. The key (row position) and the reduction are the same as for the Dc
  draw, which is already proven. Stream 37 is new, so no earlier draw moves.
- **Walk and decode.** Host `walk_reference` (`SYM_VARIABLE => cells[f.dv_off + fk]`), `walk.wgsl` `walk_gene` and
  `typed_decode.wgsl` :430 read the same word, and fk < head + tail = `dv_len`. The Dc counter now advances only on
  `SYM_CONSTANT`, at all three decode sites and in `cl_walk` and `cl_place`. No `sym_const[..] != 0u` is left in any
  kernel. `check_cell` and `validate_gene` test `CELL_DV` before the symbol check, so a Dv word is never read as a
  symbol, and they report defect 12 in fuller's own text.
- **Point mutation.** Host (`typed_vary.rs:855`) and device (`typed_point_main`) branch on `CELL_DV` before any
  symbol logic and redraw on `STREAM_MUT_SYMBOL` at the cell's own slot. The lanes are appended per gene in the same
  block order on both sides, because `cells_nondc` is `point.cells`.
- **Inversion, IS, RIS.** Only head domains carry a lane (`Dom.lane` on the host, `dom_pick` `kind == 0` on the
  device). Word i of the lane belongs to head cell i. The same `start`, `len` and `ins` drive `reverse_range` and
  `seg_move` over the head and over its lane. The scratch for the lane is the head's length, which is within
  `max_seg`. No operator moves a segment across types.
- **Recombination.** Host `swap_cells` keys each Dv word on `dv_owner`, gene-relative. Device `typed_cross_swap_main`
  keys it on `dv_owner(w)`, row-absolute, against row-absolute ranges, and forces the chunk to run when it holds a Dv
  cell whose owner may lie outside the chunk. Applying the three ranges in sequence on the host and taking the
  parity on the device give the same swaps. `swap_gene` and gene transposition copy the gene's whole width, Dv
  lanes included (the lanes sit inside the gene).
- **Cleanse.** `cl_walk` captures the word of the node's old row and `cl_place` writes it at the node's new frame row
  `r`. A collapse writes the `?_T` placeholder, never a `var.T`. The host twin goes through fuller's `write_back`,
  which places Dv words (G2).
- **LEAF and point tables.** A `var.T` keeps its symbol position and adds its Dv word as a `LEAF_DC` position with
  `class_len = DV_WORDS` and `self_pos` = the word. `build_point` uses `edit_cells` (G2's fix) and checks the word's
  range. The host draw (`edits`) and the device draw (`v = below(len - 1)`, skip self) are one rule.
- **Ring walk.** The lane rotates with its window, and the freed words come from the index ring.
- **Keys.** `GeneCodec::subtree_keys`/`tree_key`, `MarriageParts::tree_keys`, `PairCheck`'s identical check,
  `Genetics::marry` and `expressed_key` all go through `Resolution` when `has_variables`. Each fails with "our
  software has a bug" when no resolution is given (`FutharkEvaluator` sets `None` today, so a master 6 view errors out
  there and does not run on raw keys). `subtree_key` carries free variables with their properties (G3 finding 4).
  The tests show one key for two words naming one variable and two keys for two variables, for codec, marriage and
  `expressed_key`.
- **Kingdoms without `var.T`.** Every changed branch on the host is guarded by `dv_len == 0`, by `lane: None`, or by
  `sym_const` being 0 or 1. The 15 fingerprints (layout and draw, variation and keys, marriage) were taken before
  each commit.
- **House rules.** No `#[allow]`, no `_`-prefixed names, no stubs. Every gap is a named bug, except finding 7.

## Findings

1. **Marriage grafts raw index words into another scope** (medium; `evolve/marriage.rs:444-450`
   `subtree_offspring`, `futhark_eval/evaluator.rs:1494-1497` `Genetics::marry`). The identical-pair check compares
   the two subtrees' keys, each in its own parent's scope. The graft (`replace`, then `write`/`write_back`) then
   carries the donor's raw words into the destination, where a word means `w mod c'` of a different set of
   variables. *Failure:* the entry has one `S` variable, `a`. Parent A's `var.S` with word 1 sits under a binder of
   an `S` variable `x`, so its scope is (`a`, `x`) and it names `x`. Parent B's `var.S` with word 1 sits at the root,
   whose scope is (`a`), so it names `a`. The keys differ (free `x` against free `a`), so the pair is let in. After
   the swap, B's word 1 in A's slot names `x` again, and A's word 1 in B's slot names `a` again. Both offspring are
   their parents, so the pair was a no-op, and the subtree B gave is not the program A received. The reverse also
   happens: a pair kept out as identical would have produced a new program. Design decision 3
   says the bridge writes a variable back from the destination's environment, and marriage is such a bridge.
   *Fix:* after `replace`, decode the offspring. For each `var.T` in the grafted subtree whose donor choice is a free
   variable, set its word with G3's `Environment::index_of` for the same binder in the destination environment.
   Keep the raw word only when that binder is not in scope there, and record it as a retarget. Then drop pairs whose
   offspring `tree_key` equals its parent's. Test it on a master 6 view with a binder, such as histogram's
   `tab`/`map`.

2. **The MIDDLE mode picks among Dv lanes as well as symbol domains** (low to medium; `typed_vary.rs:843`, and the
   device's `mode_table`). The lanes are appended to `point.domains`, so `pick_domains(point.domains.len(), x, ..)`
   draws from twice as many domains when every type has a `var.T`. *Failure:* the k-means view has 9 types, all with
   a `var`, so there are 18 point domains per gene. With x = 4, a child gets on average 2 symbol domains instead of 4.
   The commit's claim "symbol cells are hit as without var.T" holds for HIGH and LOW but not for MIDDLE. *Fix:* pick
   among the symbol domains only, and let a picked (gene, type) hit its own lane at the lane's scale. Or draw the
   lanes on a pick stream of their own. Test that MIDDLE's symbol hits are the same with and without `var.T`.

3. **The size guard builds and decodes the genes it exists to skip** (medium, run time;
   `futhark_eval/evaluator.rs:1510-1514`, called only from `render_gene` :2570 for genes over the size bound). For a
   kingdom with `var.T`, `expressed_key` calls `express`, builds the tree, then runs `Decoder::new` and all of G3's
   passes, on genes that may be 64 times the reference program or more. That is exactly what F3a's guard was written
   to avoid. The resolved key is not needed there. Decoding is a function of the tree, so two genes with equal
   level-order symbols and words always name one program: a raw whole-tree key can split one program into two keys,
   but it can never merge two programs. For an always-worst oversized gene, a split costs nothing. *Fix:* key
   oversized genes with the raw walk hash in every kingdom, and keep the resolved key for comparisons where a split
   matters (marriage). The error text in `typed_resolve.rs:582` ("two programs one") is true only for subtree keys,
   so correct it.

4. **`Resolution` rebuilds the decoder on every call** (low to medium, run time; `typed_resolve.rs:542-552`). Each
   `decode`, `tree_key` and `subtree_keys` call runs `Decoder::new`: `tables.check()`, every symbol's rules,
   bindings and `decode_order`. That happens twice for each marriage pair checked, and once for each key. *Fix:*
   build the decoder once per beat or per evaluator, for example by having `Resolution` own `Arc` tables and contract
   and hold the prepared decoder state (a small fuller change, so that `Decoder` can own its inputs or be built from
   `Arc`s).

5. **`NodeTree` drops a `var.T`'s word, so `relevel` reports the wrong reason** (low, latent;
   `typed_write_back.rs:467-474`, :495, :526). `nodes()` records `ci` only for `is_constant`. `relevel` then rebuilds
   a `var.T` with `const_index: None`, fuller's `place` returns `MissingVariableIndex`, and line 526 reports that as
   `Unfit::NotClosed`. *Failure:* the first time fold (`fold_to_dag.rs:441`, :724) or snap (`write_back.rs:440`)
   runs on a kingdom with `var.T`, every rewrite that holds a variable is silently counted as "does not close". Per
   the rule "bring the full machinery to every kingdom", it will run there. *Fix:* record the word for
   `is_variable` too (relevel's `ci != MAX` test then carries it), and map `MissingVariableIndex` and
   `DvWordPastRange` to "our software has a bug".

6. **`Genetics::neighbours` dedupes by the raw tree** (the open item; low now, medium as soon as anything calls it;
   `futhark_eval/evaluator.rs:1434-1452`). The only caller today is `evaluator/tests.rs:202`. For each expressed
   `var.T`, fuller's `alphabet` lists `DV_NEIGHBOURS` = 64 words, and the raw `HashSet<Tree>` keeps all 63 that
   differ from the current word. At most c - 1 of them are different programs, and the rest are the gene itself
   (`t != own` compares raw trees). With c = 1, all 63 are the gene. A hill-climb or ranking over the neighbours
   would evaluate copies of its own starting point. *Fix:* when `has_variables`, dedupe on the resolved
   `tree_key`, excluding the gene's own key, and require a resolution (a named bug without one). Test it with one
   variable and with two.

7. **The ring walk silently keeps old words when the index ring is empty** (low; `futhark_eval/rings.rs:362`).
   `Rings::complete` requires an index ring for every frame with a Dv lane (:234), so an empty ring after
   `complete` is our bug. Keeping the rotated words hides it. This breaks the house rule "no graceful diversions".
   *Fix:* return "our software has a bug: the index ring of {ty} is empty after complete".

8. **The goldens cover 15 kingdoms, one gene each** (low; `typed_init.rs` `every_kingdom_without_variables_..._p1`,
   `typed_vary.rs` `kingdoms_without_variables`, `marriage.rs` fingerprint). fuller's G2 fingerprints also cover the
   master 5 A7 view, the three pixel views, wgsl, safewgsl, true wgsl and the k-means view. The WGSL kingdoms run on
   phylu's typed core, and they are not here. Every fingerprint uses `ChromosomeShape::single`, so the multi-gene
   branches (`swap_cells` with g1 ≠ g2, gene transposition, `cx_gene`) are not fingerprinted. Their no-Dv branch is
   the old loop verbatim and `swap_gene` is unchanged, so the risk is small. The device side for these kingdoms rests
   on the existing GPU parity tests, which must run again because P2 changed `dom_pick`, IS/RIS and `cross_swap` for
   every kingdom. *Fix:* add the missing kingdoms and a two-gene shape to the three fingerprints.

9. **LEAF gives an expressed `var.T` two positions of full weight** (low; `typed_vary.rs:396-405`). One position is
   its cell and one is its word, both of weight w, so a variable leaf takes twice the edit mass of any other leaf
   under the Strahler or depth weighting. The point table matches fuller's `point_mutate`, where the word is a cell
   of its own. In the same-signature table it is a change of distribution that nothing records. *Fix:* give each of
   the two positions w/2, or record the doubling as a decision in the plan's P2 row.

10. **The key tests use the root scope only** (low; `typed_resolve.rs` tests, `marriage.rs:1166-1215`). The toy
    resolution has no bindings, so "the same word in two scopes names two variables" and binder-relative free
    variables are never exercised through phylu's `Resolution`. They rely on fuller's G3 tests. *Fix:* with
    finding 1's test, key a master 6 view whose trees hold a binder.

11. **Wording** (low). Both commit messages say the device "refuses" a Dv layout ("P2 refusal lifted"), a word the
    house rules ban; say "a named bug until P2". `stable_hash` (`typed_resolve.rs:516`) relies on the sequence of
    writes that std's derived `Hash` makes, which std does not promise to keep across Rust versions. The keys live
    within one run today. If `oversized {key}` text ever reaches a cache that persists, finding 3's fix removes the
    issue there.

## Severity summary

| # | Where | Severity | Blocks |
|---|---|---|---|
| 1 | marriage graft, raw words across scopes | medium | marriage on a master 6 view |
| 2 | MIDDLE picks Dv lanes | low to medium | a run with `middle > 0` on a `var.T` kingdom |
| 3 | size guard decodes oversized genes | medium (run time) | the evaluator on a master 6 view |
| 4 | decoder rebuilt per call | low to medium (run time) | none |
| 5 | `NodeTree` drops var.T words | low (latent) | fold or snap on a `var.T` kingdom |
| 6 | `neighbours` raw dedupe | low (no caller yet) | the first caller |
| 7 | ring walk's silent fallback | low | none |
| 8 | goldens' coverage | low | none |
| 9 | LEAF double weight | low | none |
| 10 | key tests at root scope only | low | none |
| 11 | wording, hash portability | low | none |
