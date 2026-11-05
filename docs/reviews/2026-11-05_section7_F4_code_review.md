# Section 7 F4 code review: master 6's reader (`Reader6`, `src/futhark_reader.rs`)

Reviewed 2026-11-05: fuller `e75be462` (plan `docs/PLAN_futhark_kingdom.md` section 7, row F4). Against the plan's F4
row and 7.1 "Literal indices", master 6's format `kingdoms/futhark/master/README.md` (section 9 on `tabrows`),
`kingdoms/futhark/master/6/rows.tsv`, `CLAUDE.md` and the house rules. Read only, no build, no run. The commit touches
two files: `src/futhark_reader.rs` (+930, -22) and `src/futhark_reader/tests6.rs` (235, new). Line numbers are at
`e75be462`.

## Verdict

**Accept. The reader does what the plan's row asks, the round trips are tested as stated, and the two pathfinder
programs that do not read are a gap of the master table, not of the reader. Seven findings. Make finding 1 before the
reader is used on text the renderer did not write (the LLM pools, the k-means pool count); finding 4 is a master row for
the lead to add. The rest are small.**

What was checked and is right:

- **The round trips are tested as the agent states.** `random_genes_read_back...` (tests6.rs:68) asserts, for all 200
  genes of all 7 views, that the tree read renders the same text byte for byte and the same adapters by name; the pool
  test (tests6.rs:120) asserts all 842 programs (10 + 48 + 577 + 9 + 45 + 153), 840 byte for byte and the 2 not read
  named exactly (tests6.rs:169, 174). The 15 F2a trees read back to themselves (tests6.rs:46).
- **The `tabrows` rule is deterministic and always names a variable in scope.** The candidate list is
  `typed_environment::in_scope` (typed_environment.rs:72), the environment's own order, which is the index word order
  (`index_of`, typed_environment.rs:89); the first match is taken, and its word is `index_of` of that same list
  (futhark_reader.rs:2237-2244). No match is a gap, not a guess.
- **`kinds.rs` was not used.** The commit's file list is the two files above; `src/futhark_reader.rs` has no `mod kinds`
  and no reference to it.
- The index words come from G3's `index_of` and every leaf is checked by a second decode (futhark_reader.rs:2265-2272);
  a property placeholder is checked against the decoded property (2273-2280). Master 5's `Reader` is unchanged except
  for the tuple loop state, which its parser never produces.

## Findings

### 1. An adapter's length is thrown away, so a text with an adapter the gene does not render reads as a gene

`strip_adapters` (futhark_reader.rs:1916-1926) removes every `fit`, `map (fit ..)`, `clampix` and `let ad_*`, keeping only
the outermost `fit`'s length as a mark, and only `tabrows` reads that mark. Nothing after the decode checks that the
adapters the tree renders are the ones the text had.

Failure: an LLM pool text `map2 (+) (fit 5 0f32 xs) ys` with `xs` and `ys` of one length reads as `map2 (+) xs ys`,
whose requirement holds, so the tree renders no `fit` at all; the text truncates to 5 elements, the tree does not. The
same for `clampix 3 (i)` on an index whose requirement holds. `read_program` returns `Ok` for a program the gene cannot
express, with other values. The tests never see this: every text they read was written by a renderer.

Fix: record each stripped adapter (kind, length expression) at the tree path where `read6` reads its operand, and in
`read_body` after the last decode compare them with the adapters `program6` renders for the tree (kind, path, length
text); a difference is a gap "adapter the gene does not render". Test: the text above is a gap, and the same text with
the adapter pass 5 renders reads.

### 2. The canonical-tree comparison skips a gene on any error, and its count is not asserted

tests6.rs:93 compares the tree read with `reader.canonical(&tree)` only when that is `Ok`. `canonical`
(futhark_reader.rs:2176-2193) is `Err` for the two intended reasons (a node substituted by its constant, a variable
rendering its constant) but also for a decode failure and for "a variable out of scope" (2185), which would be bugs,
and they are skipped silently. The final assertion (tests6.rs:101, `r.1 + r.2 > 0`) passes with zero genes compared in a
view, provided one gene had an adapter.

What is excluded by design: a whole gene with one constant node anywhere, because at that node the reader reads another
tree that renders the same text (a terminal of the same spelling, or the first substitute candidate). That does not hide
a fault in the text round trip, which those genes still pass, but the tree identity of the rest of such a gene is not
checked.

Fix: return a typed reason from `canonical`, skip only the two constant reasons and panic on the others; assert a floor
of compared genes per view (the counts this run prints); better, compare the two trees everywhere except under the
constant paths.

### 3. "Values checked" is by the text, and the 18 `i32` programs compare one line

The pool test checks the master 5 tree's values against the fixture (tests6.rs:141) and then that master 6 renders the
same text (148, 160). The master 6 tree is never evaluated (there is no master 6 interpreter at this commit); the same
values follow because the same text is the same program. That is sound; the commit message should say so. The 18
`i32:c` programs (histogram bench 9, pathfinder bench 9) are compared through `render_expression` (tests6.rs:106-112),
which keeps only the last line of the rendered program. They are literals, so it holds today; compare the whole
expression so a multi-line one cannot pass on its last line. When F5 or later brings a master 6 interpreter, evaluate
the tree read against the fixture's values directly.

### 4. The two pathfinder programs: a master table gap, and `next` of a literal is dead in every view

Pathfinder llm 12 and 26 index `walls[i64.min (rows - 1) ((0i64) + 1)]` (the fixture's `row walls (next zero)`): row 1.
The reader is right to stop. Master 6's `next@i64` (M0019, `master/6/rows.tsv:19`) is `bound=in0.bound`, spelled
`i64.min ({0.bound} - 1) ({0} + 1)`, and `zero@i64` (M0021) has `bound=1` (plan 7.1 "Literal indices"), so `next zero`
renders `i64.min (1 - 1) ((0i64) + 1)`, row 0. The reader reads `{0.bound}` = `rows` from the text, the decode says 1,
and the check fails (futhark_reader.rs:2273-2280). That is a gap of the table, not of the reader.

It is wider than two programs: `next` of any index of bound 1 is the identity, so `next zero` can never reach row 1 in
any view. The value is reachable by another text (`next (clamp rows 0i32)`), but not this one.

Smallest fix, leaving M0019 and the F2a references unchanged: a new master row beside it, `next@i64` over `size;i64`,
`bound=in0.value`, spelled `i64.min ({0} - 1) ({1} + 1)`. Master 5's text reads as it with the size `rows`, both programs
read, and the test's expected gaps (tests6.rs:174) become empty and its pathfinder llm count 45 of 45.

### 5. The `tabrows` rule is not the one `canonical` applies

Where `cols` is unspelled and unfitted, the reader takes the first size in scope whose value is the body's length
(futhark_reader.rs:2237); `canonical` keeps the gene's own word. Today they agree because every size in scope is a root
size with its own value, so at most one matches. If a later row binds a size variable whose value equals a root size,
the reader picks the least word, the gene may hold the other, and the random-gene test fails at that node with no
fault in the reader. Fix: apply the same rule in `canonical` (the least word whose value matches), so the canonical tree
is the reader's by definition.

### 6. `kinds.rs` is still untracked in `src/`

The plan's F4 row says the lead moves `src/futhark_reader/kinds.rs` out of `src/` first. F4 wrote its own reader and
does not use it; the file is still in `src/futhark_reader/`, untracked. It does not build (no `mod`), but it is a stale
second reader beside the live one. Move it to scratch or delete it, as the lead decides.

### 7. Wording

The code comment at futhark_reader.rs:1798 says "the lead's question, decided here": a session reference that means
nothing to a later reader; say what was decided and why. "its Dc index" (1787) is not defined anywhere the comment
points; say "its literal's position in the data type's pool". The rest uses the plan's and the master README's terms.
