# Code review: plan section 7, tasks F1b (0f2d4d44) and F2a (d499f633)

**Verdict: accept, with seven findings to fix.** Nothing blocks F2. The master 6 views are computed the same in Rust
and in `view.py` for all 7 committed views. The rule that a demanded data type with no leaf gets its `var` row is
sound: it is the design's `c = 0` fallback, not a cover for a defect in the table. The k-means gene sizes come from
the real layout. Every reference's index words come from G3's `index_of` and are checked by decoding again. The fixture
is the master 5 renderer's output. The `random_gene` change cannot alter a master 5 draw. Findings 1 and 2 matter most:
one is a test gap that F2 would otherwise inherit, the other a rewrite path that ends in a passing test.

Read only: `git show` of both commits, `view.py`, `gen_tables.py` (witness V6 and `master6_main`), `master6.rs` in
full, `futhark_table.rs` `Kingdom6`, `geneframe.rs` `KingdomTables::check`, `typed_environment.rs` (`Binder`,
`index_of`, `check_entry`, `Entry::of_kingdom6`), `DESIGN_futhark_types_and_properties.md` 5 and 99-100, master 6 README
section 9, `rows.tsv` M0087, plan rows F1b, F2a, F2 and F4. No build was run.

## The questions asked

| Question | Answer |
|---|---|
| `master6.rs` equals `view.py` byte for byte, for all views or some? | **All 7**, though only through the committed files: `every_master6_view_is_what_view_py_writes_byte_for_byte` (`master6/tests.rs:20`) compares Rust's output with the committed `sig/<p>/m6/*.tsv`, and `gen_tables.py --check` (`master6_main`, run by a Rust test) compares the committed files with `view.py`'s output. Nothing keeps the Rust list `COMMITTED6` in step with `SIG6_PROBLEMS` (finding 4). |
| Is the leafless `var` row sound? Is it the `c = 0` fallback? | **Sound, and yes.** DESIGN lines 99-100 say a `var.T` with `c = 0` becomes T's constant. M0087's own note says the same, and README section 9 says "admitted in every view". `[][]i32` in `kmeans_i32` is used only inside the program (`row`, `histrows`, `let.a.i32@m.i32` take it; only `tabrows` builds it), and neither an input nor a binder gives it. So the table is not hiding a missing row. The rule does have costs and holds only as long as one fact stays true (finding 3). |
| Does the loader check every column? | **Most of them.** The symbol columns are checked through `KingdomTables::check`: arguments, rules, requirements, adapters, bindings, trips, constants. Not checked at load: the header lines, the kind of a row against its inputs, constant properties when the constant is empty, and the root and its variables (finding 5). |
| Do 32,484 and 123,020 words come from the real layout? | **Yes.** `the_kmeans_views_load_and_their_genes_are_the_measured_size` (`master6/tests.rs:69`) lays out the committed view with `GeneLayout::from_program_size(..., 192)` and pins `(head, tail, Dc, Dv, width)`. The 32,484 is 5 % above design 7.1's "about 31,000". |
| Do the index words come from G3's `index_of`, and does the test decode and compare the chosen variables? | **Yes, for every leaf.** `build6` (`futhark_sig.rs:3009`) sets each word to `leaf.environment.index_of(ty, binder)`, decodes the whole tree again, and requires `Choice::Variable(binder)` at every variable leaf. `Binder` is (depth, ordinal), so it names one variable. What the round trip cannot see is a label put on the wrong one of two variables of the same type (finding 1). |
| Is the fixture the master 5 renderer's output? | **Yes.** `the_stored_texts_are_master_5_renderings_byte_for_byte` builds every reference in master 5, renders it with `program`, and compares the whole file and each `expected_text6` entry. |
| Does the `random_gene` fix leave master 5 draws unchanged, and is that tested? | **Unchanged by construction; no test pins it** (finding 6). |
| Can `FULLER_WRITE_REFERENCES6` silently rewrite the fixture in a normal run? | **Not in a normal run**: the variable must equal `1`. With it set, though, the test rewrites a source file and then passes (finding 2). |

## Findings

**1. The k-means variables are not checked against what master 5 names (medium).**
`tests_references6.rs:111-113`: the doc comment says "k-means's `assign` is the `let`'s variable", but the test only
asserts that the root symbol is `let.a.f32@m.f32`. Several k-means binders bind more than one variable of the same
type: `loop1@f32.f32[best,bi,c]` binds two `f32`s (`best`, `bi`), and `loop@f32.f32.f32[sum,comp,count,p]` binds three.
`build6` maps labels to ordinals by position (`futhark_sig.rs:3026`), checks only the data type, and the decode round
trip confirms the word it wrote. *Failure scenario:* `[bi,best,c]` written for `[best,bi,c]` builds, writes back,
decodes with no failure, and passes every F2a test. The swapped program then becomes F2's target, and the only thing
left to catch it is F2's comparison with the master 5 text. *Fix:* for the k-means reference, assert as the
histogram and pathfinder checks already do. The `assign` leaf under `get@f32` should read the `let`'s binder (depth 1,
ordinal 0). The `best` in `(sub@f32 DIST best)` should read `loop1`'s ordinal 0 and `bi` its ordinal 1. `sum`, `comp`
and `count` should read ordinals 0, 1 and 2. Then add one general check: each labelled binder has as many labels as
its row has `{2.vN}` names in its spelling, so labels are tied to the spelling's order and not just to the order of
the `binds` column.

**2. The fixture rewrite path ends in a passing test (medium).**
`tests_references6.rs:19-22`: with `FULLER_WRITE_REFERENCES6=1`, the test writes
`tests/fixtures/futhark_m6_references.txt` and `return`s, so the test reports a pass. The fixture is compiled in with
`include_str!`, so the binary running the test still holds the old text. *Failure scenario:* the variable is left set
in a shell. Every `cargo test` from then on rewrites the fixture from master 5 as it currently renders, and the
byte-for-byte test can no longer fail. A change to master 5's renderer then moves F2's target with nothing going red.
Rules.md: "Design tests to fail loudly". *Fix:* panic after writing ("fixture rewritten; run again without
`FULLER_WRITE_REFERENCES6`"), so a rewrite never counts as a pass. Better still, move the writer to a `--write` path
outside `cargo test`, as `gen_tables.py` does with a staging directory and witnesses.

**3. The leafless `var` row: a dead Dv lane and a single pass (low to medium).**
`view.py:533-536`, `master6.rs:712-715`. (a) No binder or root variable is ever of type `[][]i32`, so `var@m.i32` has
`c = 0` in every environment. Its frame's Dv lane (h + t words, part of the 40,881 Dv words) cannot affect the program,
and every such cell renders `[[0i32]]`, a 1 × 1 matrix whose `rows` and `cols` are 1. Requirements that compare a
matrix's rows or columns will usually fail on it and call for adapters. The pinned-size test does not show the waste.
*Fix:* print, then pin, the Dv words of data types that never have a variable in scope. Then decide, with numbers,
whether the layout should give a zero-width Dv to a `var` row with no possible variable. (b) The fix is one pass, not
a loop to a fixed point. The comment "a var row has no argument, so the reachable rows are otherwise unchanged" holds
only because M0087 is the one row quantified `T:vars`. A future row quantified over `vars` with arguments could make
new types demanded and leafless after the recomputation. Witness V6 would then fail loudly, so this is not silent, but
the code states something the master does not guarantee. *Fix:* repeat the leafless step until nothing changes, or
assert that the recomputed rows differ from the old ones only by `var` rows.

**4. Nothing ties `COMMITTED6` to the problems `gen_tables.py` writes (low).**
`master6.rs:791` lists 7 problems by hand. A new entry in `SIG6_PROBLEMS` (`gen_tables.py:2455`) is written and
checked against `view.py`, but Rust never computes its view and the claim "Rust equals `view.py` for every view" is
then false without any test failing. *Fix:* a test that the names in `COMMITTED6` are exactly the directories
`kingdoms/futhark/sig/*/m6/`.

**5. The loader leaves the headers, row kinds and root unchecked (low).**
`futhark_table.rs` `Kingdom6::from_tables`. `data_lines` (`futhark_table.rs:116`) skips the header without reading
it. *Failure scenario:* a regenerated `symbols.tsv` with two columns swapped but the same column count (for example
`rules` and `requires`) loads, and the requirement parser misreads it. A `var`, `const` or `terminal` row with inputs
is accepted. `constant_props` are parsed even when `constant` is empty. The root's properties and requirements, and
each root variable's properties, are checked only by `Decoder::check_entry`, which runs on a decode. No test decodes
the `kmeans_i32` view (it has no reference), so its entry is never checked. *Fix:* compare each header with
`VIEW6_*_COLUMNS` and `adapters.tsv`'s header; reject a row that is not a `function` but has inputs; reject constant
properties with no constant; call `check_entry(&Entry::of_kingdom6(&k))` in
`every_master6_view_loads_its_tables_and_contract_and_lays_out` (`master6/tests.rs:40`) or in `Kingdom6::contract`.

**6. `random_gene`: draws are the same, but nothing tests that, and the commit misnames the change (low).**
`futhark_sig.rs:2109`. The old code partitioned every span's alphabet with `contract.is_terminal`, but split only head
spans, and `partition` draws no random numbers. So every gene the old code produced without panicking is drawn the
same way today, in master 5 and master 6 alike. What actually changed: the old code called
`is_terminal(word)`, which indexes `signature[s as usize]`, on Dc and Dv words. A Dv word at or above the number of
symbols then panics with an index out of bounds. Master 6's Dv alphabets run over 0 .. h + t, so it did. The commit
message ("so a Dv span's index words are not read as symbols") describes a fix to how symbols are read, when the fix
is to that panic. No golden test pins master 5's draws. *Fix:* a test that a master 5 layout's `random_gene` for a
fixed seed equals a stored gene (or its hash), and the right reason in the code comment.

**7. The master 6 s-expression parser accepts empty and duplicate labels (low).**
`futhark_sig.rs:2926`. `tab@a.i32[i,]` gives an empty label, and `loop1@a.i32[s,s]` gives a duplicate that
`labels.iter().position` resolves to the first `s` without an error. Neither can occur in today's 15 references, but
the syntax exists so that F4 and later tasks can write trees by hand. *Fix:* reject an empty label and a repeated
label in one binder, and add both to `a_master_6_text_that_names_nothing_in_scope_is_not_built`.

## What holds

- Witness V6 (`gen_tables.py:2972-2974`) re-derives the leafless rule from the emitted rows (types demanded with no
  non-`var` leaf), which matches `view.py`'s rule (types demanded with no leaf at all) because both union with the
  types that have a variable.
- `view.py`'s `_leafless` and Rust's `leafless` both fail with a bug message when a leafless type has no constant
  (`size`).
- The pinned layout test asserts that a Dv lane is exactly h + t on the frames of types with a `var` row and absent on
  the others.
- `build6`'s scoping is right: a label is in scope only under the bound argument (`path.starts_with(binder path +
  position)`), the innermost binder wins, and a root variable is used only when no binder names the leaf.
- The plan's F1b condition "the pixel family's tests unchanged" holds: neither commit touches the pixel family or the
  master 5 loader.
