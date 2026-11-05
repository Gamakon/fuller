# Section 7 G3 code review: the decoder's passes 2 to 4 (`src/typed_environment.rs`, `src/geneframe.rs`)

Reviewed 2026-11-05: fuller `52112171` (plan `docs/PLAN_futhark_kingdom.md` section 7, row G3). Against the design
`docs/DESIGN_futhark_types_and_properties.md` sections 4 to 6, master 6's format `kingdoms/futhark/master/README.md`
section 5 and `kingdoms/futhark/master/6/rows.tsv`, the G1 review `docs/reviews/2026-11-05_section7_G1_code_review.md`,
`CLAUDE.md` and the house rules. Read only, no build, no run. Every line of the diff was read: `typed_environment.rs`
(481), its tests (506), `geneframe.rs` (+773, -64), the plan (+10). Line numbers are at `52112171`.

## Verdict

**Accept. The passes match the design and F0, the decoding is deterministic, and the G1 findings 3 to 6 are fixed.
Eight findings. Finding 1 is the open point, which the lead decides: `hist` should render its constant. Make findings 1
and 2 before F2 renders a substitution, finding 4 before F5 and P2 key on subtrees. The rest are small.**

What was checked and is right:

- **Environment order** (`in_scope`, `typed_environment.rs:68`): the entry's parameters in order, then each binding's
  variables, outermost first, in Binding row order. `env.child` (line 91) appends one depth per binding argument.
  `choose` takes `w mod c`, `index_of` is its least inverse, and a `var.T` with `c = 0` reads `T`'s constant. A `size`
  with no size in scope is a bug, as README 2 says.
- **Properties**: bottom-up through `substitute`, so every property is in its one spelling (`normal`). A var's
  properties are its variable's. A substituted node's properties are its substitute's, so its parent's requirements
  see what renders.
- **Relations** (`geneframe.rs:1022-1055`): exactly README 5.1. The one widening is `a` a literal `<= 1` rather than
  "the literal 1", which is sound. `?` fails every relation, and `<` holds only between literals.
- **Work bound** (`typed_environment.rs:455-477`): the bound is checked first, and when it fails, no other adapter
  renders (`failures`, line 223). A body adapter runs at the node's trips plus one, a node adapter at the node's.
- **Trips**: they come from the `trip` column only. So `tabulate`, `tabrows`, `loop` and `loop1` count, and `let`
  adds a depth but no trip. The count nests correctly: a loop inside a `tabulate` body is at trips 1 and its body at
  trips 2, since `child` clones the parent's count. No lambda is missed: the function-typed arguments of `map_rows`
  and `map2_rows` are fixed terminals (M0047-M0055) with no body and no binder.
- **Determinism**: no `HashMap`/`HashSet` in the commit. The two-process test is real: it runs the test binary
  again with `--exact` as a child process, which gets its own hash seed, and compares the `Debug` text of four
  decodings.
- **"Fails exactly two"**: requirement 0, `hist`'s `in2.len = in1.len -> fit.pair` (`n` against `k`), and
  requirement 3, `get`'s `in1.bound <= in0.len -> clampix` (`k` against `n`). These are design section 6's two:
  `fit (i64.min k n)` on `hist`'s operands and `clampix n i` on `is[i]`. The root keeps `len = k`, which the test
  asserts.
- **G1 findings**:
  - 3: `sum` and `min` sort, `parse` builds through them, and `shape_defects` reports any other spelling.
  - 4: every branch is provoked alone, by message and count.
  - 5: a golden of 81 rows as literals (6×6 + 15×2 + 5×3).
  - 6: `to_text` checks first, and `parse` rejects a second `kingdom` line.

## Findings

### 1. The open point: a substituted `hist.*` renders its index array, where F0 says its constant (lead decides)

`typed_environment.rs:439` substitutes by the first input whose data type is the output's. `hist.add@a.{E}` takes
`size;[]{E};[]{E}` (rows.tsv M0033, M0057), so in every view, not only `i32`, it renders as **in1, the index array**,
of length `n`. README 5.3 (line 175-177) gives the general rule and then lists `hist.*` among the rows that render
their constant. `adapters.tsv:9` states only the general rule. So F0 contradicts itself, and the code follows the
general rule. `histrows.*` (M0064, M0065: `size;size;[]{E};[][]{E}`) has the same fault: the general rule gives it
in3, the values matrix.

**Recommendation: the constant, by F0's explicit list.** Plan 7.1's rule is meant for an input that is the result
carried through: `map2` and `scan` keep their operand's length and meaning, and a loop that runs no trips returns its
state. `hist`'s own-type inputs are its indices and its values, so using either one changes the meaning and makes the
length `n` instead of `k`. The constant `[0{E}]` fitted to `k` by the parent's slot is `replicate k 0`, which is
exactly the histogram with nothing counted, `hist`'s own neutral result. The same holds for `histrows` and `[[0]]`.
The design leaves the choice open (section 8 names only SR's case, "renders as its argument").

**Fix.** Make the substitute a table fact, not a type coincidence. Add a second substitution adapter,
`subst.const` (renders the constant). `hist.*` and `histrows.*` name it in `requires`, and the decoder takes
`Substitution::Constant` when the work-bound requirement's adapter is `subst.const`. Correct README 5.3 and
`adapters.tsv` to match. Add a test that a `hist` at trips 2 renders `Constant`.

### 2. The check does not tie `subst` to `env.trips` (geneframe.rs:1595-1603)

- `env.trips < 2 -> fit.pair` passes the check, and the decoder still treats a failure of it as a substitution
  (line 456 finds the work bound by its subject alone).
- `in1.len = in0.len -> subst` passes too, and renders as `Failure::Adapter { adapter: "subst" }`, an adapter with no
  rendering.
- Two `env.trips` requirements on one symbol pass the check, and the decoder reads only the first.

Failure: a typo in F1b's view becomes a wrong program, with no message. Fix: the check reports a Trips requirement
whose adapter is not a substitution adapter, a substitution adapter on any other subject, and a second Trips
requirement on a symbol. One case each in `every_branch_of_the_table_check`.

### 3. A heavy adapter on a row with no work bound escapes the bound (typed_environment.rs:456)

`substitution` returns `None` when the node has no Trips requirement. So a `fit` on a light row renders at any
depth. README 5.3 promises that no program does more than `max(size) ** depth`, adapters included. This is latent
today: every master 6 row that names a heavy adapter also carries the bound (checked against rows.tsv), and A7 has no
depth. But no check keeps it so. Fix: when any requirement of the tables is on `env.trips`, the check reports a
symbol that names a heavy adapter and has no Trips requirement.

### 4. A resolved subtree does not carry its free variables' properties (for F5 and P2; ResolvedTree, line 182)

`Choice::Variable(Binder { depth, ordinal })` uses binding levels, which are right for whole trees of one entry. A
subtree is a different matter.

- `get(is, var.i64)` under `tab n` and under `tab k` both resolve to `Variable(1, 0)`, so they compare equal. One
  renders `is[i]`, the other `is[clampix n i]`. Failure: F5's fold or marriage shares the two as one, and the shared
  part renders, or is scored, under the wrong adapter. Plan F5's test "a shared part used under two different
  environments evaluates bit for bit" is meant to catch exactly this, but the key belongs to G3.
- The converse: the same body under one more `let` resolves differently (depth 2 for depth 1). This only costs
  sharing.

Fix: a `Decoding::subtree_key(path)` that adds, for each free variable, its binder, data type and properties. That
is the form F5 and P2 key on. State in `ResolvedTree`'s doc that it compares whole trees of one entry. Test both
cases.

### 5. The entry is not checked (typed_environment.rs:357-383)

A `RootRequirement` is never checked against the tables.

- Its property, its relation and kinds, and its adapter (that it exists and runs at the root) are unchecked.
- `out.lenn` reads `?`, so every gene fails `fit.root` silently, where it should be reported as a bug.
- An entry variable of a data type the tables do not have, a property reading `in<k>`, or a duplicate name passes.

Fix: `Entry::check(&KingdomTables)`, called once (in `decode`, or by a `Decoder::with_entry`), on README 5.2's table,
with tests.

### 6. A missing constant is found by a gene, not by the tables (typed_environment.rs:345-350)

When a data type other than `size` has `constant: None` and needs one, the error surfaces only when a random gene
reaches that data type, which is in the middle of a run (see the "never run with a known bug" rule). Two cases need
one: the output of a `var` row, and the output of a heavy row with no input of its own type. Fix: `Decoder::new`
reports these types. The decode reports, once, an entry with no size variable when the contract has `var.size`
(binders bind `i64`, never `size`, so the entry is the only source of sizes).

### 7. Three tests prove less than they could

- `every_random_histogram_gene_decodes_the_same_twice` (tests.rs:415) prints `with_failures` and `substituted`, but
  does not assert them. Assert both > 0, so the 300 genes exercise failures and substitutions.
- The section 6 tables are written by hand (tests.rs:67), not loaded from `kingdoms/futhark/sig/histogram/m6/`.
  F1b should run the same two tests on the loaded view.
- `geneframe.rs:1588-1590`: no test compares a `range` or `flag` subject with a literal. Add one.

### 8. A banned word (house rule)

`tests.rs:492`: "The decoder refuses tables that disagree with the contract." Fix: "The decoder reports tables that
disagree with the contract as a bug."
