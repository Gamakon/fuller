# Section 7 G1 code review: the gene frame's data model (`src/geneframe.rs`)

Reviewed 2026-11-05: fuller `e526a62f` (plan `docs/PLAN_futhark_kingdom.md` section 7, row G1; the design
`docs/DESIGN_futhark_types_and_properties.md` section 3). Read only, no build, no run (other agents were building).
Every line of the commit's diff was read (`src/geneframe.rs` +1,081, the plan +12), and the pre-commit
`typed_depth_table` was compared with the new one. Line numbers are at `e526a62f`.

## Verdict

**Accept. G1 is correct and changes nothing for an existing kingdom; three findings belong to F0 and G3, three are
small fixes to make before F1.**

- The six tables match the design table by table; `Min` is justified by plan 7.1 (finding 7).
- The derived arity equals today's for every symbol, not only the counts: `derives_todays_arity` compares whole
  `Symbol` rows (line 1675) and every signature in order (line 1684).
- The only change outside the new code is `typed_depth_table`, now counted from `typed_depth_signatures`: the same
  rows, in the same order, with the same ids and fields as before. Nothing phylu matches on changed.
- The text form round-trips every checked table tested; `check` catches the user's three cases except the third,
  which the plan gives to G3 (finding 1).
- No "nucleotable", no `#[allow]`, no new `_` names; every defect reads "our software has a bug: ...".

Findings 1, 2 and 3 must be settled in F0 (master 6's README) before G3 is built. Findings 4 to 6 are small.

## Findings

### 1. The "same kind" requirement check that G3 adds would reject the design's own index requirement (F0, G3)

`src/geneframe.rs:1206-1217` checks that a requirement names an existing input and a property it carries; it does not
compare the kinds of the two sides. The plan gives that check to G3 (7.1, data model row: "the check that its two
sides are properties of the same kind"), so its absence is not a G1 defect. But the commit's own `get` row
(line 1859, `in1.bound = in0.len`) and the plan's "an index Requirement is `bound ≤ len`" (7.1, adversarial H2) compare
a `Bound` with a `Size`. Failure: G3 implements the plan's sentence literally and the check rejects every index row
(`get`, pathfinder's `zero`/`last`/`next`, k-means's `centres[0i64]`). A side that is a size name (`Size("n")`) has no
property type at all, so it has no kind to compare. Fix: F0 writes the relation table: `=` between two properties of
one kind; `≤` from a `Bound` to a `Size`; a size name and a literal count as `Size`. G3 implements that table.

### 2. A Requirement cannot express the work bound or the lambda-body length that plan 7.1 chose (F0, G3)

`Requirement` (line 908) is "input `position`'s `property` equals an expression". Plan 7.1 choice 1 needs "every heavy
row requires `trips < depth`", where `trips` is a property of the environment (pass 2), not of an input, and the
relation is strict. Plan 7.1's array-valued lambda body needs "its length is a size from outside the lambda", which is
a condition on the body's scope, not an equality with one expression. Failure: F2a/G3 find no row to write these
into and add a side channel beside the tables. Fix: F0 states the Requirement's two extra parts (a side that is an
environment property; the relations `=`, `≤`, `<`), and G3 adds them with the relation of finding 1.

### 3. Sums and minimums are equal only when their terms come in the same order (G3)

`PropertyExpr::sum` (line 750) flattens but keeps the order, `Min` keeps the order, and equality is the derived
`PartialEq`. So `in0.len + in1.len` differs from `in1.len + in0.len`, and `min(n, k)` from `min(k, n)`. Design 8
weakness 3 accepts "a sum is equal only to the same sum", but a sum is the same sum in any order. Failure: `concat`
of two arrays fed to a `map2` whose other operand is a `concat` in the other order fails its requirement, and the
rendered program carries a `fit` that does nothing; the e-graph, which commutes operands, makes this common. Fix:
`sum` and a new `min` constructor sort their terms (`PropertyExpr` already derives `Ord`), `parse` builds through
them, and `shape_defects` reports an unsorted sum or min, so the text form and the comparison see one spelling.

### 4. Most of `check`'s branches are untested, including "an argument with no symbol"

`the_table_check_lists_every_defect` (line 1928) provokes 11 defects. Untested branches of `check` (lines 1099-1243):
an argument naming a symbol that is not in the table (line 1146), a symbol with no Argument rows (line 1172), an
output at a position other than 0, a duplicate symbol, data type, data type name or property type id, a data type
whose element or component is not a data type, a property name that is not an identifier, a requirement on a property
its input does not carry, a binding setting a property its data type does not carry or setting one twice, and an
`in<k>.<p>` on an existing input that does not carry `p`. Failure: a later edit inverts one of these conditions and
`cargo test` still passes. Fix: one table per branch (or one table with all of them), each defect asserted by its
message and the total count asserted, as the existing test does.

### 5. Two of the derivation tests compare one counting of the rows with another counting of the same rows

`derives_todays_arity` for the typed-depth kingdom (line 1694) builds the tables from `typed_depth_signatures()` and
compares them with `typed_depth_table()`, which now counts the same signatures (lines 471-490); for SR and Law the
ordered inputs are `vec![Ty::F; total_in]` built from today's arity (line 1691). These prove that
`Arity::from_arguments` counts as each kingdom does, which is what the plan's G1 row asks, and I checked the
`typed_depth_table` refactor by hand against `e526a62f^`. But no test pins today's typed-depth table independently:
failure: a later change to `typed_depth_signatures` (a dropped absorbing row, a wrong depth) changes both sides and
passes. Fix: a golden for the typed-depth kingdom: its row count and the full arity of one terminal, one raising, one
absorbing row at each depth, written as literals.

### 6. `to_text` writes tables that read back different, without an error, when the table fails the check

`to_text` (line 1246) checks only that text fields hold no tab or line break. A nested `Sum` is written flat and read
back flat, so it reads back different; a size name or a binding property name holding `;`, `=`, a space or a tab
writes a line that `parse` (line 1294) cannot read or reads as other rows; a second `kingdom` line silently replaces
the first. All are tables that `check` rejects, but nothing makes a writer check first. Failure: a tool writes an
unchecked table and reads back a different one with no message. Fix: `to_text` returns `check`'s defects when there
are any, and `parse` rejects a second `kingdom` line.

### 7. `Min` is justified, but the plan's 7.1 layer table does not list it

The plan's "Sizes are chosen, not computed" paragraph in 7.1 says "every length and bound is a size name, a literal
or a symbolic expression of them (`i64.min n k`) ... the `min` of two sizes is symbolic, not `?`", so `Min`
(line 740) is the plan's, not an addition. The 7.1 layer table's data-model row still lists "a sum kept symbolic" and
no minimum. Fix: the lead adds "and a minimum" to that row (the plan is the lead's file).

### 8. Notes, no change needed in G1

- `from_signatures` (line 1011) makes every data type a `Scalar` named by its `Debug` text (`Futhark(3)`). Right for
  the carry-over tests; F1 must build the Futhark tables from master 6, not through this function, or every array
  would carry no `len`.
- `PropertyType.id` is checked for uniqueness and referenced by nothing: rules, requirements and bindings name a
  property by its data type and name. F0 should say which of the two is the key in master 6's files.
- The k-means contract test (line 1794) loads the 14,359-type view in the default `cargo test`; its seconds are
  printed, as the plan asks of long tests.
