# Review: the shape-classed Futhark kingdom, the proof of concept (codex, Claude Opus, Fable)

**Date:** 2026-10-26 · **Reviewers:** `codex exec` (codex-cli 0.151.0, read-only sandbox), `claude -p --model opus`, `claude -p --model fable` (plan permission mode, read-only) · **Target:** the class kingdom's implementation — `kingdoms/futhark_shapes/`, `src/futhark_shapes.rs`, `src/futhark_shapes_interp.rs`, `src/futhark_shapes_egraph.rs`, `src/futhark_exec.rs`

Three reviews of the same code, run on the command line, each told to **try to break it** rather than to praise it. The first review of the *design document* is `2026-10-26_first_class_shapes_codex.md`; this one is of the *implementation* that document describes.

**What the reviewers found, in the end: 18 distinct findings, of which 17 were accepted and fixed and 1 was refuted with a measurement.** The PoC's central claim — *a shape class is a type, the unification happens offline, and the committed table is the trust boundary* — survives all three, but it did not survive them *unchanged*: the trust boundary was not enforced by anything that ran, and three of the new checks could not see the defects they were written for.

---

## 1. The disposition, finding by finding

| # | reviewer | finding | disposition |
|---|---|---|---|
| 1 | codex | **blocking** — the generator *wrote* the artefacts before witnessing them, so a hand-edited table was laundered | **accepted** — `--check` is now the default; the committed artefacts are compared byte for byte, and the witnesses read the committed files |
| 2 | codex | **blocking** — W8 counted a spelling's placeholders but not their *order*, so `matmul {1} {0}` passed | **accepted** — W8 now checks the placeholders' order (Futhark's application is positional) and that the spelling is the operator's own template |
| 3 | codex | **blocking** — W11 checked only that the constructors a rule names *exist*, so `FRowFnId` → `FRowFnAdd1` passed | **accepted** — W11 now matches every rule against one of three hand-written schemas |
| 4 | codex | **blocking** — the interpreter held a matrix as `Vec<Vec<f32>>`, which cannot represent `[0][m]` | **accepted** — the matrix carries its dimensions; `transpose` of `[0][m]` is `[m][0]` and a `matmul` over an empty inner dimension is a zero matrix, both measured against Futhark |
| 5 | codex | should-fix — "element for element" was not true; the policy was not stated | **accepted** — the policy is stated, tested, and (after opus's measurement) *tightened* |
| 6 | codex | should-fix — the interpreter's dispatch was coupled to the table's row names | **accepted** — a deterministic per-row test now covers every row, and the generators' `--check` guards the spellings |
| 7 | codex | should-fix — W12 did not check that a row maps to *its own* constructor | **accepted** — W12 now checks the mapping is bijective and that each constructor's argument sorts are the row's argument classes' sorts; `Bridge::load` refuses a duplicate |
| 8 | codex | nit — `--check` was not byte-for-byte (universal newline translation) | **accepted** — the artefacts are read in binary mode; a CRLF artefact now fails |
| 9 | codex | nit — W11's duplicate-rule check never fired (`op` against tuple keys) | **accepted** — fixed; the key is now `(op, class)` |
| 10 | opus | **blocking** — *nothing runs the generator's check*: a hand-edited table reaches the engine through `include_str!` | **accepted** — a Rust test shells out to `gen_tables.py --check` and fails if it fails; a missing `python3` is a failure, not a skip |
| 11 | opus | should-fix — the `map2` operand order was never checked; both operands were the same matrix | **accepted** — two arguments of one class now get *different* values, and the rule test binds each variable name to its own value |
| 12 | opus | should-fix — the docs claimed the text format cannot carry a column count | **accepted** — it can: Futhark writes `empty([3][0]f32)`, the engine writes and reads it, and the degenerate shapes are checked against the compiler |
| 13 | opus | should-fix — `parse_f32` could not read Futhark's `f32.inf`, `-f32.inf`, `f32.nan` | **accepted** — the specials are parsed in Futhark's spelling, and written in it too |
| 14 | opus | should-fix — the tolerance was an *absolute* `1e-4` below 1.0, and its stated reason was false | **accepted** — the policy is now exact, with **no tolerance at all**: the stated reason was false (the `c` backend reduces with a sequential left fold, bit-identical over 600 of 600 elements), and a margin that is never needed is a blind spot |
| 15 | opus | should-fix — `output_with_timeout` could deadlock, and a failed write orphaned the child | **accepted** — the input is written on its own thread after the drains start, and the error carries the child's own output |
| 16 | opus | should-fix — `--write` overwrote the real files before checking them | **accepted** — `--write` stages the artefacts, witnesses them, and only then replaces the committed ones |
| 17 | opus, fable | nit — the docs, the check counts, and a review path that did not exist | **accepted** — fixed, and this file is the review the assessment cites |
| 18 | fable | nit — the candidates were cut at `k` *before* they were sorted, so which survived depended on a hash order | **accepted** — the candidates are sorted first, then cut |
| 19 | fable | nit — `heritable()` is not the same as "encodable" | **accepted as a documentation fix** — the claim is now what the code does |
| 20 | fable | nit — `program()` used 1 for a role with no size | **accepted** — it is an error now |
| 21 | fable | nit — `let _ =` silencers in a test | **accepted** — removed, as CLAUDE.md requires |
| 22 | opus | nit — two shape kingdoms would share the `FutharkShape` dual codes | **accepted as a stated limit** — §15.7 of the assessment |
| 23 | codex | the *shipped* rules are wrong | **REFUTED** — the three rule families are sound; codex, opus and fable all found no defect in the rules themselves, only in the check that guarded them |

---

## 2. The two findings that changed the design

### 2.1 The trust boundary was not wired to anything

Codex found that the generator's witnesses could not see a corrupted table, because the generator *wrote* the table first. Opus found the deeper version of the same thing: **nothing in the repository ran the generator at all**. A hand edit to `functions.tsv` would be compiled into the engine through `include_str!` and the whole suite would pass.

That is the finding that matters, and it is now closed by construction:

* `gen_tables.py --check` is the **default**; the committed artefacts are compared byte for byte with what the generator would write, and the witnesses then read the **committed** files — the ones the engine loads.
* A Rust test runs it, so `cargo test` fails on a hand-edited table.

Fable's corruption run is the evidence: **21 of 21 corruptions** were caught by `--check`, including CRLF line endings, a BOM, a missing file, a moved root flag, a changed header, an appended row, a swapped `map2` operand, and a rule corrupted to name a different identity function. Opus re-ran its own fourteen corruptions against the same state and reached the same conclusion.

The honest limit, which fable and opus both state and which the documentation now says: the *witnesses* are the generator's **self-test**. They re-derive the shape relations from the emitted text and compare them with a model written separately from the enumeration code — so they catch a generator whose own model is wrong, but they cannot catch a table that is edited *consistently*. The **byte comparison** is what guards the committed artefact. The two are complementary, and the README now says which is which.

### 2.2 Three of the new checks could not see the defects they were written for

The checks added after the first review were themselves reviewed, and three of them had the same shape of defect they were meant to catch:

* the **rule test** replaced *every* pattern variable with the class's own terminal, so a rule that renamed a variable (`… x … => y`) computed the same thing on both sides and passed;
* the **per-row test** passed the *same* matrix to both `map2` operands, so a swapped operand pair was invisible — `id` sees the same value on both sides, `add` is commutative, and the other row functions ignore their second operand;
* the **numeric policy** floored the scale at 1.0, so below 1.0 it was an *absolute* tolerance of `1e-4` — wide enough to accept a `matmul` over an empty inner dimension that returned `1e-5` instead of the zero it must be.

All three are fixed, and the fixes are *demonstrated* on the corruptions the reviewers named:

| corruption | caught by | what it says |
|---|---|---|
| `fn.arr2…id` spelled `\_xs ys -> ys` | the per-row test | "futhark says 3.9580936, the interpreter says 1.9790468" |
| `const` spelled as the identity | the per-row test, compared **bit for bit** | "futhark says -0, the interpreter says 0" — caught by value, after the re-spelling below |
| the `map2` rule rewritten to `y` | the rule test | the two sides no longer agree |
| `matmul {1} {0}` | W8 | "the spelling uses the arguments out of order ([1, 0])" |
| `FRowFnId` → `FRowFnAdd1` | W11 | "the rule for mat.n.n does not match any schema" |
| a rule stated twice | W11 | "the rule map_rows for mat.n.n is stated twice" |
| two rows mapped to one constructor | W12 | "the constructor FRowFnId is mapped to two rows" |
| CRLF line endings | the byte comparison | "the committed functions.tsv is not what the generator produces" |

The signed-zero case is the one that needed a *measurement* to fix, and the measurement took two rounds to get right. What is now known: `futhark run` is not the C backend — in Futhark 0.28.1 it is the **interpreter** (`src/Futhark/CLI/Run.hs` evaluates the programme with `Language.Futhark.Interpreter`), while the engine compiles with `futhark c` — and the two disagree about the sign of a zero, because Futhark's simplifier folds `x + 0` away. The compiled `x + 0.0` is therefore the identity and keeps `-0.0`'s sign, where the source semantics turns it into `+0.0`. The fold is narrow, not the sign: the compiled program still tells the zeros apart where the fold does not apply — `1.0/x` on `-0.0` gives `-f32.inf`.

That is a trap for a table whose row is spelled as an *addition of zero*: the `const` row compiled to the identity, so it and the identity differed only in the sign of a zero, and no comparison could see the difference — the `-0.0` in every test matrix, which was there precisely to make it visible, had become inert. The row is now spelled `\xs -> map (\_ -> 0.0) xs`, a genuine constant, so it differs from the identity by **value**; the elementwise rows are compared **bit for bit**, so a row that differed only in the sign of a zero could not slip through; and the reduction path, where the compiled program may legitimately differ, is compared on exact values. The tolerance is **gone** — its stated reason (a backend that sums in another order) was false: the `c` backend reduces with a sequential left fold, bit-identical to the interpreter's over 600 of 600 elements.

---

## 3. What the reviewers measured that the code did not know

| measurement | by | what it changed |
|---|---|---|
| Futhark writes a matrix with no elements as `empty([rows][cols]f32)`, and **rejects** `[]` and `[[],[]]` | fable, opus | the engine writes and reads that form, and the degenerate shapes are checked against the compiler |
| `transpose` of a `[0][3]` matrix is `[3][0]`; a `matmul` over an empty inner dimension is a zero matrix | fable, opus | the interpreter's `Mat` carries its dimensions |
| Futhark's `c` backend was **bit-identical** to the interpreter's left fold over 600 of 600 elements | opus | the tolerance's stated reason was false, so the tolerance was made much tighter |
| Futhark writes the specials as `f32.inf`, `-f32.inf`, `f32.nan` | fable, opus | the parser and the writer were corrected |
| `empty([0][3]f32)` round-trips through Futhark's reader | this review | the degenerate end-to-end test is possible at all |
| Futhark's `c` backend **drops the addition of zero**: `x + 0.0` on `-0.0` gives `-0f32` compiled, `0.0f32` under `futhark run` | this review | the `const` rows are spelled as a genuine constant, so no row depends on the fold |
| `futhark run` is Futhark's **interpreter**, not the C backend (`src/Futhark/CLI/Run.hs`) | this review | the two tools' disagreement is a difference of *tools*, not a mystery |
| the compiled program **can** tell the zeros apart where the fold does not apply: `1.0/x` on `-0.0` gives `-f32.inf` | this review | the documents no longer claim the opposite |
| the `c` backend reduces with a **sequential left fold**: `[16777216, 1, -16777216, 1]` gives the left-fold answer, and 600 of 600 elements are bit-identical to the interpreter's | fable, opus | the tolerance is gone — its stated reason was false |

---

## 4. What the reviewers found *nothing* wrong with

Worth recording, because a review that finds something in everything is not a review:

* **the rule families themselves** — `map id`, `map2 id-first` and double transpose are sound; no fusion rule means the 1-D kingdom's `map id` divergence cannot arise (codex, opus, fable);
* **cross-kingdom safety** — `Ty::Futhark` and `Ty::FutharkShape` cannot be confused, in the dual table, in `homologous()`, in the write-back or in the chromosome encoding (codex, opus, fable);
* **the rendering** — every child is bracketed, the curried `map2` is spelled with two arguments, and Futhark accepts the rendered programs (fable);
* **the interpreter's arithmetic** — the `transpose` index swap and the `matmul` contraction are independent of the helper's form and agree with Futhark on the empty inner dimension (fable);
* **determinism** — no hash-ordered iteration leaks into a returned ordering (codex, opus), and the candidate cut no longer depends on one either (fable's finding, now fixed);
* **the round trip** — gene → tree → gene, and every e-graph candidate surviving it (opus).

---

## 5. One limit the reviewers named that is not fixed, and should not be

`Ty::FutharkShape` is a **single family**. If a second interface were generated with different roles, `FutharkShape(0)` would mean a different class in each of the two kingdoms — the very collision the Plain/Shape split was introduced to prevent. The Plain/Shape split separates the *families*; it does not separate two shape kingdoms from each other.

This is stated, not fixed: a second shape kingdom would need its own family (its own code base), and that is a change to the type space, not to the class machinery. It belongs with the next design.

---

## 6. The second round: the signed-zero adjudication (codex, Fable)

The first round's per-row test failed on the `const` row, and the diagnosis took two rounds to get right — so the question was put to two reviewers again, with the measurements and the two candidate resolutions in front of them.

**The question.** `futhark run` gives `0.0f32` where `futhark c` gives `-0f32` for the same program. Which is the engine's ground truth, and what should the comparison be?

**What the reviewers found.**

| finding | reviewer | severity |
|---|---|---|
| the policy that treats the signed zeros as equal **masks a known disagreement** between the oracle and the compiled program — and every future one | codex | blocking |
| with that policy the `const` row is **uncheckable**: corrupting it to the identity changes only the sign of a zero, which the policy accepts | fable | blocking |
| the two `const` rows are **compiled aliases of the identity**, so they contribute no semantic diversity | codex | should-fix |
| "the compiled program cannot tell the two zeros apart" is **false** — `1.0/x` on `-0.0` gives `-f32.inf` | fable | should-fix |
| "a real compiler defect" is **unsupported** — it is a listed simplifier rule (`Optimise/Simplify/Rules/Simple.hs`), and whether Futhark's spec promises signed zeros was not verified | fable | should-fix |
| the tolerance in `close` has **no measured basis**: the `c` backend reduces with a sequential left fold, 600 of 600 elements bit-identical | fable | should-fix |
| a **singleton** inner dimension folds `0.0 + x` away, so a strict-sign interpreter would have to model the optimiser — and the test's sizes merely happen to dodge it | fable | should-fix |
| the README's "the format cannot say how many columns a matrix with no rows had" is **stale** — it contradicts finding 12 and the code | fable | should-fix |

**Where they differed, and how it was settled.** Codex wanted the interpreter to model the **compiled** program (strict signs); Fable wanted the interpreter to keep the **source** semantics and the comparison to accept the two zeros — on the evidence that a strict-sign interpreter would have to model the simplifier's folds, including the singleton reduction, and that the test's sizes would then merely happen to dodge them.

Both agreed on the part that matters: **the `const` row must not depend on the fold at all.** So it is now spelled `\xs -> map (\_ -> 0.0) xs`, a genuine constant, and it differs from the identity by **value** rather than by the sign of a zero. With that, the two positions converge:

* the elementwise paths — a row function, a `map`, a `map2`, a `transpose` — are compared **bit for bit** (`identical`), so a row that differed only in the sign of a zero could not slip through, and the "corruption table" above is true again;
* the reduction path (`matmul`) is compared on **exact values** (`equal`), because there the compiled program may legitimately differ from the source semantics;
* there is **no tolerance** anywhere: its stated reason was false, and a margin that is never needed is a blind spot.

The oracle therefore models the *source* semantics of the spellings — which is what a table's rows are supposed to mean — and the comparison is sharp enough to see a fold that erases a row's meaning.

---

## Appendix — how to reproduce the review

```sh
# the generator's own check, which the test suite now runs
python3 kingdoms/futhark_shapes/gen_tables.py --check

# the codex review, as it was run
codex exec --sandbox read-only -C "$PWD" --ephemeral -o review.md - < prompt.md

# the two Claude reviews, as they were run
claude -p --model opus   --output-format json --no-session-persistence \
      --permission-mode plan --allowedTools "Read,Grep,Glob,Bash" \
      --system-prompt "$SYSTEM" < prompt.md
claude -p --model fable  --output-format json --no-session-persistence \
      --permission-mode plan --allowedTools "Read,Grep,Glob,Bash" \
      --system-prompt "$SYSTEM" < prompt.md
```

The prompt asked each reviewer to check the fixes, to corrupt copies under `/tmp`, to prefer measured claims, and to say "no findings" where an area was clean. Fable's review ran against a tree that was being edited at the time, and said so: its first two findings were a compile error and a class-name test that were both already fixed when it reported them. Its later findings stand.
