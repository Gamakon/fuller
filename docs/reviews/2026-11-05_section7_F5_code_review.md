# Code review: plan section 7 task F5 (master 6's e-graph) and the genic_cells fix

Commits `ef5c52cc` (typed_chromosome: genic_cells reads no Dv word as a symbol) and `208e23e7` (master 6's e-graph,
a sort a data type). Read only: `git show` and the files, no build, no test run. 2026-11-05.

## Verdict

**Accept, with one soundness fix required before the master 6 fold runs on a population (finding 1).** The variants
path (`Egraph6::variants`) is sound and is the same saturation as master 5. The fold's sharing across scopes is sound
on the path the test covers. It is NOT sound on the refit path: a refitted individual can take a shared part whose
common word names another variable there, and the check meant to catch this compares the form with itself.

What was checked and holds:

- **0.13 s against 244 s is the same saturation, no rule dropped.** Rows 1 to 26 of `master/6/rules.tsv` are
  byte-identical to `master/rules.tsv`. The 4 added `mat` rows are A7's matrix rules: master 5 kept them in a
  separate e-graph (`kingdoms/futhark/egraph.rules`, one copy for each shape class). Master 5's k-means view never had
  them. The bounds are the same (3 iterations, 32 members, 48 variants), and so is the ranking (cost, then text). The
  only behaviour difference: master 6 also skips a variant whose text equals the reference's own (master 5 skipped
  only an identical tree), and the fixture shows this changes nothing for k-means. The speed-up comes from the size of
  master 5's view (14,359 types: the plan's "Why" paragraph), not from fewer rules. The 244 s figure itself is not
  recorded with its conditions (finding 8).
- **Variables in the e-graph are resolved variables** (`Xvar__T depth ordinal`, `master6.rs:190`). This is sound
  because no rule names a binder (witness E7), so every rule is an identity under any environment, congruence
  included. A variable resolved to the same `(depth, ordinal)` under two different binders is the same e-node, and that
  is still sound for the same reason.
- **An extracted variable written back is in scope** (`place`, `master6.rs:231-279`). A resolved variable that is not
  in scope gives `None`, and the result is decoded again to confirm the variable. The test covers both: a packed
  `(1,0)` at pathfinder's root, where nothing is bound, gives `None`; words shifted by their count import identically.
- **`common_word`** (`saturated_fold.rs:100`) searches the whole congruence class below `DV_WORDS`, so it finds the
  least solution or correctly reports that none exists.
- **The k-means fixture cannot pass silently.** It has 31 entries, an empty fixture fails the live test, and write
  mode panics after writing.
- **The genic_cells fix is correct**: a Dv cell holds an index word, never a symbol.

## Findings

### 1. HIGH: the refit path can bind a shared part's variable to another variable, and the check that should catch it compares a form with itself

`src/saturated_fold.rs:589`, `:635`, `:644`, `:655`; `src/futhark_saturation/master6.rs:241`.

The common word of a shared part's variable is computed only from the occurrences in the extracted root forms
(`sites`, `:452`). In the refit branch (`write_back` of the extracted form failed), `refit` adopts the ADF reference
at positions of the individual's own tree `t`. At such a position the part may stand in an environment that no site
was counted for: for example, the part sits inside a subtree the extracted form dropped through
`fst (pair a b) -> a` or `map2_rows id a d -> a`. The refit form is then unfolded with `bodies_expanded`. Those are
the ADF bodies, already carrying the common word, which is an index word. `Egraph6::place` reads an index word
(`Word::Index`) as whatever variable it names *there* and writes that variable back. So `form` is built from the same
common word as the gene, and the comparison at `:655` (`import(back) != import(form)`) cannot differ.

Failure scenario: individuals A and B share class `c = tab[c] k (get zeros_k c)`. Its `c` is place 0 of 1 at every
counted site, so the common word is 0. Individual D refits, and its own tree holds `c` under a binder where the
`i64`s in scope are 2 and `(1,0)` is place 1. The ADF's word 0 names the other `i64`. `place` accepts it, the check
passes, and D's program silently computes something else. Nothing reports it.

Fix: in the refit branch, unfold with the bodies as RESOLVED trees (`expanded[c]`, packed words) and place that. The
placed form then names the right variable, so the existing comparison against the gene's expression detects the
mismatch, and the individual falls back. Better still, have `joint` refuse an ADF at a position whose `VariableSite`
for any body variable is not congruent with `words_of[c]`, so `c` is inlined there instead. Add a test with a refit
individual whose part stands at an uncounted site.

### 2. MEDIUM: a mismatch on the non-refit path is kept silently instead of reported as a bug

`src/saturated_fold.rs:655-665`.

On the non-refit path the common word is built so that `import(back) == import(form)`; a mismatch there is our
bug. The new branch keeps any individual whose form has variables (`kept += 1`) whichever path it came from. This
contradicts Rules.md ("no failovers", "no hiding bugs"), and the comment ("A refit put a shared part ...") describes
only the refit case.

Fix: keep the individual only when it was refitted, and only after finding 1's fix. On the direct path, return
`bug(...)`.

### 3. MEDIUM: no test exercises a part inlined for scope, or a refit in master 6

`src/futhark_saturation/master6/tests.rs:270`.

The only fold test asserts `unshared_by_scope == 0` and `kept == 0`. The case "two places bind different variables
under the same index word" (place 0 of 2 against place 1 of 2: no common word) is never built, so the inlining loop
(`:447-520`) and its fixed point run untested. Neither does any test reach the refit branch with variables (finding 1).

Fix: add (a) two individuals whose part's `c` is place 0 of 2 in one and place 1 of 2 in the other, asserting
`unshared_by_scope == 1` and bit-identical output on the c backend; (b) a refit case as in finding 1.

### 4. LOW: structural fate counts a part with no e-class as unreached instead of reporting a bug, for every kingdom

`src/saturated_fold.rs:516-520`.

Before this commit, a structural part with no e-class was `bug("a structural part has no e-class")`. Now every lookup
miss counts as `fate.unreached`, including a subtree of a saturable tree whose class is missing, which is a bug.
`class_of_subtree` is also keyed by the `Tree` (words), so in master 6 the same words under two environments take the
class of their first occurrence: the statistic is attributed to one environment.

Fix: record the subtrees of kept (non-saturable) trees separately. Count a miss as unreached only when the body occurs
only there, and return a bug otherwise.

### 5. LOW: "defaults unchanged for every other kingdom" is not quite true

`src/saturated_fold.rs:263`, `:655`.

(a) Distinct trees are now keyed by `import(t)`, not by `Tree`. (b) The final check now accepts `back != form` when the
two imports are equal; before, any difference was a bug. For a kingdom whose `import` is not one-to-one (it drops a
word the gene carries), a difference that was a bug now passes. (c) The scope loop (`:447-520`) clones a path at every
node of every root form, even when no kingdom variables exist. That is extra memory and time for WGSL-size trees.

Fix: apply the relaxed check and the scope loop only when `kingdom.variables` returned anything (all `variables_of`
empty means skip both).

### 6. LOW: the fold rewrites every master 6 index word to its least form

`src/saturated_fold.rs:607` (`vars.get(p).map(|v| v.place)`).

An individual whose form is unchanged still gets every Dv word rewritten to `place`. This counts in `rewritten`, which
then overstates what the fold changed, and it erases the neutral variation the Dv words carry: a word `w` and
`w mod count` are not equivalent after a later mutation changes the count.

Fix: keep the gene's own word wherever it is congruent to `place` (and to the common word inside an ADF). Otherwise
count Dv-only changes separately from `rewritten`.

### 7. LOW: rule admission drops a rule silently when a named row is missing

`kingdoms/futhark/gen_tables.py:3161` (and the same step in `src/futhark_view/egraph6.rs`).

A rule whose `needs` row is present but whose other named rows are missing (for example `id@fn.arr_f32_arr_f32`
renamed) is skipped with no message. A7 would lose its `map_rows` and `map2_rows` rules, and the Rust copy would agree
byte for byte, so `--check` passes. Today all 4 `mat` rules are present in A7 and k-means.

Fix: a witness E8. When a rule's `on`, `elems` and `needs` admit it for a data type, every row it names must exist,
or the generator fails.

### 8. LOW: the k-means fixture and the 244 s are not recorded with their run

`tests/fixtures/futhark_m6_kmeans_variants.txt`; `docs/PLAN_futhark_kingdom.md` F5 status.

The fixture's only check is `#[ignore]`d, so `cargo test` never compares it with master 5. Neither the commit nor the
plan records when `--ignored` last passed, on which commit, or its wall clock, and "244 s" does not say what it
includes (master 5's view load, its 14,359-type e-graph, or both). Rules.md asks for timestamped wall-clock figures.

Fix: add one line to the F5 status giving the UTC time, the commit, the command and the wall clock of the `--ignored`
run, and split 244 s into the view load and the saturation.

### 9. LOW: genic_cells excludes two domains instead of naming the two that hold symbols

`src/typed_chromosome.rs:576`.

`!matches!(s.domain, Domain::Dc | Domain::Dv)` will read a future data domain as symbols, which is the same gap again.
Separately, the test helper `random_words` (`:1100`) fills Dv cells with symbol ids, so random-gene tests never draw a
Dv word above the symbol count.

Fix: write `matches!(s.domain, Domain::Head | Domain::Tail)`, and draw Dv cells in `random_words` below `DV_WORDS`.

### 10. LOW: wording

The commit messages and comments use plan codes as if they were words: "a G2 gap", "plan 7.5 risk 3", "E1-E7",
"Dv word". Say what each is the first time it appears: "a gap in the gene layout task (G2)", "the risk that an index
word names different variables in different scopes".
