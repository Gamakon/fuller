# Final review: plan section 7 (data types, properties and scope), 2026-11-05

Reviewed independently of the advisor's review (read afterwards, section 6 below): the design
`docs/DESIGN_futhark_types_and_properties.md`, plan section 7 at `5cc9c96e`, and the code the tasks touch
(`src/geneframe.rs`, `src/kingdom_contract.rs`, `src/typed_gep.rs`, `src/futhark_table.rs`,
`kingdoms/futhark/master/rows.tsv`, `frames.tsv`, `types.tsv`; phylu `src/evolve/futhark_typed.rs`,
`src/futhark_eval/evaluator/start.rs`; the untracked `src/futhark_reader/kinds.rs`).

## Verdict

**Go. The design is sound typed GEP and the plan implements it faithfully in the right order. Four additions before
the agents start, none a change to the design.** The additions: (a) the static cost must charge every adapter, so the
cost axis removes them; (b) an e-graph rule is an identity only where no adapter sits between its two sides, so the
saturation runs on trees with no failing requirement, or a witness proves each rule holds under `fit`; (c) a loop
state or a `map2` operand whose length is `?` must have that length bound to a name before Futhark sees it, a fixture
for F2; (d) the index word should be read by clamping, not modulo, or the choice measured in G3 (section 4). The
three decisions of 7.5a: yes, yes, and give the file to F4 (section 3).

## 1. Is the design sound as typed GEP?

**Every gene expresses a legal program: yes.** Arity is fixed by the Argument rows, so Karva's frames are sized as
today (`heads_from_program_size`, `bounds` in `kingdom_contract.rs` read only counts per type; nothing in the design
changes that). A variable terminal always resolves: the (w mod c)-th variable, or the type's constant when none is in
scope, and the plan gives every data type a constant. A requirement that fails renders an adapter, not an error. The
one place legality is not by construction is Futhark's own size checker: the decoder says two lengths are equal when
their expressions match, and Futhark must agree. Where the decoder is more conservative than Futhark (two `?`
lengths) the cost is a needless `fit`; where it is less conservative (`let l = i64.min ...` feeding a `loop` state,
an existential size reaching `map2`) `futhark check` fails. Proof 1's 1,000 genes a problem, N of N, catches this on
F2's first day, which is where it belongs. Two spellings I would add as F2 fixtures now: a `loop` whose state length
is `?` (bind `let l = length s` before the loop, since a loop state's type must name its size), and `map2` over two
`?`-length operands.

**Does evolution keep useful structure? Mostly, with one thing to measure.** The adapters do not flatten the
landscape; they move genes that were unreachable today (excluded by the 14,359 types) into "legal but almost always
wrong". Under a bit-for-bit correctness fitness those genes score as failures, exactly as a wrong-but-legal gene does
today, so the landscape is no flatter than master 5's; it is larger, and the share of it that can win is smaller. The
search therefore spends evaluations master 5 never spent. Two things keep that in check and both belong in the plan:
the start population and the reference's variants are adapter-free (proof 2 guarantees it), and the static cost charges
adapters (`fit` is a `tabulate`, traffic of the target length; `clampix` is two compares), so a near-correct gene with
an adapter loses on the cost axis to the same gene without. F3's row says "the cost polynomial reads properties" and
should say this in so many words.

Index-word retargeting is the rugged part. Risk 1 names it: a binder added or removed above a node changes c for every
variable below, and with `w mod c` every one of them can retarget at once, so one head edit can rewrite the meaning of
a whole subtree's leaves. GEP-RNC lives with the same effect for constants, and the point mutation of index words gives
a smooth local move, so it is acceptable; but it is a choice, not a given. Clamping (`min(w, c-1)`, environment ordered
entry parameters first, binders outer to inner) keeps every variable that pointed at an outer name stable when an inner
binder appears, and sends the overflow to the innermost variable, which is usually the useful one. G3 owns the function
that reads the word; it should take the reading as a parameter and the lead should pick by the count of expressed
variables that change meaning under one head edit, on G3's test kingdom. Not a blocker.

**Does it fix the type explosion? Yes, and exactly as claimed.** The 14,359 k-means types are data types crossed with
sizes and scope (85 % depth-3 scope bodies, `rows.tsv`'s `C`, `V:vars(..)`, `D:dims`, `frames.tsv`); the design puts
sizes in properties and scope in the environment, leaving the data types, about 8. The gene width then follows from
`bounds` over 8 columns instead of 14,359, so the 27.6 million to about 31,000 words is arithmetic, not hope. The one
figure to watch is `n_max` per column: fewer types means more symbols share a column, and a column's tail is sized by
the most inputs of that type any symbol takes (`triple`, `psel`, `sel` take 3), so the positions a column may grow a
little over the design's 1,920; L1's table reports it.

## 2. Does the plan implement the design faithfully, in a sensible order?

Yes. Every item of the design's section 9 has a task: the six tables (G1), arity from Argument (G1), `var.T` with its
Dc-style word array (G2, named `Dv_T`), passes 2 to 4 (G3), master 6 and the views (F0, F1a, F1b), adapters and prelude
(F2), interpreter and cost (F3), reader (F4), e-graph (F5), phylu's layout, variation and start population (P1 to P3),
then the deletions (D1) and the other kingdoms only after Futhark works (7.7). The order is data model, Futhark proof,
other kingdoms, as Andrew asked. The one departure from the design's letter is the work bound (section 3), which the
plan flags rather than hides. G3's "one public environment function, used by F2, F3 and F4" is the right spine: the
meaning of an index word is the environment's order, and three readers of it must never rebuild it.

Two faithfulness gaps I would close in the task text:

1. **F5, the e-graph.** The design says the rules "rewrite rendered programs and need no change". That was true when a
   tree's meaning was its tree. Now a tree's meaning includes its properties, and a rule that is an identity when
   lengths match (`map2.add a b = map2.add b a`, `map2.min a a = a`) is still an identity under `fit` only because
   `fit` is applied symmetrically; a rule that reshapes a subtree (a projection through a tuple, a fused reduction)
   can move an adapter from one side to the other. Risk 3's binder witness covers scope, not properties. Either the
   saturation runs only on trees with no failing requirement (the elites, which is the e-graph's role in this
   kingdom anyway) or `gen_tables.py` gains a witness that each rule's two sides render to the same text with the
   adapters in. The first is one line in F5 and I recommend it.
2. **F3, the cost.** Say that `fit` and `clampix` are charged (section 1), so the regenerated `costs.tsv` carries it and
   the witness checks it.

The proof (7.3) is the right shape: legality N of N, references byte for byte, host and device agree word for word, the
gene size measured against the design's table, R0 rerun and then the 5a.4 regressions. The sizes (22 agent days, 11 on
the critical path) are plausible from the line counts; F2 and F4 are the two I would expect to run long, since the
renderer's binder names and the reader's `var.T` emission are the two places today's text must be reproduced byte for
byte from a different representation.

## 3. The three decisions for Andrew (7.5a)

1. **The work bound as a property and a requirement (`trips < depth` on the heavy rows): yes.** It is what master 5's
   `shallow` condition is for, the head length does not bound nesting, and the GPU rule (a runaway kernel cannot be
   killed) needs it. It is the design's own mechanism, the same move the design makes for SR's depth. One amendment,
   for the adapter rather than the rule: a failing heavy node renders as its data type's constant, which throws the
   whole subtree away. Where the node has an input of its own data type (`map2.*`, `scan.*`, `loop`'s state), render
   it as that input instead; the constant only when it has none (`sum`, `tabulate`). More of the gene stays
   expressed, and it is the substitution the design's section 8 already describes for SR. If Andrew wants the plain
   constant, the plan as written is fine.
2. **`i64.max 1 (...)` for a gene-computed size: yes, and write it into the design's section 5 table as the third
   adapter.** It is of the same kind as `fit` and `clampix` (it makes a property hold at run time), and `clampix`'s
   correctness depends on it; a design that lists the adapters should list all three. One line in the design, no change
   of meaning.
3. **`src/futhark_reader/kinds.rs`: do not delete it; give it to F4.** It is not declared as a module (`futhark_reader.rs`
   has `mod tests;` only), so it compiles into nothing and the tree is green with or without it. Its header says what it
   is: a kind inference over the normalised Futhark expression (i32, i64, f32, bool, arrays, tuples, functions, with the
   prelude's signatures and Futhark's integer-literal default). The advisor reads it as made unnecessary by the `i32` +
   `f32` view loading, and the i32-to-f32 rewrites in it are; but a reader that must emit `var.T` with the right data
   type for every name and every literal needs exactly a kind inference, and F4 has 1.5 days. Recommendation: the lead
   moves it out of `src/` now (to F4's scratch, not `git stash`), F4 reads it on day one and either wires it
   (`mod kinds;`, its tests) or writes its own and the file is deleted at F4's commit. Andrew need only agree that F4
   owns it.

## 4. The single biggest risk, and the guard

**The search gets worse while every proof passes.** Proofs 1 to 4 can all hold (legal N of N, references byte for
byte, host and device agree, gene 890 times smaller) and the evolution still find less than master 5 in the same
budget, because the smaller gene is bought with a search space most of which renders adapters, and the only proof that
sees this (proof 5, R0 and the 5a.4 regressions) is at the end of the critical path, after about 11 days.

The guard is to measure it early and to give selection the means to remove it:

1. **At F2, not L1:** for each problem, over the 1,000 random genes, print the share of genes with no adapter and the
   share of nodes with one. Master 5 is 100 % adapter-free by construction, so the number has a reference. If the
   adapter-free share of random genes is below a few per cent on k-means, the lead knows on day 7, not day 22, and the
   remedies (a tighter `view.tsv`, the cost charge, clamping over modulo) are all cheap then.
2. **The cost charges adapters** (sections 1 and 2), so an elite with a needless `fit` is beaten by its own clean form.
3. **Proof 5 is the gate for 7.7**, as the plan says; add that the k-means regression reports the adapter share of the
   population per generation beside its best and median, so the plot shows selection removing adapters, or not.

A second risk worth one sentence: the plan's binder names must reproduce today's text byte for byte (risk 2); if they
cannot, do not bend the design to the old names, re-publish the reference texts and re-key the quality caches. The
texts are a regression anchor, not a contract.

## 5. Smaller points

- 7.1's note that an index word is any `u32` below 2^16 and the decoder reads it modulo c means a point mutation of
  an index word is a uniform redraw of the variable; drawing words in a small range (say below 16) gives the same
  behaviour with smaller numbers in the stream and the checkpoints. Cosmetic.
- G1's check "rules and requirements name properties that exist" should also check that every Requirement's two sides
  are properties of the same kind (a length against a length), or the decoder compares a bound with a size.
- The kingdoms that already work (SR, regex) must derive today's Arity row for row from Argument rows (G1's test).
  Good; add the WGSL contract to that test, since it is the next kingdom and has the most argument positions.

## 6. The advisor's review (`2026-11-05_section7_advisor_review.md`), agree and differ

**Agree** with all nine, and they are applied in `5cc9c96e`: the work bound kept and named for what it is; sizes as
`i64` arguments producing `?`, rendered `i64.max 1`; proof 1 runs the programs, not only `futhark check`; master 6
beside master 5 until D1 and phylu built after every fuller task; F1 split so the critical path runs through G3;
phylu's rings and operators over Dv and the frame-table readers at the new stride; one environment function and
non-shadowing binder names; the e-graph's binder witness and two-process determinism; the side measurement removed.
The `typed_environment.rs` name and the merge note are right.

**Differ** on three points:

1. **`kinds.rs`.** The advisor says the design makes it unnecessary; I say the i32-to-f32 rewrites in it are
   unnecessary but the kind inference is what F4 needs (section 3, item 3). Give it to F4 rather than deleting it.
2. **The e-graph is covered by the binder witness.** The advisor's witness is about scope (a rule never moves a subtree
   under another binder) and is right as far as it goes; it does not cover properties (a rule is an identity only
   where no adapter sits between its sides). Section 2, item 1.
3. **The adapter share is "free, a counter printed by proof 1, no separate study".** It is free, and it should stay a
   counter; but it is the earliest signal of the biggest risk (section 4), so it is printed at F2 and again per
   generation in the k-means regression, not read once at L1. That is still no study.

One addition the advisor does not make: the cost charges adapters (sections 1, 2, 4). Without it selection has no
reason to prefer a clean gene over the same gene with a `fit`, and the adapter share has no force pushing it down.

The adversarial review (`2026-11-05_section7_adversarial_review.md`) was not present when this was written.
