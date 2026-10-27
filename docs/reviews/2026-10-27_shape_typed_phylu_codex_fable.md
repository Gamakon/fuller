# Reviews of the shape kingdom on the typed core (phylu)

**Date:** 2026-10-27. Two reviewers, two rounds, every finding dispositioned
below. The change under review: the shape kingdom (fuller's
`kingdoms/futhark_shapes`) as a **third typed kingdom** in phylu's GPU
population engine — `phylu::evolve::shape_typed`, `examples/shape_fit.rs`,
the `wgsl_typed_stage` → `walk_stage` rename, and fuller's widened seam
(`Scorer`, `fitness_of`, `score_lenient`, `suite`).

## Why the shape kingdom is its own module

Every other typed kingdom phylu runs has a GPU **evaluator**: the device
decodes a gene into a node tree and evaluates it. The shape kingdom has none,
because its semantics are a Futhark program's, and a Futhark program's
arithmetic is not reproducible by a WGSL evaluator bit for bit (the GPU
contracts the multiply-add into an FMA where the interpreter does not —
measured: 758 of 4096 elements agree exactly, the rest differ by at most
3.7e-4).

So the device draws, varies and selects, and the host scores. The first
review round found that this had been **described** more strongly than it is
built, and that is finding 1 below: the compile is not on the scoring path
either, and the doc said it was. The doc now says what runs.

## The reviewers

| | |
|---|---|
| codex (`gpt-5.6-terra`) | 9 findings: 2 blocking, 7 minor |
| fable | 8 findings: 2 blocking, 6 minor, plus one design note |

## Every finding, and what was done

### Blocking

**1. (fable) The doc claimed more than the code does.**
`shape_typed.rs` said the device decodes and that the scoring "is a
compilation". Neither was true: the driver never called `dev.decode`, and
`Scorer::score_one` is a host `express` + interpreter + cost polynomial — no
compile anywhere on the scoring path. The only compile was the example's
`--measure` on the final best.

*Done.* The doc now says what runs where, exactly: the device draws, varies,
selects **and walks** (the walk's node count is compared with the host's
expression of the same gene); the host expresses, interprets and costs; the
compile happens only where the **measured** cost is wanted. The walk check is
now real: `ShapeRun::walk_check` runs the device's walk at generation 0 and
every `walk_check_every` generations, checks the device's own defects, and
compares the walk's node count with the host's expression. It agrees — that
is the invariant the check exists to protect.

**2. (codex) An empty suite makes everything correct.**
`--samples 0` was accepted; `evaluate` left `correct = true` with no input
sets to disagree about, so every expressible program entered the correct band.

*Done.* `Scorer::new` refuses an empty suite: *"the evaluation service needs
at least one input set: an empty suite decides nothing"*. A phylu test
(`an_empty_suite_is_refused`) pins it.

**3. (codex) The correct band was not strictly ordered.**
`traffic / reference_traffic` clamped to one, so every correct program dearer
than the reference tied with it at exactly `0.49` — and a less expensive but
still over-reference candidate could not replace it.

*Done.* The transform is now `traffic / (traffic + reference_traffic)`:
**strictly increasing** in the traffic (so no two correct programs tie for
want of precision, however far apart they are) and **bounded** by one (so the
band never crosses into the incorrect one). The reference, whose traffic is
its own, sits at half the band. The test now pins the strictness.

**4. (fable) The polish repeated itself and consumed rows it did not use.**
The polish saturated the all-time best every generation with no `seen` set, so
the same candidate was offered every generation; and `children.remove(0)` ran
before the improvement test, so a rejected candidate consumed the worst row.

*Done.* The polish now works on the **generation's** correct individuals
(`elites`, best first, up to `polish_elites` of them), keeps a `seen` set of
the population's genes and the ones it writes, and only removes a row **after**
a candidate has beaten it. The genome is read once and all the replaced rows
are written in one call.

### Minor

**5. (codex) `polish_elites` was only an on/off switch.**
*Done* — see 4: it is now the number of current-generation elites polished.

**6. (codex, fable) Equal width is not the whole layout invariant.**
The two layouts could drift in the root, the frame order, the per-type
head/tail/Dc offsets or the constant lengths without the width changing, and
the host would then score a different program from the one the device varies.

*Done.* `check_layout` asserts the width, the number of genes, the root, and
every type's frame (offsets and lengths) and constant length against fuller's
`GeneLayout`. It is a host-only function with a unit test
(`the_device_layout_and_the_host_layout_are_the_same_frame`).

**7. (codex) `elites >= pop` was accepted and then refused by `vary`.**
*Done.* `shape_of` computes the population and the elite count together: the
elites are capped below the population, and the offspring are always even (the
device mates them).

**8. (codex, fable) A polished row's provenance is now false.**
The device's parent buffer still says the row was cloned from the parent it
was cloned from, and the row keeps that child's cohort and wrapper.

*Done* — stated, in the driver's doc: nothing reads either today, and a
lineage added later must mark a polished row.

**9. (codex) `best` was computed before the polish.**
*Done* — the best is now read off the fitness the device will select on,
after the polish.

**10. (fable) The inexpressible gene sat at 1.0, where it tied with an
agreement of zero.**
*Done* — it is at 1.5 now, so the selection's weight (`pi/2` less the fitness)
nearly vanishes on it.

**11. (fable) The seed was written to the device and trusted.**
*Done* — the seed is read back and compared, as the WGSL driver does.

**12. (fable) `Bridge::load()` was called every generation.**
*Done* — loaded once, in the run's construction.

**13. (fable) `correct_ever` counted the elites again every generation.**
*Done* — it is `correct_sum`, and it says so.

**14. (fable) The stale "wgsl-walk-stage-dummy" buffer label.**
*Done* — it is `walk-stage-dummy`.

**15. (fable) The stale plan reference in a dated report.**
`fuller/docs/reports/2026-10-21_A1_post_expression_inventory.md:29`. Left
alone: it is a dated report of what was true then.

### The design note (fable, question 7)

> The reason given for no device evaluator is aimed at the wrong comparison:
> correctness is judged interpreter against interpreter, not against the
> compiled program, so a device evaluator would need to reproduce the
> interpreter's reduction order, and the only real obstacle is naga/Metal
> contracting `a*b+c`. What does need the host is the measured cost: compile
> plus run, which this driver never calls.

This is half right, and it changed the design.

**Right:** the driver did not call the measured tier at all, and the measured
cost is the *decision* cost in this plan — the compiled GPU program's, not the
polynomial's. So there was a hole: the search was ranking on a static
approximation of the thing that matters.

**Not right:** that the correctness could be judged on the device. The
correctness here is bit-exact **on purpose** — both sides are the interpreter,
and the compiled program may legitimately differ where there is a reduction
(Futhark's simplifier folds `0.0 + x` away). A device evaluator would have to
reproduce the interpreter's reduction order *and* avoid the FMA contraction,
which naga/Metal does not promise; the alternative is a tolerance, and the
whole design refuses tolerances. So the seam stays where it is.

**What was done:** the measured tier is now in the driver.
`ShapeConfig::measure_elites` asks for the top correct individuals of a
generation to be **compiled for the GPU and measured**, and the run's best is
then the cheapest of them by the measured cost. It is off by default (a
compile is expensive and it is not on the per-generation path), and the
example turns it on with `--measure-elites N`. The measured run:

    reference  map2 (add) (A.B) ((B'.A')')   13 device allocations, tiled
    winner     map2 (add) (A.B) (A.B)         9 device allocations, not tiled

The same winner, at the measured cost — the transposed-product rule rewrote
the second operand into the first and the compiler's own CSE removed the
duplicate.

## The measured run

`cargo run --release --features gpu --example shape_fit -- --pop 16 --gens 4
--size 6 --samples 2 --measure-elites 2`

    # shape reference   row 0   the incumbent, in a layout of 322 words, traffic 396
    # shape layout      program size N = 3, 11 types framed, row 322 words, 16 rows, 2 elites
    # shape polish      3 child rows replaced, traffic 396
    # shape measured    13 device allocations, 9488 bytes literal, tiled
    # shape gen 0       correct 1/16, best 0.15313, polished 3
    # shape measured    9 device allocations, 9056 bytes literal, not tiled
    # shape gen 1       correct 6/16, best 0.15313
    ...
    # shape best        correct true, agreement 1.0000, traffic 180, alloc 36, depth 6

The winner is the same one fuller's host-side loop found: the product computed
**once** instead of twice. The population was made on the GPU, the polish and
the scoring were made on the host, and the device selected on what the host
wrote.

## What the reviewers did not find

Neither reviewer found a defect in the *host seam's* reuse of fuller's
generic `express`/`write_back`/`typed_gep` — the gene really is a Karva-typed
word vector over the contract, and the walk and the host expression agree on
its node count. That agreement is now checked every run, not assumed.
