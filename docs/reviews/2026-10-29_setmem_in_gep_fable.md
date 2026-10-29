# Review: SetMem in multi-typed GEP — is a memory block a homeotic gene?

Independent answer (Fable, 2026-10-29). Read in full: `docs/MultiTypedGEP.html`, `docs/GOAL.md`,
`docs/PROPOSAL_memory_views_setmem.md`, `docs/PLAN_population_dag.md`, `docs/PLAN_futhark_rust_wgpu.md`,
`src/homeotic.rs`, `src/population_dag.rs`, `src/typed_gep.rs`, `src/kingdom_contract.rs`, `src/futhark_sig.rs`,
`kingdoms/futhark_shapes/sig/*/{types,functions,interface}.tsv`, `futhark-rust-backend/` (README, six patches), and
the failing programs in `/private/tmp/claude-501/fcheck/A1_histogram/bugs.log`.

## 1. Memory stays the compiler's job

**Conclusion: keep SetMem, aliasing and reuse out of the genotype; the gene already decides memory at source level.**

The kingdom expresses source Futhark, which has value semantics. Which arrays are loop-carried, where an update is in
place, where a `copy` is forced: those ARE in the gene (`loop.k@a.n` carries an array state). Blocks, double buffers
and the `SetMem` that rebinds a loop's memory parameter are derived from that by Futhark's memory pipeline, which is
correct and tested. The failing programs are not wrong genes: every one I read is a legal `loop (s1: [n]i32) = … for
i < D do body` whose body is an array in another block, nested under a `hist` or `tabulate` so the loop runs
sequentially inside a kernel thread. Futhark emits `SetMem mem_param_ext <- color` at the end of each iteration, and
OUR WGSL generator (patch 0001) resolves a SetMem only at control depth 0. The bug is in the backend; a genotype
cannot fix a backend.

Memory in the gene would also make two layers decide one thing: the gene says "reuse this block", Futhark's memory
pass still allocates, double-buffers and merges as it likes, so the gene's memory words are non-coding at best and
contradicted at worst.

## 2. A memory block is not a homeotic gene, on either reading

**Conclusion: the analogy holds for naming and sharing a declaration, and breaks on value, time and consumption.**

The design reserves "homeotic gene" for Ferreira's typed linker (`MultiTypedGEP.html` §6, `homeotic.rs` header); a
block is not a linker. The intended reading is the population-wide shared definition, the ADF row of
`population_dag.rs`: an immutable expressed tree whose value depends only on its inputs, memoised by (node, data
version, type, input binding), referrers counted, references commuting. A block is storage whose contents depend on
statement order. `SetMem x y` means "from HERE ON x is y": a statement about time, and the DAG's memo key has no
time.

Where it holds: both are "find the same thing, declare it once, refer to it by a stable id"; the PROPOSAL's
share-equivalence of blocks is the same move as share-equivalence of subtrees.

Where it breaks:
- a loop's state: the memory parameter's identity flips every iteration (that IS the in-loop SetMem); an ADF has
  one identity for the population;
- in-place update and uniqueness: `with` consumes its array, and a shared value cannot be consumed. The referrer
  count that makes a subtree worth folding (`is_adf`: `uses >= 2`) is exactly what makes a block un-updatable. The
  analogy inverts;
- lifetimes: reuse is legal by the block's LAST reader, elsewhere in the tree; a definition is legal by itself.

## 3. What the types can carry, concretely

**Conclusion: consumption is carried by putting the in-place update in a binder row's SPELLING; linear single use
cannot be an output type.**

The contract gives a symbol one output type and an ordered signature, nothing more. A unique array `a.n!` read in
two slots of one body is consumed twice, and the walk cannot stop a terminal appearing twice, so "consumed once" is
not a symbol-table property. What works is the shape kingdom's own trick (the `const` and loop rows): the row spells
the consumption and the body never names the unique thing. In the sig family's style, histogram instance, `/U-n-a.n`
the body of an in-place loop over `[n]`:

```
types.tsv
ref     dual               slot  form  literals  root  meaning
T0610   a.n:c/U-n-a.n      arr   a.n             no    a [n]i32 in the body of an in-place loop over [n] carrying s1 (read-only)
T0611   i32:c/U-n-a.n      i32   i32   0i32,...  no    an i32 in that body
T0612   ix.n:c/U-n-a.n     ix    ix.n            no    an index in range for [n] in that body (i1 bound)

functions.tsv
ref     name               kind      out             args                                spelling
F04390  upd.n@a.n:c        function  a.n:c           a.n:c;ix.n:c/U-n-a.n;i32:c/U-n-a.n  loop (s1: *[n]i32) = copy {0} for i1 < n do s1 with [{1}] = {2}
F04391  copy@a.n:c         function  a.n:c           a.n:c                               copy {0}
F04392  s1@a.n:c/U-n-a.n   terminal  a.n:c/U-n-a.n                                       s1
F04393  i1@ix.n:c/U-n-a.n  terminal  ix.n:c/U-n-a.n                                      i1
```

Legal by construction: the state is unique because the row copies its initial value; the value slot is a scalar, so
it cannot alias `s1`; the body reads `s1` freely (the value is evaluated before the consumption); the index is in
range by its type. No row's signature asks for a unique array, so no gene spells a double consumption and Futhark's
checker has nothing to refuse. `U` nests like `L`: histogram's 607 types and 4,389 rows grow by about a third.

Block, view and lifetime types (the PROPOSAL's batch 10) belong to the WGSL kingdom, which has no compiler to derive
them. There a lifetime must be a context suffix of the dual, as `L-k-a.n` is now, because a reuse row's legality
depends on a phase elsewhere in the tree; that multiplies types by the phase count, and it is the only way the rule
can be local to a slot.

## 4. Expression, variation and fold

**Conclusion: with memory as source-level rows nothing changes in the walk or the operators, and fold stays sound.**

`upd.n` is an ordinary row with a context, like `loop.k`: the FIFO walk, `alphabet`, `swap_range`, `write_back` and
`promotions` are untouched. Fold stays correct: `a with [i] = v` is a pure function of `a` in source, so two equal
`upd` subtrees ARE one value; the compiler later gives them one block or two. Only if blocks were genotype would fold
change: a rebind's meaning is per program and per statement order, so an ADF body could not hold one (as it cannot
hold a genic terminal), and the memo key would need time. Another reason not to.

## 5. The immediate problem, and the search

**Conclusion: the genotype does not help the failing programs at all; the backend fix does, and it is small.**

Measured in `fcheck/A1_histogram` (program size 8, `--eval rust`, 1,529 programs): 5 at fault for "SetMem inside a
loop or conditional", 3 other failures; the studies' worst run had 13 of 1,400. I found no run at one in six;
whatever run gave that figure, every failing program has the one shape above. Random search hits it six times more
than evolution (29 against 5) because those programs are nonsense (the loop state is never read). The cost is not
the rate: the shape penalised, an array-carrying loop, is pathfinder's reference shape, so scoring it worst biases
that search.

The fix is patch 0007 in the fork: in a kernel every block is a module-scope binding, so the blocks a memory variable
can stand for are a finite set known at code generation. Give the variable a `u32` tag, make `SetMem x y` the
assignment `x_tag = TAG_y`, and spell every access through `x` as a `switch` on the tag. Exact at any control depth;
patch 0001's rename is the depth-0 case and patch 0002's value copy the register case. Or run Futhark's
double-buffering copy for kernel loops. A day in `ImpGen/WebGPU.hs`.

Memory in the genotype would make the search worse: wider genes (the PROPOSAL's own risk), a cost axis the fitness
already reads from the compiled program (`futhark_shapes_gpu`: allocations, shared memory), and a compiler free to
undo the gene's choice.

## 6. Recommendation and the cheapest test

**Conclusion: fix the backend, add in-place rows to the kingdom, measure allocations; build no memory genotype.**

1. Patch 0007. Test: the 5 histogram and 5 pathfinder programs at fault compile, run on the GPU and equal `futhark c`
   and the interpreter bit for bit; re-run A1 and count bugs 0. Half a day.
2. `upd.n`/`upd.k` and `copy` rows in `gen_tables.py` (its witness tests re-check them); 1,000 random genes accepted
   by Futhark; one planted win, the histogram reference as an in-place loop against `hist`, scored on device
   allocations and GPU time. That measures whether source-level memory rows give evolution anything, with no
   genotype memory. One day.
3. Only if (2) wins and the WGSL kingdom needs it: the PROPOSAL's `mem.in_place` row alone, on its planted win 3,
   gene width measured, before any block, view or lifetime type.
