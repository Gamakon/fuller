# Futhark GPU pipeline: "Migration graph sink could be reached from source" on a histogram module (2026-11-05)

**What happens.** `futhark rust-wgpu` (our fork) stops on one module of 24 histogram rows of about 1,000 nodes:

    Internal compiler error.
    Please report this at https://github.com/diku-dk/futhark/issues.
    Migration graph sink could be reached from source after it had been attempted routed.

**Where.** The size-bound measurement of plan step 5a,
`/private/tmp/claude-501/fbench/plan5a/size_bound/out_small2/histogram/900_1100/s1/prog.fut` (430 kB, 24 programs,
28.6 s to the error). Each row compiles alone on the C backend, which is why it looked like a fork fault: the C backend
does not run the pass. Compiled one program at a time on rust-wgpu, ONE fails alone, `kq20`: the reproducer beside this
note, `2026-11-05_futhark_migration_graph_sink.fut`.

**Which compilers.** Futhark 0.28.1's `futhark opencl` fails on `kq20` the same way, and so do the fork's `webgpu` and
`opencl`; `futhark c` compiles it. The fault is upstream's, in the GPU pipeline's "reduce device synchronizations" pass
(`src/Futhark/Optimise/ReduceDeviceSyncs/MigrationTable.hs`, the same on master on 2026-11-05 as in the fork's base).
No upstream issue names it.

## Reduced

    entry main (k: i64) (acc: [k]i32) (vs: []i32) : [k]i32 =
      loop acc for _i < k do
        let c = loop s = (loop _t = 0 for _l < k do 2) for _j < k do i32.min acc[0] s
        in replicate k (if c < 0 then i32.maximum vs else 0)

(`tests/corpus/futhark_compile_failures/minimal/migration_graph_sink.fut`). Two smaller relatives fail in the same
pass with other messages, and the fix below fixes them too: the inner loop alone returned (`Type error after pass
'reduce device synchronizations' ... Use of unknown variable main_res_N`), and it under an `if` whose branch reduces
(`Cannot migrate a host-only operation to device`).

## Cause

The pass builds a graph of the program's scalars and routes every device read (a source) to the host uses that need
it (sinks), a max-flow min-cut. For a loop, `graphLoop` then connects each operand of the loop body to the loop's
bindings it can reach (`connectOperand`, through `MG.reduce`), adds those edges and marks the operand's edges into the
loop exhausted. All operands were searched with ONE cache (`ReachableBindingsCache`, the one the `deviceBindings`
search left). `MG.reduce` resolves a cyclic reference to `mempty` and caches what it computed under that assumption, so
the cached result of a vertex on a cycle (here the inner loop's parameter `s` and its update `min acc[0] s`) is
complete only for the search that computed it. In the reduced program the first operand searched is the merge
argument (the constant loop's result, a graphed scalar), which reaches `s` and then the update; the update's cached
result is empty. The second operand, `acc[0]` (a device read), reaches the update, finds the empty cached result, gets
no edge to the loop's result `c`, and has its only edge exhausted. Later `c < 0` must be on the host (the `if` holds a
kernel), the routing cannot find a path from the read, and the check after it finds the sink by the exhausted edge.

## Fix

Patch 0016 (`futhark-rust-backend/patches/0016-fix-reduce-device-syncs-each-loop-operand-is-searche.patch`): each
operand is searched with a cache of its own (`mapM_ (connectOperand MG.none)`). Sound: within one search from one root
the cycle-cut results are only intermediate, and every vertex reachable from the root is explored, so the union of
bindings the root gets is complete; only a cached result reused for ANOTHER root could be short, and that reuse is what
the patch removes. The cost is one search a loop operand instead of a shared one. Programs whose operands never met a
short cached result get the same graph and the same code. Futhark test: `tests/migration/loop-operand-cycle.fut` in
the fork (its two cases pass on `c`; it compiles with `opencl` and no longer stops in the pass).

**Checked (CPU only).** With the rebuilt fork (`rust-wgpu @ fb53cf1`): the reduced program, its two relatives, `kq20`
and the whole 24-program module compile with `futhark rust-wgpu` (and `kq20` with `futhark opencl --library`). The
output of the fork against the 5f6c334 build over the corpus is in the README's rebuild note. Not run on a GPU.

**In our test suite.** `kq20` is in the compile-failure corpus as
`tests/corpus/futhark_compile_failures/migration_graph_sink__27a1a0bf4392.fut` (index kind `migration_graph_sink`),
with the reduced program in `minimal/`; the corpus tests require both to compile through the rust path and the webgpu
path. Not in `known_faults.tsv`.

## For the Futhark team

Futhark 0.28.1 and master, any GPU backend: the program under "Reduced" fails with `Migration graph sink could be
reached from source after it had been attempted routed`. In `graphLoop`, `foldM_ connectOperand rbc` shares one
`ReachableBindingsCache` across the loop's operands, but `MG.reduce` caches results computed with a cyclic reference
resolved to `mempty`, which are complete only for the search that made them; an operand that reaches a vertex on the
loop's parameter cycle through the shared cache gets no edge to the loop's binding and its entry edges exhausted.
Searching each operand with a fresh cache (`mapM_ (connectOperand MG.none)`) fixes it; the patch and a test are in
`futhark-rust-backend/patches/0016-*`.
