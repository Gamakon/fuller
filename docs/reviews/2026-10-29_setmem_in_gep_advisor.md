# SetMem in multi-typed GEP: an independent advisor's answer (2026-10-29)

Question: how would SetMem go into our multi-typed GEP; is it not just a homeotic gene? Checked on the CPU at 07:46Z (no GPU, no code changed): the logged failures, and Futhark's GPU memory IR of one.

## The six conclusions

1. **Memory stays the compiler's job in the Futhark kingdom.** Futhark source has no memory, so a gene has nothing to name. In the WGSL kingdom, where we write the memory ourselves, memory belongs in the genotype.
2. **A memory block is not a homeotic gene, and not an ADF either.** An ADF shares one value among many users. A block shares one piece of storage among different values whose lifetimes do not overlap.
3. **The types can carry consumption exactly.** Use a second array type per class for fresh arrays, and spell each consuming use inside a row. Lifetimes and reuse can be types only in a kingdom that owns its memory.
4. **The Karva walk and the operators do not change.** Fold must never make a shared definition of a fresh-array type.
5. **The failing programs are legal; our backend is wrong.** All 8 logged histogram failures compile with `futhark c`. The SetMem is Futhark's own aliasing of a loop's state inside a kernel, so no genotype change removes it.
6. **Fix the WebGPU generator, then test it on the 10 logged programs.** Turn an in-kernel memory parameter into a block number with a switch on every access. Put memory into the WGSL genotype only when a planted memory win needs it.

## 1. Genotype or compiler

What fails, measured. `/private/tmp/claude-501/fcheck/A1_histogram/bugs.log` (1,400 random genes, the rust path) charges 8 programs. (The message appears 23 times; batch halvings repeat it.) `fcheck/A4_pathfinder` charges 2 more. That is 8 in 1,400, not 1 in 6; the 1-in-6 figure may count batches. Seven of the 8 histogram programs have a loop with an array state inside a `tabulate` body. The smallest:

```
tabulate k (\j1 -> i32.minimum (loop (s2: [n]i32) = (hist ...) for t2 < n - 1 do zeros_n))
```

`futhark dev --gpu-mem` shows the cause inside the `segmap`. The loop's memory parameter starts in one existing device block (`mem_8860`, offset `gtid`, stride `k`) and moves to another (`mem_8845`, offset 0, stride 1). Nothing is allocated; the compiler only chose to alias. Patch 0001 renames a SetMem only at control depth 0, and patch 0002 handles register memory anywhere, so device memory under a kernel loop fails. All 8 programs compile with Futhark 0.28.1's C backend (9 s, `scratchpad/sm/`).

So the gene is legal and the fault is our backend's. Memory in this genotype would mean evolving Futhark's memory IR, the compiler's own output, which it would plan again anyway.

## 2. The homeotic analogy

Revision 4 (§6) keeps **homeotic** for the chromosome linker and calls population-wide shared rows **ADFs** (`population_dag.rs`: `is_adf`). The question's "homeotic row" is an ADF row.

| | ADF row (`population_dag.rs`) | memory block |
|---|---|---|
| what is shared | one value, hash-consed by structural equality | storage, between different values |
| when sharing is legal | always: equal text means equal value | when lifetimes do not overlap, which depends on execution order across the whole program |
| changes | never; its memo is keyed by node and data version | written; an in-place update changes its contents |
| order | acyclic by level | a loop's state goes back and forth between blocks, a cycle in time |
| unfold | inline it: always keeps the value | a fresh block for every array: always legal, the no-reuse default |

**It holds** in form: named, typed (element type and size), referenced by href, counted, with a value-keeping fold and unfold. **It breaks** on lifetimes, which an ADF lacks and which make memory hard. SetMem binds a value to a resource: register allocation, an attribute of a node, not a shared subtree.

## 3. Types that carry the rules

A Karva tree is linear: every subtree has one parent, so an array a function row makes has one consumer. Consumption is unsafe only where a value is shared (an input, a loop state, an ADF reference). So no counting is needed:

types.tsv:

```
ref	dual	slot	form	literals	root	meaning
T0002	a.n:c	arr	a.n		no	a [n]i32 that may be shared, read only
T0608	u.n:c	uarr	u.n		no	a fresh [n]i32 with one consumer, may be updated in place
T0609	a.k:c/W-n-k	arr	a.k		no	an update loop's state: readable, consumed by no row
```

functions.tsv:

```
ref	name	kind	out	args	spelling	note
F04391	tab@u.n:c	function	u.n:c	i32:c/T-n	tabulate n (\j1 -> {0})	fresh
F04392	read@a.n:c	function	a.n:c	u.n:c	{0}	a fresh array read as a shared one
F04393	upd@u.n:c	function	u.n:c	u.n:c;ix.n:c;i32:c	({0} with [{1}] = {2})	consumes its first argument
F04394	fresh@u.n:c	terminal	u.n:c		(replicate n 0i32)	the fresh-array terminal
F04395	loopupd.n@u.k:c	function	u.k:c	u.k:c;ix.k:c/W-n-k;i32:c/W-n-k	loop (s1: *[k]i32) = {0} for i1 < n do s1 with [{1}] = {2}	the consumption is in the spelling
F04396	s1@a.k:c/W-n-k	terminal	a.k:c/W-n-k		s1	readable, never consumed
```

The rules:
- Inputs (`is`, `vs`) are terminals of the `a` types only.
- The only `u` terminal is `fresh`.
- A consuming use of a name exists only inside a row's spelling, never as a gene terminal.
- The generator's witness tests check Futhark accepts every row.

These rows let the language write the sequential histogram (`loopupd.n fresh (clamp (get.n is i1)) (add …)`), which it cannot today. For WGSL, the proposal's `block`, `view` and `lifetime<p>` types are right; blocks are declarations, and reuse is a liveness pass, not a fold.

## 4. Expression, variation, fold

- **The Karva walk and the operators:** unchanged; `u` is just more types. Array classes double; the kingdom report gives the width.
- **Fold:** must not make an ADF of a `u` type. It cuts a shared fresh subtree at its `read` parent instead.
- **The contract:** a per-type "consumable" flag, and a defect in `Epoch::new` if a reference terminal of a consumable type is ever made.
- **The e-graph:** its rules must keep a `u` term a `u` term.

## 5. The immediate problem and the search

Neither change to the genotype fixes the failures: Futhark makes the SetMem from programs that never mention memory.

**The fix is in the generator.** A kernel loop's memory parameter can point only to a few blocks known at code generation (here two), and its offset and stride are already scalars. Make it a `u32` block number and every access a `switch` over those blocks: exact, a branch per access in those loops only.

Forbidding array-state loops under a parallel binder (contexts `L-*-a.*` under `T-*`) would hide our bug by removing legal programs; do not.

**Effect on the search.**
- Memory in the Futhark genotype: **worse** (wider genes, neutral cells, re-planned by the compiler).
- The uniqueness types enlarge the language at a cost in width; whether that helps must be measured.
- The backend fix makes fitness honest: today these legal programs score worst, a false cliff (fault 14).

## 6. Recommendation and the cheapest experiment

**Recommendation.** Fix the backend first; keep memory out of the Futhark genotype; add `u` types only when in-place rows are wanted; design WGSL memory as typed resources with lifetimes, not ADFs.

**Experiment.**
1. Patch 0007, the block-number switch.
2. Compile the 10 logged programs (A1 histogram 8, A4 pathfinder 2) through `rust-wgpu`. Expected: 10 of 10 compile, and naga validates them.
3. Outputs on the stored cases bit-identical to `futhark c` and the host interpreter (integer kingdom, no tolerance).
4. Replay the A1 check. Expected: zero SetMem bugs, every other row unchanged.

Cost: a fork rebuild, ten compiles, one short GPU session.
