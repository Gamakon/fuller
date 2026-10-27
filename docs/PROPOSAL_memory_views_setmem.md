# PROPOSAL: memory views and SetMem in our WGSL tools

## Context

**Objective.** Give our WGSL kingdom, and every tool around it, the memory techniques that Futhark's compiler treats as essential:
- one memory block standing for another (`SetMem`);
- zero-copy views of an array (transposes, slices, strides);
- reuse of a block once its contents are dead;
- ping-pong buffers that swap roles each step of a loop;
- in-place updates.

**Why.**
- Our kingdom has none of these. When the WGSL symbol table was designed, "pointers as values" went on the deferred list: every buffer has one fixed name and one fixed role for the whole kernel. That kept the race and ownership proofs simple, and nobody went back to it.
- The result is that evolution cannot discover tiling reuse, double-buffering or copy-free transposes. Those are exactly the changes that make GPU kernels genuinely faster.
- The volumepassage and ingest runs found no real speed-up, and this gap is a likely part of the reason.

**Approach.** Carry the technique in the types, as the design requires, so that every gene still expresses a legal kernel:
- **No runtime pointers.** WGSL cannot pass storage memory around as a value; it can only pass pointers to workgroup and private memory. So a "view" is compiled to index arithmetic, and a SetMem is resolved by renaming at render time.
- **The same idea as share-equivalence**, applied to memory instead of to computations.

**Status.** Proposal; nothing built.

## 1. What Futhark does, in our terms

| Futhark concept | What it means | Our equivalent, to add |
|---|---|---|
| memory block in a space (device, shared, private) | storage with a lifetime | a **block** type per space and element type |
| index function (LMAD: offset plus strides per dimension) | an array as a view of a block; transpose and slice cost nothing | a **view** type: (block, layout), with an affine layout |
| `SetMem x y` | from here on, x is y | a **rebind** row inside a phase, resolved at render time |
| loop memory parameters | a buffer that changes identity each iteration (ping-pong) | a **swap** row at the end of a uniform loop iteration |
| memory reuse and short-circuiting | a dead block's memory serves a new array, and a result is written where it will be consumed | a **reuse** row that takes a block whose last phase has ended |

## 2. The types (a new batch, 10, in `MultiTypedGEP_s9_WGSL_draft.md`)

1. **`block<space, e, size>`.** A declared buffer: a binding, a `var<workgroup>` array, or a private array, sized by the interface. It is declared once at its largest use.
2. **`view<space, e, shape, layout>`.** How code sees a block: an offset and strides per dimension, all fixed when the shader is compiled, or uniform. Every load and store row takes a view instead of a named array. Index arithmetic is the view's own text: `offset + i*stride_i + j*stride_j`, clamped as now.
3. **Phase-stamped ownership.** The ownership types (owned element, owned pixel, slot, pair, block, segment) attach to a view within a phase. Phases are separated by barriers, as now, so two views of one block cannot race inside a phase, by the same proof as today.
4. **`lifetime<p>`.** A block carries the index of the last phase that reads it, so a reuse row only accepts a block whose lifetime has ended. This is decided by the types, never checked afterwards.

## 3. The rows

| Row | Takes → gives | Spelled as | Legal because |
|---|---|---|---|
| `view.of` | block → view (identity layout) | nothing; the view is the block's own index | always |
| `view.transpose` | view [n][m] → view [m][n] | strides swapped in the index text | it is the same elements, read only |
| `view.slice` | view, uniform offset and length → view | the offset added in the index text | in range by the clamp |
| `view.stride` | view, uniform step → view | stride multiplied | in range by the clamp |
| `mem.rebind` (SetMem) | view a, view b, same shape and space → b refers to a for the rest of the phase | at render time, every use of b is spelled with a's block and layout; no runtime cost | b's phase-ownership is a's, so no two threads write one element |
| `mem.swap` (ping-pong) | two views of equal shape, at an L rung at the end of a uniform loop iteration → swapped | one block of double size, `base = (k & 1u) * N`, flipped by the loop counter's parity | only at uniform control after the barrier, so every thread swaps together |
| `mem.reuse` | a block whose lifetime has ended → a fresh view for a new array | the same declared memory, a new layout | the old contents are dead by the lifetime type |
| `mem.in_place` | the input view of an elementwise map whose input dies at that map → output into the same view | the store goes to the input's element | each thread reads then writes only its own element |

Nothing here needs a WGSL pointer at run time. If a later row wants a helper function over workgroup memory, WGSL allows `ptr<workgroup, …>` and `ptr<function, …>` parameters. That is spelled by a prologue helper, as other duplicating rows are now.

## 4. Every tool that uses WGSL

| Tool | Change |
|---|---|
| **The kingdom tables** (`kingdoms/wgsl_true`) | batch 10 rows and types in a new family module `views.py`; every load and store row moves to taking a view; the generator's kingdom check covers the new types |
| **The renderer** | resolves rebind at render time; spells layouts as index arithmetic; double-size blocks for swap |
| **The reader** (seeds into Karva) | reads seeds that alias, transpose by index or ping-pong, including Futhark's generated WGSL and the V24 ingest kernels that do it by hand (several of the 13 "shape gaps" are this) |
| **The e-graph** (`true_kingdom::egraph`, `true_opt`) | view algebra as exact rules: transpose of transpose is the identity, a copy followed by a read becomes a view, and slice composition. A new optimiser, **reuse**, applies liveness over phases to fold dead blocks into reuse, which is Futhark's short-circuiting as an in-loop step |
| **The share-equivalence** (`true_share`) | memory sharing joins computation sharing: two blocks never alive at once share one declaration |
| **Fitness** | a third measurement group alongside quality and speed: memory footprint per thread and per workgroup, so reuse that saves memory without saving time still counts |
| **The Futhark shim** (`futhark-wgpu-shim`) | the WebGPU back end's missing SetMem gets the same render-time resolution. Our rule, written as a patch to Futhark's WebGPU code generator and offered upstream, lets the A7 programs compile with their tiling kept |
| **phylu** (`true_wgsl_fit`) | nothing new: the rows are ordinary rows, and the optimisers run on the existing hooks |

## 5. Steps, each with its check

1. **Specify:** batch 10 in the s9 draft (types, rows, the legality argument per row), reviewed against `GOAL.md` and rev 4.
2. **Build the table:** generate the rows; 1,000 random genes per instance pass naga and Metal and run; 10,000 random genes never race. The race check runs every kernel twice in different thread orders.
3. **Reader and renderer:** volumepassage, nebula and the 50 readable ingest kernels still read exactly. Some of the 3 remaining ingest gaps and 13 shape gaps should now read; count them.
4. **E-graph rules and the reuse optimiser:** exact rules checked by identical output on the device.
5. **Planted wins, before any real run:**
   - a kernel that copies a transposed matrix and then reads it (the view should remove the copy);
   - a reduction with two buffers allocated every step (ping-pong should remove the allocations);
   - a tile loaded into a fresh array every iteration (reuse should fold it).
   Evolution must find each one within 30 minutes.
6. **Real runs:** volumepassage, then the ingest module, then the Futhark A7 programs through the patched shim. Each new best is re-timed against the original, alternating the two.
7. **The Futhark patch:** the SetMem rule in Futhark's WebGPU code generator, built from source, A7 through the shim matching the C back end, offered upstream.

## 6. Risks

- **Gene size:** every load and store gains a view argument, so heads and tails grow. The view rows are terminals in most programs (`view.of`), so the expressed size grows little; the layout width grows, measured in step 2.
- **Proof scope:** rebind and swap are only safe at uniform control and inside one phase. The rows exist only at those rungs, so the rule is carried by the types; a row that slipped outside them would be a table bug, which the race check catches.
- **Metal's compiler:** index arithmetic for views can block vectorised loads. This is measured in step 6, and where it costs speed, an exact rule that specialises an identity layout back to plain indexing fixes it.
